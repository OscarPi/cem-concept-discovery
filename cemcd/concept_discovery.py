from pathlib import Path
import yaml
import wandb
from tqdm import tqdm, trange
import sklearn.metrics
import numpy as np
import torch
import lightning
import cemcd.turtle as turtle

def calculate_embeddings(model, dl):
    trainer = lightning.Trainer()
    results = trainer.predict(model, dl)

    c_pred = np.concatenate(
        list(map(lambda x: x[0].detach().cpu().numpy(), results)),
        axis=0)

    c_embs = np.concatenate(
        list(map(lambda x: x[2].detach().cpu().numpy(), results)),
        axis=0)
    c_embs = np.reshape(c_embs, (c_embs.shape[0], -1, model.embedding_size))

    y_pred = np.concatenate(
        list(map(lambda x: x[1].detach().cpu().numpy(), results)),
        axis=0)

    return c_pred, c_embs, y_pred

def fill_in_discovered_concept_labels(datasets, labels, max_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xs = datasets[0].train_x
    for dataset in datasets[1:]:
        xs = torch.concat((xs, dataset.train_x), dim=1)
    non_nan_xs = xs[np.logical_not(np.isnan(labels))]
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            non_nan_xs,
            torch.from_numpy(labels[np.logical_not(np.isnan(labels))].astype(np.float32))),
        batch_size=1024
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(non_nan_xs.shape[1], 1),
        torch.nn.Sigmoid()
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    loss_fn = torch.nn.BCELoss()
    
    model.to(device)
    model.train()
    for _ in range(max_epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred.squeeze(), y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
    model.eval()
    with torch.no_grad():
        full_labels = model(xs.to(device)).detach().cpu().numpy()
    return full_labels.squeeze() > 0.5

def match_to_concept_bank(labels, dataset, chosen_concept_bank_idxs=None):
    if chosen_concept_bank_idxs is None:
        chosen_concept_bank_idxs = range(dataset.concept_bank.shape[1])
    not_nan = np.logical_not(np.isnan(labels))
    best_roc_auc = 0
    best_roc_auc_idx = None
    for i in chosen_concept_bank_idxs:
        if np.all(dataset.concept_bank[:, i][not_nan] == 0) or np.all(dataset.concept_bank[:, i][not_nan] == 1):
            continue
        auc = sklearn.metrics.roc_auc_score(
            dataset.concept_bank[:, i][not_nan],
            labels[not_nan])
        if auc > best_roc_auc:
            best_roc_auc = auc
            best_roc_auc_idx = i

    return best_roc_auc, best_roc_auc_idx

def discover_concepts(config, save_path, initial_models, datasets):
    save_path = Path(save_path)

    train_dataset_size = len(datasets[0].train_dl().dataset)
    test_dataset_size = len(datasets[0].test_dl().dataset)

    predictions = []
    embeddings = []
    for dataset, model in zip(datasets, initial_models):
        c_pred, c_embs, _ = calculate_embeddings(model, dataset.train_dl())
        predictions.append(c_pred)
        embeddings.append(c_embs)
    predictions = np.stack(predictions, axis=0)

    discovered_concept_labels = np.zeros((train_dataset_size, 0))
    discovered_concept_train_ground_truth = np.zeros((train_dataset_size, 0))
    discovered_concept_test_ground_truth = np.zeros((test_dataset_size, 0))
    discovered_concept_semantics = []
    discovered_concept_roc_aucs = []
    n_discovered_concepts = 0
    did_not_match = 0
    n_duplicates = 0

    for concept_idx in trange(initial_models[0].n_concepts):
        for concept_on in (True, False):
            if concept_on:
                sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] > 0.5, axis=0)
            else:
                sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] < 0.5, axis=0)

            Zs = []
            for e in embeddings:
                Zs.append(e[:, concept_idx][sample_filter])

            best_score = - len(Zs)
            best_n_clusters = None
            for n in range(config["min_n_clusters"], config["max_n_clusters"] + 1):
                cluster_labels, _ = turtle.run_turtle(
                    Zs=Zs, k=n, warm_start=config["warm_start"], epochs=config["turtle_epochs"])
                score = 0
                for Z in Zs:
                    score += sklearn.metrics.silhouette_score(Z, cluster_labels)
                
                print(f"n={n}, score={score}")
                if score > best_score:
                    best_score = score
                    best_n_clusters = n

            cluster_labels, _ = turtle.run_turtle(
                Zs=Zs, k=best_n_clusters, warm_start=config["warm_start"], epochs=config["turtle_epochs"])
            clusters = np.unique(cluster_labels)

            for cluster in clusters:
                labels = np.repeat(np.nan, train_dataset_size)
                labels[sample_filter] = cluster_labels == cluster

                if np.sum(labels == 1) < config["minimum_cluster_size"] * train_dataset_size or np.sum(labels == 0) < config["minimum_cluster_size"] * train_dataset_size:
                    continue

                labels = fill_in_discovered_concept_labels(datasets, labels, config["max_epochs"])
                roc_auc, matching_concept_idx = match_to_concept_bank(labels, datasets[0])

                if roc_auc < config["match_threshold"]:
                    did_not_match += 1
                    continue

                discovered_concept_name = datasets[0].concept_names[matching_concept_idx]
                if discovered_concept_name[:4] == "NOT ":
                    not_discovered_concept_name = discovered_concept_name[4:]
                else:
                    not_discovered_concept_name = "NOT " + discovered_concept_name
                if (discovered_concept_name in discovered_concept_semantics
                    or not_discovered_concept_name in discovered_concept_semantics):
                    n_duplicates += 1
                    continue

                discovered_concept_labels = np.concatenate(
                    (discovered_concept_labels, np.expand_dims(labels, axis=1)),
                    axis=1)
                discovered_concept_train_ground_truth = np.concatenate(
                    (discovered_concept_train_ground_truth, np.expand_dims(datasets[0].concept_bank[:, matching_concept_idx], axis=1)),
                    axis=1)
                discovered_concept_test_ground_truth = np.concatenate(
                    (discovered_concept_test_ground_truth, np.expand_dims(datasets[0].concept_test_ground_truth[:, matching_concept_idx], axis=1)),
                    axis=1)
                discovered_concept_semantics.append(discovered_concept_name)
                discovered_concept_roc_aucs.append(roc_auc)
                n_discovered_concepts += 1
                if n_discovered_concepts == config["max_concepts_to_discover"]:
                    break
            else:
                continue
            break
        else:
            continue
        break

    np.savez(save_path / "discovered_concepts.npz",
        discovered_concept_labels=discovered_concept_labels,
        discovered_concept_train_ground_truth=discovered_concept_train_ground_truth,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)

    with (save_path / "results.yaml").open("a") as f:
        yaml.safe_dump({
            "n_discovered_concepts": int(n_discovered_concepts),
            "n_duplicates": int(n_duplicates),
            "did_not_match": int(did_not_match),
            "discovered_concept_semantics": list(map(str, discovered_concept_semantics)),
            "discovered_concept_roc_aucs": list(map(float, discovered_concept_roc_aucs)),
        }, f)

    if config["use_wandb"]:
        wandb.log({
            "n_discovered_concepts": n_discovered_concepts,
            "n_duplicates": n_duplicates,
            "did_not_match": did_not_match,
            "discovered_concept_semantics": discovered_concept_semantics,
            "discovered_concept_roc_aucs": discovered_concept_roc_aucs,
        })

    return discovered_concept_labels, discovered_concept_train_ground_truth, discovered_concept_test_ground_truth, discovered_concept_roc_aucs

def split_concepts(config, save_path, initial_models, datasets, concepts_to_split):
    save_path = Path(save_path)

    train_dataset_size = len(datasets[0].train_dl().dataset)
    test_dataset_size = len(datasets[0].test_dl().dataset)

    predictions = []
    embeddings = []
    for dataset, model in zip(datasets, initial_models):
        c_pred, c_embs, _ = calculate_embeddings(model, dataset.train_dl())
        predictions.append(c_pred)
        embeddings.append(c_embs)
    predictions = np.stack(predictions, axis=0)

    discovered_concept_labels = np.zeros((train_dataset_size, 0))
    discovered_concept_train_ground_truth = np.zeros((train_dataset_size, 0))
    discovered_concept_test_ground_truth = np.zeros((test_dataset_size, 0))
    discovered_concept_semantics = []
    discovered_concept_roc_aucs = []
    # turtle_classifiers = []
    n_discovered_subconcepts = [0] * predictions.shape[2]

    for concept_idx in tqdm(concepts_to_split):
        sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] > 0.5, axis=0)

        Zs = []
        for e in embeddings:
            Zs.append(e[:, concept_idx][sample_filter])

        best_score = - len(Zs)
        best_n_clusters = None
        for n in range(config["min_n_clusters"], config["max_n_clusters"] + 1):
            cluster_labels, _, _ = turtle.run_turtle(
                Zs=Zs, k=n, warm_start=config["warm_start"], epochs=config["turtle_epochs"])
            score = 0
            for Z in Zs:
                score += sklearn.metrics.silhouette_score(Z, cluster_labels)

            print(f"n={n}, score={score}")
            if score > best_score:
                best_score = score
                best_n_clusters = n

        cluster_labels, _, _ = turtle.run_turtle(
            Zs=Zs, k=best_n_clusters, warm_start=config["warm_start"], epochs=config["turtle_epochs"])
        clusters = np.unique(cluster_labels)

        for cluster in clusters:
            labels = np.repeat(0, train_dataset_size)
            labels[sample_filter] = cluster_labels == cluster

            roc_auc, matching_concept_idx = match_to_concept_bank(labels, datasets[0], datasets[0].sub_concept_map[concept_idx])

            discovered_concept_name = datasets[0].concept_names[matching_concept_idx]

            discovered_concept_labels = np.concatenate(
                (discovered_concept_labels, np.expand_dims(labels, axis=1)),
                axis=1)
            discovered_concept_train_ground_truth = np.concatenate(
                (discovered_concept_train_ground_truth, np.expand_dims(datasets[0].concept_bank[:, matching_concept_idx], axis=1)),
                axis=1)
            discovered_concept_test_ground_truth = np.concatenate(
                (discovered_concept_test_ground_truth, np.expand_dims(datasets[0].concept_test_ground_truth[:, matching_concept_idx], axis=1)),
                axis=1)
            discovered_concept_semantics.append(discovered_concept_name)
            discovered_concept_roc_aucs.append(roc_auc)
            n_discovered_subconcepts[concept_idx] += 1

    np.savez(save_path / "discovered_concepts.npz",
        discovered_concept_labels=discovered_concept_labels,
        discovered_concept_train_ground_truth=discovered_concept_train_ground_truth,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)

    with (save_path / "results.yaml").open("a") as f:
        yaml.safe_dump({
            "n_discovered_subconcepts": n_discovered_subconcepts,
            "discovered_concept_semantics": list(map(str, discovered_concept_semantics)),
            "discovered_concept_roc_aucs": list(map(float, discovered_concept_roc_aucs)),
        }, f)

    if config["use_wandb"]:
        wandb.log({
            "n_discovered_subconcepts": n_discovered_subconcepts,
            "discovered_concept_semantics": discovered_concept_semantics,
            "discovered_concept_roc_aucs": discovered_concept_roc_aucs,
        })

    return (discovered_concept_labels,
            discovered_concept_train_ground_truth,
            discovered_concept_test_ground_truth,
            discovered_concept_roc_aucs,
            n_discovered_subconcepts)
