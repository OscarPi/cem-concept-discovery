import numpy as np
import lightning
import pickle
from pathlib import Path
import sklearn.metrics
import wandb
import cemcd.turtle as turtle
import sklearn
from tqdm import tqdm
import yaml
import torch
from cemcd.training import train_cem

def calculate_embeddings(model, dl):
    trainer = lightning.Trainer()
    results = trainer.predict(model, dl)

    c_pred = np.concatenate(
        list(map(lambda x: x[0].detach().cpu().numpy(), results)),
        axis=0,
    )

    c_embs = np.concatenate(
        list(map(lambda x: x[2].detach().cpu().numpy(), results)),
        axis=0,
    )
    c_embs = np.reshape(c_embs, (c_embs.shape[0], -1, model.embedding_size))

    y_pred = np.concatenate(
        list(map(lambda x: x[1].detach().cpu().numpy(), results)),
        axis=0,
    )

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

    return full_labels.squeeze() >= 0.5

def majority_vote(xs):
    not_nan = xs[np.logical_not(np.isnan(xs))]
    if not_nan.size == 0:
        return np.nan
    return np.bincount(not_nan.astype(int)).argmax().astype(float)

def match_to_concept_bank(labels, dataset, save_path, max_epochs):
    train_dataset_size = len(dataset.train_dl().dataset)
    test_dataset_size = len(dataset.test_dl().dataset)

    deduplicated_discovered_concept_labels = np.zeros((train_dataset_size, 0))
    discovered_concept_test_ground_truth = np.zeros((test_dataset_size, 0))
    discovered_concept_train_ground_truth = np.zeros((train_dataset_size, 0))
    discovered_concept_semantics = []
    discovered_concept_roc_aucs = []

    for idx in range(labels.shape[1]):
        not_nan = np.logical_not(np.isnan(labels[:, idx]))
        best_roc_auc = 0
        best_roc_auc_idx = None
        for j in range(dataset.concept_bank.shape[1]):
            if np.all(dataset.concept_bank[:, j][not_nan] == 0) or np.all(dataset.concept_bank[:, j][not_nan] == 1):
                continue
            auc = sklearn.metrics.roc_auc_score(
                dataset.concept_bank[:, j][not_nan],
                labels[:, idx][not_nan],
            )
            if auc > best_roc_auc:
                best_roc_auc = auc
                best_roc_auc_idx = j

        if dataset.concept_names[best_roc_auc_idx] not in discovered_concept_semantics:
            discovered_concept_test_ground_truth = np.concatenate(
                (discovered_concept_test_ground_truth, np.expand_dims(dataset.concept_test_ground_truth[:, best_roc_auc_idx], axis=1)),
                axis=1
            )
            discovered_concept_train_ground_truth = np.concatenate(
                (discovered_concept_train_ground_truth, np.expand_dims(dataset.concept_bank[:, best_roc_auc_idx], axis=1)),
                axis=1
            )
            deduplicated_discovered_concept_labels = np.concatenate(
                (deduplicated_discovered_concept_labels, np.expand_dims(labels[:, idx], axis=1)),
                axis=1
            )
            discovered_concept_semantics.append(dataset.concept_names[best_roc_auc_idx])
            discovered_concept_roc_aucs.append(best_roc_auc)

    return deduplicated_discovered_concept_labels, \
        discovered_concept_test_ground_truth, \
        discovered_concept_train_ground_truth, \
        discovered_concept_semantics, \
        discovered_concept_roc_aucs

def discover_concepts(config, save_path, initial_models, datasets):
    save_path = Path(save_path)

    train_dataset_size = len(datasets[0].train_dl().dataset)

    predictions = []
    embeddings = []
    for dataset, model in zip(datasets, initial_models):
        c_pred, c_embs, _ = calculate_embeddings(model, dataset.train_dl())
        predictions.append(c_pred)
        embeddings.append(c_embs)
    predictions = np.stack(predictions, axis=0)

    concepts_to_cluster = []
    for concept_idx in range(initial_models[0].n_concepts):
        for concept_on in (True, False):
            if concept_on:
                n = np.sum(np.logical_and.reduce(predictions[:, :, concept_idx] > 0.5, axis=0))
            else:
                n = np.sum(np.logical_and.reduce(predictions[:, :, concept_idx] < 0.5, axis=0))
            concepts_to_cluster.append((n, concept_idx, concept_on))
    # concepts_to_cluster.sort(key=lambda t: t[0], reverse=True)

    discovered_concept_labels = []
    n_discovered_concepts = 0

    for (_, concept_idx, concept_on) in tqdm(concepts_to_cluster):
        if concept_on:
            sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] > 0.5, axis=0)
        else:
            sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] < 0.5, axis=0)

        Zs = []
        for e in embeddings:
            Zs.append(e[:, concept_idx][sample_filter])

        cluster_labels, _ = turtle.run_turtle(
            Zs=Zs, k=config["n_clusters"], warm_start=config["warm_start"])

        clusters = np.unique(cluster_labels)

        for cluster in clusters:
            labels = np.repeat(np.nan, train_dataset_size)
            labels[sample_filter] = cluster_labels == cluster

            if np.sum(labels == 1) < config["minimum_cluster_size"] or np.sum(labels == 0) < config["minimum_cluster_size"]:
                continue

            # labels = fill_in_discovered_concept_labels(
            #     datasets=datasets,
            #     labels=labels,
            #     max_epochs=config["max_epochs"])
            
            # most_similar = 0
            # for already_discovered in discovered_concept_labels:
            #     similarity = sklearn.metrics.roc_auc_score(
            #         already_discovered,
            #         labels
            #     )
            #     if similarity > most_similar:
            #         most_similar = similarity
            # if most_similar > 0.5:
            #     duplicates_caught += 1
            #     continue

            # for idx in range(len(discovered_concept_labels)):
            #     discovered_concept = np.apply_along_axis(majority_vote, axis=1, arr=discovered_concept_labels[idx])
            #     overlap = np.logical_and(
            #         np.logical_not(np.isnan(labels)),
            #         np.logical_not(np.isnan(discovered_concept))
            #     )
            #     if np.sum(overlap) < config["minimum_cluster_size"] or np.all(labels[overlap] == 0) or np.all(labels[overlap] == 1) or np.all(discovered_concept[overlap] == 0) or np.all(discovered_concept[overlap] == 1):
            #         continue
            #     similarity = sklearn.metrics.roc_auc_score(
            #         discovered_concept[overlap],
            #         labels[overlap]
            #     )
            #     if similarity > 0.9:
            #         discovered_concept_labels[idx] = np.concatenate((discovered_concept_labels[idx], np.expand_dims(labels, axis=1)), axis=1)
            #         break
            # else:
            discovered_concept_labels.append(labels)
            n_discovered_concepts += 1
            if n_discovered_concepts == config["max_concepts_to_discover"]:
                break
        else:
            continue
        break

                # similarities = np.expand_dims(labels, axis=1) == discovered_concept_labels
                # if np.any(np.mean(similarities, axis=0) > 0.9):
                #     continue

                # best_roc_auc = 0
                # best_roc_auc_idx = None
                # for i in range(datasets[0].concept_bank.shape[1]):
                #     if np.all(datasets[0].concept_bank[:, i][sample_filter] == 0) or np.all(datasets[0].concept_bank[:, i][sample_filter] == 1):
                #         continue
                #     auc = sklearn.metrics.roc_auc_score(
                #         datasets[0].concept_bank[:, i][sample_filter],
                #         labels[sample_filter],
                #     )
                #     if auc > best_roc_auc:
                #         best_roc_auc = auc
                #         best_roc_auc_idx = i

    #             discovered_concept_labels = np.concatenate(
    #                 (discovered_concept_labels, np.expand_dims(labels, axis=1)),
    #                 axis=1
    #             )
    #             discovered_concept_test_ground_truth = np.concatenate(
    #                 (discovered_concept_test_ground_truth, np.expand_dims(datasets[0].concept_test_ground_truth[:, best_roc_auc_idx], axis=1)),
    #             axis=1
    #             )
    #             discovered_concept_semantics.append(datasets[0].concept_names[best_roc_auc_idx])
    #             discovered_concept_roc_aucs.append(best_roc_auc)
    #             if config["use_wandb"]:
    #                 wandb.log({"best_roc_auc": best_roc_auc, "semantics": datasets[0].concept_names[best_roc_auc_idx]})
    #             n_discovered_concepts += 1

    #             if n_discovered_concepts > config["max_concepts_to_discover"]:
    #                 break
    #         else:
    #             continue
    #         break
    #     else:
    #         continue
    #     break
    # else:
    #     continue
    # break
    # for idx in range(len(discovered_concept_labels)):
    #     discovered_concept_labels[idx] = np.apply_along_axis(majority_vote, axis=1, arr=discovered_concept_labels[idx])
    # n_discovered_concepts = len(discovered_concept_labels)

    discovered_concept_labels = np.stack(discovered_concept_labels, axis=-1)
    
    deduplicated_discovered_concept_labels, \
    discovered_concept_test_ground_truth, \
    discovered_concept_train_ground_truth, \
    discovered_concept_semantics, \
    discovered_concept_roc_aucs = match_to_concept_bank(
        labels=discovered_concept_labels,
        dataset=datasets[0],
        save_path=save_path,
        max_epochs=config["max_epochs"])

    n_duplicates = discovered_concept_labels.shape[1] - deduplicated_discovered_concept_labels.shape[1]

    with (save_path / "discovered_concepts.pkl").open("wb") as f:
        pickle.dump((
            discovered_concept_labels,
            discovered_concept_test_ground_truth), f)

    with (save_path / "results.yaml").open("a") as f:
        yaml.safe_dump({
            "n_discovered_concepts": int(n_discovered_concepts),
            "discovered_concept_semantics": list(map(str, discovered_concept_semantics)),
            "discovered_concept_roc_aucs": list(map(float, discovered_concept_roc_aucs)),
            "n_duplicates": int(n_duplicates),
        }, f)
    
    if config["use_wandb"]:
        wandb.log({
            "n_discovered_concepts": n_discovered_concepts,
            "discovered_concept_semantics": discovered_concept_semantics,
            "discovered_concept_roc_aucs": discovered_concept_roc_aucs,
            "n_duplicates": n_duplicates,
        })

    return deduplicated_discovered_concept_labels, discovered_concept_train_ground_truth, discovered_concept_test_ground_truth, discovered_concept_roc_aucs
