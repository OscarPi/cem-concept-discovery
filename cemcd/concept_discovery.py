from pathlib import Path
import yaml
import wandb
from tqdm import tqdm
import sklearn.metrics
import numpy as np
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

def split_concepts(config, save_path, initial_models, datasets, concepts_to_split):
    save_path = Path(save_path)

    train_dataset_size = len(datasets[0].train_dl().dataset)
    test_dataset_size = len(datasets[0].test_dl().dataset)

    predictions = []
    embeddings = []
    if config["cluster_representations"]:
        for dataset in datasets:
            concept_labels = np.zeros((0, dataset.n_concepts), dtype=np.float32)
            representations = np.zeros((0, dataset.latent_representation_size), dtype=np.float32)
            for x, _, c in dataset.train_dl():
                representations = np.concatenate((representations, x.cpu().detach().numpy()), axis=0)
                concept_labels = np.concatenate((concept_labels, c.cpu().detach().numpy()), axis=0)
            predictions.append(concept_labels)
            embeddings.append(representations)
    else:
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
    n_discovered_sub_concepts = [0] * datasets[0].n_concepts
    n_duplicates = 0

    for concept_idx in tqdm(concepts_to_split):
        sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] > 0.5, axis=0)

        if config["cluster_by_class"]:
            class_labels = datasets[0].train_y[sample_filter].cpu().detach().numpy()
            classes = np.unique(class_labels).tolist()
            cluster_labels = []
            for l in class_labels:
                cluster_labels.append(classes.index(l))
            cluster_labels = np.array(cluster_labels)
        else:
            Zs = []
            for e in embeddings:
                if config["cluster_representations"]:
                    Zs.append(e[sample_filter])
                else:
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

            if config["only_match_subconcepts"]:
                roc_auc, matching_concept_idx = match_to_concept_bank(labels, datasets[0], datasets[0].sub_concept_map[concept_idx])
            else:
                roc_auc, matching_concept_idx = match_to_concept_bank(labels, datasets[0])

            discovered_concept_name = datasets[0].concept_names[matching_concept_idx]

            if discovered_concept_name in discovered_concept_semantics:
                idx = discovered_concept_semantics.index(discovered_concept_name)
                discovered_concept_labels[:, idx] = discovered_concept_labels[:, idx] + labels
                assert np.all(discovered_concept_labels[:, idx] <= 1) # Bug occurs if we aren't only discovering subconcepts
                discovered_concept_roc_aucs[idx] = sklearn.metrics.roc_auc_score(discovered_concept_train_ground_truth[:, idx], discovered_concept_labels[:, idx])
                n_duplicates += 1
            else:
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
                n_discovered_sub_concepts[concept_idx] += 1

    n_discovered_top_concepts = 0

    np.savez(save_path / "discovered_concepts.npz",
        discovered_concept_labels=discovered_concept_labels,
        discovered_concept_train_ground_truth=discovered_concept_train_ground_truth,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)

    with (save_path / "results.yaml").open("a") as f:
        yaml.safe_dump({
            "n_discovered_top_concepts": n_discovered_top_concepts,
            "n_discovered_sub_concepts": n_discovered_sub_concepts,
            "discovered_concept_semantics": list(map(str, discovered_concept_semantics)),
            "discovered_concept_roc_aucs": list(map(float, discovered_concept_roc_aucs)),
            "n_duplicates": n_duplicates
        }, f)

    if config["use_wandb"]:
        wandb.log({
            "n_discovered_top_concepts": n_discovered_top_concepts,
            "n_discovered_sub_concepts": n_discovered_sub_concepts,
            "discovered_concept_semantics": discovered_concept_semantics,
            "discovered_concept_roc_aucs": discovered_concept_roc_aucs,
            "n_duplicates": n_duplicates
        })

    return (discovered_concept_labels,
            discovered_concept_train_ground_truth,
            discovered_concept_test_ground_truth,
            discovered_concept_roc_aucs,
            n_discovered_sub_concepts)
