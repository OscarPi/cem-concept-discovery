from pathlib import Path
import yaml
import wandb
from tqdm import trange
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

def match_to_concept_bank(labels, dataset):
    not_nan = np.logical_not(np.isnan(labels))
    best_roc_auc = 0
    best_roc_auc_idx = None
    for i in range(dataset.concept_bank.shape[1]):
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

            cluster_labels, _ = turtle.run_turtle(
                Zs=Zs, k=config["n_clusters"], warm_start=config["warm_start"], epochs=config["turtle_epochs"])
            clusters = np.unique(cluster_labels)

            for cluster in clusters:
                labels = np.repeat(np.nan, train_dataset_size)
                labels[sample_filter] = cluster_labels == cluster

                if np.sum(labels == 1) < config["minimum_cluster_size"] * train_dataset_size or np.sum(labels == 0) < config["minimum_cluster_size"] * train_dataset_size:
                    continue

                roc_auc, matching_concept_idx = match_to_concept_bank(labels, datasets[0])

                if roc_auc < config["match_threshold"]:
                    did_not_match += 1
                    continue
                if datasets[0].concept_names[matching_concept_idx] in discovered_concept_semantics:
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
                discovered_concept_semantics.append(datasets[0].concept_names[matching_concept_idx])
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
