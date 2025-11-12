from pathlib import Path
from tqdm import tqdm
import sklearn.metrics
import numpy as np
import torch
import lightning
import cemcd.turtle as turtle
import cemcd.sae as sae
from cemcd.data import get_latent_representation_size

def calculate_embeddings(model, dl):
    trainer = lightning.Trainer()
    results = trainer.predict(model, dl)

    c_pred = np.concatenate(
        list(map(lambda x: x["predicted_concept_probs"].detach().cpu().numpy(), results)),
        axis=0)

    c_embs = np.concatenate(
        list(map(lambda x: x["concept_embeddings"].detach().cpu().numpy(), results)),
        axis=0)
    c_embs = np.reshape(c_embs, (c_embs.shape[0], -1, model.embedding_size))

    y_pred = np.concatenate(
        list(map(lambda x: x["predicted_labels"].detach().cpu().numpy(), results)),
        axis=0)

    return c_pred, c_embs, y_pred

def split_by_class(datasets, sample_filter):
    train_dataset_size = len(datasets.data["train"])
    discovered_concept_labels = np.zeros((train_dataset_size, 0))

    class_labels = datasets.get_labels_and_concepts("train")[0][sample_filter].cpu().detach().numpy()
    classes = np.unique(class_labels).tolist()

    for cls in classes:
        labels = np.repeat(0, train_dataset_size)
        labels[sample_filter] = class_labels == cls
        discovered_concept_labels = np.concatenate(
            (discovered_concept_labels, np.expand_dims(labels, axis=1)),
            axis=1)

    return discovered_concept_labels

def split_by_clustering(cluster_config, train_dataset_size, sample_filter, Zs):
    discovered_concept_labels = np.zeros((train_dataset_size, 0))

    best_score = - len(Zs)
    best_n_clusters = None
    for n in range(cluster_config["min_n_clusters"], cluster_config["max_n_clusters"] + 1):
        cluster_labels, _, _ = turtle.run_turtle(
            Zs=Zs, k=n, warm_start=cluster_config["warm_start"], epochs=cluster_config["turtle_epochs"])
        score = 0
        for Z in Zs:
            score += sklearn.metrics.silhouette_score(Z, cluster_labels)

        print(f"n={n}, score={score}")
        if score > best_score:
            best_score = score
            best_n_clusters = n

    cluster_labels, _, _ = turtle.run_turtle(
        Zs=Zs, k=best_n_clusters, warm_start=cluster_config["warm_start"], epochs=cluster_config["turtle_epochs"])
    
    clusters = np.unique(cluster_labels)

    for cluster in clusters:
        labels = np.repeat(0, train_dataset_size)
        labels[sample_filter] = cluster_labels == cluster


        discovered_concept_labels = np.concatenate(
            (discovered_concept_labels, np.expand_dims(labels, axis=1)),
            axis=1)
        
    return discovered_concept_labels

def split_with_sae(sae_config, train_dataset_size, sample_filter, Zs):
    assert len(Zs) == 1, "SAE-based splitting only supports a single foundation model."
    sae_model = sae.BatchTopKSAE(sae_config)
    Z = torch.from_numpy(Zs[0]).to(sae_model.device)
    feature_acts = sae.train_sae(sae_model, Z, sae_config)
    non_dead_feature_acts = feature_acts[:, np.sum(feature_acts, axis=0) > 0]
    discovered_concept_labels = np.zeros((train_dataset_size, non_dead_feature_acts.shape[1]))
    discovered_concept_labels[sample_filter] = non_dead_feature_acts > 0
    return discovered_concept_labels

def split_concepts(config, initial_models, datasets, concepts_to_split):
    train_dataset_size = len(datasets.data["train"])

    predictions = []
    embeddings = []
    if config["use_foundation_model_representations_instead_of_concept_embeddings"]:
        for foundation_model in config["foundation_models"]:
            concept_labels = np.zeros((0, datasets.n_concepts), dtype=np.float32)
            representations = np.zeros((0, get_latent_representation_size(foundation_model)), dtype=np.float32)
            for x, _, c in datasets.get_dataloader("train", foundation_model=foundation_model):
                representations = np.concatenate((representations, x.cpu().detach().numpy()), axis=0)
                concept_labels = np.concatenate((concept_labels, c.cpu().detach().numpy()), axis=0)
            predictions.append(concept_labels)
            embeddings.append(representations)
    else:
        for foundation_model, model in zip(config["foundation_models"], initial_models):
            c_pred, c_embs, _ = calculate_embeddings(model, datasets.get_dataloader("train", foundation_model=foundation_model))
            predictions.append(c_pred)
            embeddings.append(c_embs)
    predictions = np.stack(predictions, axis=0)

    discovered_concept_labels = np.zeros((train_dataset_size, 0))
    n_discovered_sub_concepts = [0] * datasets.n_concepts

    for concept_idx in tqdm(concepts_to_split):
        sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] > 0.5, axis=0)

        if config["sub_concept_extraction_method"] == "split_by_class":
            new_discovered_concept_labels = split_by_class(datasets, sample_filter)
        else:
            Zs = []
            for e in embeddings:
                if config["use_foundation_model_representations_instead_of_concept_embeddings"]:
                    Zs.append(e[sample_filter])
                else:
                    Zs.append(e[:, concept_idx][sample_filter])

            if config["sub_concept_extraction_method"] == "clustering":
                new_discovered_concept_labels = split_by_clustering(
                    config["clustering_config"], train_dataset_size, sample_filter, Zs)
            elif config["sub_concept_extraction_method"] == "sae":
                sae_config = config["sae_config"]
                sae_config["act_size"] = Zs[0].shape[1]
                new_discovered_concept_labels = split_with_sae(sae_config, train_dataset_size, sample_filter, Zs)

        n_discovered_sub_concepts[concept_idx] = new_discovered_concept_labels.shape[1]
        discovered_concept_labels = np.concatenate(
            (discovered_concept_labels, new_discovered_concept_labels),
            axis=1)

    return discovered_concept_labels, n_discovered_sub_concepts
