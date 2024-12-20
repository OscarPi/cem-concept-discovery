import numpy as np
import lightning
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pickle
from cemcd.training import train_cem
from pathlib import Path
from cemcd.models.cem import ConceptEmbeddingModel 
import torch
import wandb
import yaml

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

def discover_concept(
        c_embs,
        c_pred,
        concept_idx,
        concept_on,
        chi,
        clustering_algorithm,
        discovered_labels_decay):
    sample_filter = c_pred[:, concept_idx] >= 0.5

    if not concept_on:
        sample_filter = np.logical_not(sample_filter)

    embeddings = c_embs[:, concept_idx][sample_filter]

    for i in range(10):
        if clustering_algorithm == "kmeans":
            clusters = KMeans(n_clusters=25, n_init=10).fit(embeddings) # TODO: used to explicitly change
        elif clustering_algorithm == "hdbscan":
            clusters = HDBSCAN().fit(embeddings)
        else:
            raise NotImplementedError(f"Unrecognised clustering algorithm: {clustering_algorithm}")

        largest_cluster_label = stats.mode(clusters.labels_)[0]
        largest_cluster_size = np.sum(clusters.labels_ == largest_cluster_label)

        if clustering_algorithm == "kmeans":
            embeddings_size = clusters.labels_.shape[0]
            buffer_size = int(embeddings_size*0.7)
            marked_as_off_size = embeddings_size-(largest_cluster_size+buffer_size)
            if marked_as_off_size < embeddings_size*0.2:
                marked_as_off_size = min(int(embeddings_size*0.2), embeddings_size-largest_cluster_size)
            largest_cluster_centre = clusters.cluster_centers_[largest_cluster_label]

            sorted_by_distance = np.argsort(np.linalg.norm(embeddings - largest_cluster_centre, axis=1))

            labels = np.repeat(np.nan, c_embs.shape[0])
            on_indices = np.arange(c_embs.shape[0])[sample_filter][sorted_by_distance][:largest_cluster_size]
            off_indices = np.arange(c_embs.shape[0])[sample_filter][sorted_by_distance][-marked_as_off_size:]
            labels[on_indices] = 1
            labels[off_indices] = 0
            if discovered_labels_decay == "linear":
                unsure_indices = np.arange(c_embs.shape[0])[sample_filter][sorted_by_distance][largest_cluster_size:-marked_as_off_size]
                labels[unsure_indices] = np.linspace(1, 0, unsure_indices.size)

        else:
            labels = np.zeros(c_embs.shape[0])
            labels[sample_filter][clusters.labels_ == largest_cluster_label] = 1
        if chi:
            labels[np.logical_not(sample_filter)] = 0
        
        similarities = np.expand_dims(labels, axis=1) == (c_pred > 0.5)
        if np.all(np.mean(similarities[np.logical_not(np.isnan(labels))], axis=0) < 0.9):
            return labels
        if clustering_algorithm != "kmeans":
            break

    return None

def _get_accuracies(test_results, n_provided_concepts):
    task_accuracy = round(test_results['test_y_accuracy'], 4)
    provided_concept_accuracies = []
    discovered_concept_accuracies = []
    provided_concept_aucs = []
    discovered_concept_aucs = []
    for key, value in test_results.items():
        if key[:7] == "concept":
            n = int(key.split("_")[1])
            if n <= n_provided_concepts:
                if key[-3:] == "auc":
                    provided_concept_aucs.append(value)
                else:
                    provided_concept_accuracies.append(value)
            else:
                if key[-3:] == "auc":
                    discovered_concept_aucs.append(value)
                else:
                    discovered_concept_accuracies.append(value)

    provided_concept_accuracy = round(np.mean(provided_concept_accuracies), 4)
    provided_concept_auc = round(np.mean(provided_concept_aucs), 4)
    if len(discovered_concept_accuracies) > 0:
        discovered_concept_accuracy = round(np.mean(discovered_concept_accuracies), 4)
        discovered_concept_auc = round(np.mean(discovered_concept_aucs), 4)
    else:
        discovered_concept_accuracy = np.nan
        discovered_concept_auc = np.nan

    print()
    print(f"After {len(discovered_concept_accuracies)} concepts have been discovered:")
    print(f"\tTask accuracy: {task_accuracy}")
    print(f"\tProvided concept accuracy: {provided_concept_accuracy}")
    print(f"\tDiscovered concept accuracy: {discovered_concept_accuracy}")
    print(f"\tProvided concept AUC: {provided_concept_auc}")
    print(f"\tDiscovered concept AUC: {discovered_concept_auc}")

    return task_accuracy, provided_concept_accuracy, discovered_concept_accuracy, provided_concept_auc, discovered_concept_auc

def find_best_concept_to_cluster(concepts_to_try, c_pred, c_embs, clustering_algorithm="kmeans"):
    best_concept = None
    best_concept_on = None
    best_score = 0
    options_to_remove = []
    for concept, on in concepts_to_try:
        sample_filter = c_pred[:, concept] >= 0.5
        if not on:
            sample_filter = np.logical_not(sample_filter)
        if np.sum(sample_filter) < 100:
            options_to_remove.append((concept, on))
            continue
        if clustering_algorithm == "kmeans":
            clusters = KMeans(n_clusters=25, n_init=10).fit(c_embs[:, concept][sample_filter])
        elif clustering_algorithm == "hdbscan":
            clusters = HDBSCAN().fit(c_embs[:, concept][sample_filter])
        score = silhouette_score(c_embs[:, concept][sample_filter], clusters.labels_)
        if score > best_score:
            best_concept = concept
            best_score = score
            best_concept_on = on

    return best_concept, best_concept_on, options_to_remove

def discover_multiple_concepts(config, resume, pre_concept_model, save_path, datasets):
    save_path = Path(save_path)

    train_dataset_size = len(datasets.train_dl().dataset)
    val_dataset_size = len(datasets.val_dl().dataset)
    test_dataset_size = len(datasets.test_dl().dataset)

    state = {
        "task_accuracies": [],
        "provided_concept_accuracies": [],
        "discovered_concept_accuracies": [],
        "provided_concept_aucs": [],
        "discovered_concept_aucs": [],
        "n_discovered_concepts": 0,
        "rejected_because_of_similarity": 0,
        "did_not_match_concept_bank": 0,
        "discovered_concept_labels": np.zeros((train_dataset_size, 0)),
        "discovered_concept_test_ground_truth": np.zeros((test_dataset_size, 0)),
        "all_concepts_to_try": tuple((i, b) for i in range(datasets.n_concepts) for b in (True, False)),
        "discovered_concept_semantics": []
    }
    state["concepts_left_to_try"] = list(state["all_concepts_to_try"])

    trainer = lightning.Trainer()

    state_path = save_path / "state.pickle"
    if resume and state_path.exists():
        with state_path.open("rb") as f:
            state = pickle.load(f)

        model_0 = ConceptEmbeddingModel(
            n_concepts=datasets.n_concepts,
            n_tasks=datasets.n_tasks,
            pre_concept_model=pre_concept_model,
            task_class_weights=torch.full((datasets.n_tasks,), np.nan),
            concept_loss_weights=torch.full((datasets.n_concepts,), np.nan)
        )
        model_0.load_state_dict(torch.load(save_path / "0_concepts_discovered.pth"))

        model = ConceptEmbeddingModel(
            n_concepts=datasets.n_concepts + state["n_discovered_concepts"],
            n_tasks=datasets.n_tasks,
            pre_concept_model=pre_concept_model,
            task_class_weights=torch.full((datasets.n_tasks,), np.nan),
            concept_loss_weights=torch.full((datasets.n_concepts + state["n_discovered_concepts"],), np.nan)
        )
        model.load_state_dict(torch.load(save_path / f"{state['n_discovered_concepts']}_concepts_discovered.pth"))

    else:
        model, test_results = train_cem(
            datasets.n_concepts,
            datasets.n_tasks,
            pre_concept_model,
            datasets.train_dl(),
            datasets.val_dl(),
            datasets.test_dl(),
            save_path=save_path / "0_concepts_discovered.pth",
            max_epochs=config["max_epochs"])
        model_0 = model

        task_accuracy, \
        provided_concept_accuracy, \
        discovered_concept_accuracy, \
        provided_concept_auc, \
        discovered_concept_auc = _get_accuracies(test_results, datasets.n_concepts)

        state["task_accuracies"].append(task_accuracy)
        state["provided_concept_accuracies"].append(provided_concept_accuracy)
        state["discovered_concept_accuracies"].append(discovered_concept_accuracy)
        state["provided_concept_aucs"].append(provided_concept_auc)
        state["discovered_concept_aucs"].append(discovered_concept_auc)
        if config["use_wandb"]:
            wandb.log({
                "task_accuracy": task_accuracy,
                "provided_concept_accuracy": provided_concept_accuracy,
                "discovered_concept_accuracy": discovered_concept_accuracy,
                "provided_concept_auc": provided_concept_auc,
                "discovered_concept_auc": discovered_concept_auc
            }, step=state["n_discovered_concepts"])

        state["c_pred"], state["c_embs"], _ = calculate_embeddings(model, datasets.train_dl())

        with state_path.open("wb") as f:
            pickle.dump(state, f)

    while len(state["concepts_left_to_try"]) > 0 and \
            state["n_discovered_concepts"] < datasets.concept_bank.shape[1] and \
            state["n_discovered_concepts"] < config["max_concepts_to_discover"]:

        concept_idx, concept_on, options_to_remove = find_best_concept_to_cluster(
            state["concepts_left_to_try"],
            state["c_pred"],
            state["c_embs"], 
            clustering_algorithm=config["clustering_algorithm"])

        for option in options_to_remove:
            state["concepts_left_to_try"].remove(option)
        if concept_idx is None:
            with state_path.open("wb") as f:
                pickle.dump(state, f)
            continue

        print(f"Decided to cluster concept {concept_idx}. On: {concept_on}")

        new_concept_labels = discover_concept(
            state["c_embs"],
            state["c_pred"],
            concept_idx,
            concept_on,
            chi=config["chi"],
            clustering_algorithm=config["clustering_algorithm"],
            discovered_labels_decay=config["discovered_labels_decay"]
        )

        if new_concept_labels is None:
            print("Failed to discover new concept: too similar to existing ones.")
            state["rejected_because_of_similarity"] += 1
            state["concepts_left_to_try"].remove((concept_idx, concept_on))
            with state_path.open("wb") as f:
                pickle.dump(state, f)
            continue

        state["n_discovered_concepts"] += 1

        state["discovered_concept_labels"] = np.concatenate(
            (state["discovered_concept_labels"], np.expand_dims(new_concept_labels, axis=1)),
            axis=1
        )

        pretrained_pre_concept_model = pre_concept_model
        pretrained_concept_embedding_generators = None
        pretrained_scoring_function = None
        if config["reuse_model"]:
            pretrained_pre_concept_model = model.pre_concept_model
            pretrained_concept_embedding_generators = model.concept_embedding_generators
            pretrained_scoring_function = model.scoring_function
        model_next, _ = train_cem(
            datasets.n_concepts + state["n_discovered_concepts"],
            datasets.n_tasks,
            pretrained_pre_concept_model,
            datasets.train_dl(state["discovered_concept_labels"]),
            datasets.val_dl(np.full((val_dataset_size, state["discovered_concept_labels"].shape[1]), np.nan)),
            datasets.test_dl(np.full((test_dataset_size, state["discovered_concept_labels"].shape[1]), np.nan)),
            save_path=save_path / f"{state['n_discovered_concepts']}_concepts_discovered.pth",
            max_epochs=config["max_epochs"],
            pretrained_concept_embedding_generators=pretrained_concept_embedding_generators,
            pretrained_scoring_function=pretrained_scoring_function)
        c_pred_next, c_embs_next, _ = calculate_embeddings(model_next, datasets.train_dl(state["discovered_concept_labels"]))

        similarities = np.mean(
            np.expand_dims(c_pred_next[:, datasets.n_concepts + state["n_discovered_concepts"] - 1] > 0.5, axis=1) == (c_pred_next[:, :-1] > 0.5),
            axis=0
        )
        if np.any(similarities > 0.8):
            print("Concept too similar!")
            state["rejected_because_of_similarity"] += 1
            state["n_discovered_concepts"] -= 1
            (save_path / f"{state['n_discovered_concepts'] + 1}_concepts_discovered.pth").unlink()
            state["discovered_concept_labels"] = state["discovered_concept_labels"][:, :-1]
            state["concepts_left_to_try"].remove((concept_idx, concept_on))
            with (save_path / "state.pickle").open("wb") as f:
                pickle.dump(state, f)
            continue

        similarities = np.mean(
            np.expand_dims(c_pred_next[:, datasets.n_concepts + state["n_discovered_concepts"] - 1] > 0.5, axis=1) == datasets.concept_bank,
            axis=0
        )
        most_similar = np.argmax(similarities)

        if similarities[most_similar] < 0.65:
            print("Concept does not seem to match any in concept bank.")
            state["did_not_match_concept_bank"] += 1
            state["n_discovered_concepts"] -= 1
            (save_path / f"{state['n_discovered_concepts'] + 1}_concepts_discovered.pth").unlink()
            state["discovered_concept_labels"] = state["discovered_concept_labels"][:, :-1]
            state["concepts_left_to_try"].remove((concept_idx, concept_on))
            with (save_path / "state.pickle").open("wb") as f:
                pickle.dump(state, f)
            continue

        if config["use_wandb"]:
            twod = TSNE(
                n_components=2,
                perplexity=30,
            ).fit_transform(state["c_embs"][:, concept_idx])

            no_nan_labels = np.repeat(-1., new_concept_labels.shape)
            no_nan_labels[np.logical_not(np.isnan(new_concept_labels))] = new_concept_labels[np.logical_not(np.isnan(new_concept_labels))]
            plt.scatter(twod[:, 0], twod[:, 1], c=no_nan_labels)
            wandb.log({"cluster_visualisation_generated_labels": plt}, step=state["n_discovered_concepts"])
            plt.scatter(twod[:, 0], twod[:, 1], c=datasets.concept_bank[:, most_similar])
            wandb.log({"cluster_visualisation_interpretation_ground_truth": plt}, step=state["n_discovered_concepts"])

        state["c_pred"], state["c_embs"] = c_pred_next, c_embs_next
        model = model_next
        print()
        print(f"Discovered concept number {state['n_discovered_concepts']} is most similar to {datasets.concept_names[most_similar]}.")
        print(f"\tSimilarity: {similarities[most_similar]:.0%}")
        state["discovered_concept_semantics"].append(datasets.concept_names[most_similar])

        state["discovered_concept_test_ground_truth"] = np.concatenate(
            (state["discovered_concept_test_ground_truth"], np.expand_dims(datasets.concept_test_ground_truth[:, most_similar], axis=1)),
            axis=1
        )

        [test_results] = trainer.test(model, datasets.test_dl(state["discovered_concept_test_ground_truth"]))
        task_accuracy, \
        provided_concept_accuracy, \
        discovered_concept_accuracy, \
        provided_concept_auc, \
        discovered_concept_auc = _get_accuracies(test_results, datasets.n_concepts)
        state["task_accuracies"].append(task_accuracy)
        state["provided_concept_accuracies"].append(provided_concept_accuracy)
        state["discovered_concept_accuracies"].append(discovered_concept_accuracy)
        state["provided_concept_aucs"].append(provided_concept_auc)
        state["discovered_concept_aucs"].append(discovered_concept_auc)

        if config["use_wandb"]:
            wandb.log({
                "task_accuracy": task_accuracy,
                "provided_concept_accuracy": provided_concept_accuracy,
                "discovered_concept_accuracy": discovered_concept_accuracy,
                "provided_concept_auc": provided_concept_auc,
                "discovered_concept_auc": discovered_concept_auc,
            }, step=state["n_discovered_concepts"])

        state["all_concepts_to_try"] = \
            state["all_concepts_to_try"] + \
            ((datasets.n_concepts + state["n_discovered_concepts"] - 1, True), (datasets.n_concepts + state["n_discovered_concepts"] - 1, False))
        state["concepts_left_to_try"] = list(state["all_concepts_to_try"])

        with (save_path / "state.pickle").open("wb") as f:
            pickle.dump(state, f)

    results = {
        "n_discovered_concepts": state["n_discovered_concepts"],
        "rejected_because_of_similarity": state["rejected_because_of_similarity"],
        "did_not_match_concept_bank": state["did_not_match_concept_bank"],
        "discovered_concept_semantics": state["discovered_concept_semantics"],
        "task_accuracies": state["task_accuracies"],
        "provided_concept_accuracies": list(map(float, state["provided_concept_accuracies"])),
        "provided_concept_aucs": list(map(float, state["provided_concept_aucs"])),
        "discovered_concept_accuracies": list(map(float, state["discovered_concept_accuracies"])),
        "discovered_concept_aucs": list(map(float, state["discovered_concept_aucs"]))
    }
    with (save_path / "results.yaml").open("w") as f:
        yaml.safe_dump(results, f)

    if config["use_wandb"]:
        semantics_data = list(zip(range(1, state['n_discovered_concepts'] + 1), state['discovered_concept_semantics']))
        wandb.log({
            "n_discovered_concepts": state['n_discovered_concepts'],
            "rejected_because_of_similarity": state['rejected_because_of_similarity'],
            "did_not_match_concept_bank": state['did_not_match_concept_bank'],
            "concept_semantics": wandb.Table(data=semantics_data, columns=["Discovered concept", "Meaning"])
        }, commit=False)

    return model, model_0, state["n_discovered_concepts"], state["discovered_concept_test_ground_truth"]
