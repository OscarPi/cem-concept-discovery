import numpy as np
import lightning
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import os
import pickle
from cemcd.training import train_cem
from lightning.pytorch import seed_everything
from pathlib import Path
from cemcd.models.cem import ConceptEmbeddingModel 
import torch
import wandb

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

def discover_concept(c_embs, c_pred, concept_idx, concept_on, random_state=42, chi=True, clustering_algorithm="kmeans"):
    sample_filter = c_pred[:, concept_idx] >= 0.5

    if not concept_on:
        sample_filter = np.logical_not(sample_filter)

    embeddings = c_embs[:, concept_idx][sample_filter]

    for i in range(10):
        if clustering_algorithm == "kmeans":
            clusters = KMeans(n_clusters=25, n_init=10, random_state=random_state + i*25).fit(embeddings)
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

def find_best_concept_to_cluster(concepts_to_try, c_pred, c_embs, random_state, clustering_algorithm="kmeans"):
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
            clusters = KMeans(n_clusters=25, n_init=10, random_state=random_state).fit(c_embs[:, concept][sample_filter])
        elif clustering_algorithm == "hdbscan":
            clusters = HDBSCAN().fit(c_embs[:, concept][sample_filter])
        score = silhouette_score(c_embs[:, concept][sample_filter], clusters.labels_)
        if score > best_score:
            best_concept = concept
            best_score = score
            best_concept_on = on

    return best_concept, best_concept_on, options_to_remove

def discover_multiple_concepts(
    resume,
    n_concepts,
    n_tasks,
    pre_concept_model,
    save_path,
    train_dl_getter,
    val_dl_getter,
    test_dl_getter,
    concept_bank,
    concept_test_ground_truth,
    concept_names,
    max_concepts_to_discover=10,
    random_state=42,
    chi=True,
    max_epochs=300,
    reuse=False,
    clustering_algorithm="kmeans",
    wandb_enabled=True):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    seed_everything(random_state, workers=True)

    train_dataset_size = len(train_dl_getter(None).dataset)
    val_dataset_size = len(val_dl_getter(None).dataset)
    test_dataset_size = len(test_dl_getter(None).dataset)

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
        "all_concepts_to_try": tuple((i, b) for i in range(n_concepts) for b in (True, False)),
        "discovered_concept_semantics": []
    }
    state["concepts_left_to_try"] = list(state["all_concepts_to_try"])

    trainer = lightning.Trainer()

    if resume and os.path.exists(os.path.join(save_path, "state.pickle")):
        with open(os.path.join(save_path, "state.pickle"), "rb") as f:
            state = pickle.load(f)

        model_0 = ConceptEmbeddingModel(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            pre_concept_model=pre_concept_model,
            task_class_weights=torch.full((n_tasks,), np.nan),
            concept_loss_weights=torch.full((n_concepts,), np.nan)
        )
        model_0.load_state_dict(torch.load(os.path.join(save_path, "0_concepts_discovered.pth")))

        model = ConceptEmbeddingModel(
            n_concepts=n_concepts + state["n_discovered_concepts"],
            n_tasks=n_tasks,
            pre_concept_model=pre_concept_model,
            task_class_weights=torch.full((n_tasks,), np.nan),
            concept_loss_weights=torch.full((n_concepts + state["n_discovered_concepts"],), np.nan)
        )
        model.load_state_dict(torch.load(os.path.join(save_path, f"{state['n_discovered_concepts']}_concepts_discovered.pth")))

        for i in range(state["n_discovered_concepts"]+1, state["n_discovered_concepts"]+max_concepts_to_discover):
            if os.path.exists(os.path.join(save_path, f"{i}_concepts_discovered.pth")):
                Path(os.path.join(save_path, f"{i}_concepts_discovered.pth")).unlink()
    else:
        if os.path.exists(os.path.join(save_path, "0_concepts_discovered.pth")):
            Path(os.path.join(save_path, "0_concepts_discovered.pth")).unlink()
        model, test_results = train_cem(
            n_concepts,
            n_tasks,
            pre_concept_model,
            train_dl_getter(None),
            val_dl_getter(None),
            test_dl_getter(None),
            save_path=os.path.join(save_path, "0_concepts_discovered.pth"),
            max_epochs=max_epochs)
        model_0 = model

        task_accuracy, \
        provided_concept_accuracy, \
        discovered_concept_accuracy, \
        provided_concept_auc, \
        discovered_concept_auc = _get_accuracies(test_results, n_concepts)

        state["task_accuracies"].append(task_accuracy)
        state["provided_concept_accuracies"].append(provided_concept_accuracy)
        state["discovered_concept_accuracies"].append(discovered_concept_accuracy)
        state["provided_concept_aucs"].append(provided_concept_auc)
        state["discovered_concept_aucs"].append(discovered_concept_auc)
        if wandb_enabled:
            wandb.log({
                "task_accuracy": task_accuracy,
                "provided_concept_accuracy": provided_concept_accuracy,
                "discovered_concept_accuracy": discovered_concept_accuracy,
                "provided_concept_auc": provided_concept_auc,
                "discovered_concept_auc": discovered_concept_auc
            }, step=state["n_discovered_concepts"])

        state["c_pred"], state["c_embs"], _ = calculate_embeddings(model, train_dl_getter(None))

        with open(os.path.join(save_path, "state.pickle"), "wb") as f:
            pickle.dump(state, f)

    while len(state["concepts_left_to_try"]) > 0 and \
          state["n_discovered_concepts"] < concept_bank.shape[1] and \
            state["n_discovered_concepts"] < max_concepts_to_discover:

        concept_idx, concept_on, options_to_remove = find_best_concept_to_cluster(
            state["concepts_left_to_try"],
            state["c_pred"],
            state["c_embs"], 
            random_state,
            clustering_algorithm=clustering_algorithm)

        for option in options_to_remove:
            state["concepts_left_to_try"].remove(option)
        if concept_idx is None:
            with open(os.path.join(save_path, "state.pickle"), "wb") as f:
                pickle.dump(state, f)
            continue

        print(f"Decided to cluster concept {concept_idx}. On: {concept_on}")

        new_concept_labels = discover_concept(
            state["c_embs"],
            state["c_pred"],
            concept_idx,
            concept_on,
            random_state=random_state,
            chi=chi,
            clustering_algorithm=clustering_algorithm
        )

        if new_concept_labels is None:
            print("Failed to discover new concept: too similar to existing ones.")
            state["rejected_because_of_similarity"] += 1
            state["concepts_left_to_try"].remove((concept_idx, concept_on))
            with open(os.path.join(save_path, "state.pickle"), "wb") as f:
                pickle.dump(state, f)
            continue

        state["n_discovered_concepts"] += 1

        state["discovered_concept_labels"] = np.concatenate(
            (state["discovered_concept_labels"], np.expand_dims(new_concept_labels, axis=1)),
            axis=1
        )

        pretrained_pre_concept_model = None
        pretrained_concept_embedding_generators = None
        pretrained_scoring_function = None
        if reuse:
            pretrained_pre_concept_model = model.pre_concept_model
            pretrained_concept_embedding_generators = model.concept_embedding_generators
            pretrained_scoring_function = model.scoring_function
        model_next, _ = train_cem(
            n_concepts + state["n_discovered_concepts"],
            n_tasks,
            pre_concept_model,
            train_dl_getter(state["discovered_concept_labels"]),
            val_dl_getter(np.full((val_dataset_size, state["discovered_concept_labels"].shape[1]), np.nan)),
            test_dl_getter(np.full((test_dataset_size, state["discovered_concept_labels"].shape[1]), np.nan)),
            save_path=os.path.join(save_path, f"{state['n_discovered_concepts']}_concepts_discovered.pth"),
            max_epochs=max_epochs,
            pretrained_pre_concept_model=pretrained_pre_concept_model,
            pretrained_concept_embedding_generators=pretrained_concept_embedding_generators,
            pretrained_scoring_function=pretrained_scoring_function)
        c_pred_next, c_embs_next, _ = calculate_embeddings(model_next, train_dl_getter(state["discovered_concept_labels"]))

        similarities = np.mean(
            np.expand_dims(c_pred_next[:, n_concepts + state["n_discovered_concepts"] - 1] > 0.5, axis=1) == (c_pred_next[:, :-1] > 0.5),
            axis=0
        )
        if np.any(similarities > 0.8):
            print("Concept too similar!")
            state["rejected_because_of_similarity"] += 1
            state["n_discovered_concepts"] -= 1
            Path(os.path.join(save_path, f"{state['n_discovered_concepts'] + 1}_concepts_discovered.pth")).unlink()
            state["discovered_concept_labels"] = state["discovered_concept_labels"][:, :-1]
            state["concepts_left_to_try"].remove((concept_idx, concept_on))
            with open(os.path.join(save_path, "state.pickle"), "wb") as f:
                pickle.dump(state, f)
            continue

        similarities = np.mean(
            np.expand_dims(c_pred_next[:, n_concepts + state["n_discovered_concepts"] - 1] > 0.5, axis=1) == concept_bank,
            axis=0
        )
        most_similar = np.argmax(similarities)

        if similarities[most_similar] < 0.65:
            print("Concept does not seem to match any in concept bank.")
            state["did_not_match_concept_bank"] += 1
            state["n_discovered_concepts"] -= 1
            Path(os.path.join(save_path, f"{state['n_discovered_concepts'] + 1}_concepts_discovered.pth")).unlink()
            state["discovered_concept_labels"] = state["discovered_concept_labels"][:, :-1]
            state["concepts_left_to_try"].remove((concept_idx, concept_on))
            with open(os.path.join(save_path, "state.pickle"), "wb") as f:
                pickle.dump(state, f)
            continue

        if wandb_enabled:
            twod = TSNE(
                n_components=2,
                perplexity=30,
            ).fit_transform(state["c_embs"][:, concept_idx])

            no_nan_labels = np.repeat(-1, new_concept_labels.shape)
            no_nan_labels[np.logical_not(np.isnan(new_concept_labels))] = new_concept_labels[np.logical_not(np.isnan(new_concept_labels))]

            plt.scatter(twod[:, 0], twod[:, 1], c=no_nan_labels)
            wandb.log({"cluster_visualisation_generated_labels": plt}, step=state["n_discovered_concepts"])
            plt.scatter(twod[:, 0], twod[:, 1], c=concept_bank[:, most_similar])
            wandb.log({"cluster_visualisation_interpretation_ground_truth": plt}, step=state["n_discovered_concepts"])

        state["c_pred"], state["c_embs"] = c_pred_next, c_embs_next
        model = model_next
        print()
        print(f"Discovered concept number {state['n_discovered_concepts']} is most similar to {concept_names[most_similar]}.")
        print(f"\tSimilarity: {similarities[most_similar]:.0%}")
        state["discovered_concept_semantics"].append(concept_names[most_similar])

        state["discovered_concept_test_ground_truth"] = np.concatenate(
            (state["discovered_concept_test_ground_truth"], np.expand_dims(concept_test_ground_truth[:, most_similar], axis=1)),
            axis=1
        )

        [test_results] = trainer.test(model, test_dl_getter(state["discovered_concept_test_ground_truth"]))
        task_accuracy, \
        provided_concept_accuracy, \
        discovered_concept_accuracy, \
        provided_concept_auc, \
        discovered_concept_auc = _get_accuracies(test_results, n_concepts)
        state["task_accuracies"].append(task_accuracy)
        state["provided_concept_accuracies"].append(provided_concept_accuracy)
        state["discovered_concept_accuracies"].append(discovered_concept_accuracy)
        state["provided_concept_aucs"].append(provided_concept_auc)
        state["discovered_concept_aucs"].append(discovered_concept_auc)

        if wandb_enabled:
            wandb.log({
                "task_accuracy": task_accuracy,
                "provided_concept_accuracy": provided_concept_accuracy,
                "discovered_concept_accuracy": discovered_concept_accuracy,
                "provided_concept_auc": provided_concept_auc,
                "discovered_concept_auc": discovered_concept_auc,
            }, step=state["n_discovered_concepts"])

        state["all_concepts_to_try"] = \
            state["all_concepts_to_try"] + \
            ((n_concepts + state["n_discovered_concepts"] - 1, True), (n_concepts + state["n_discovered_concepts"] - 1, False))
        state["concepts_left_to_try"] = list(state["all_concepts_to_try"])

        with open(os.path.join(save_path, "state.pickle"), "wb") as f:
            pickle.dump(state, f)

    with open(os.path.join(save_path, "results.txt"), "w") as f:
        f.write(f"{state['n_discovered_concepts']} concepts were discovered.\n")
        f.write(f"{state['rejected_because_of_similarity']} concepts were rejected because they were too similar to an existing concept.\n")
        f.write(f"{state['did_not_match_concept_bank']} concepts were rejected because they did not match a concept in the concept bank.\n")
        f.write(f"Task accuracies: {', '.join([str(x) for x in state['task_accuracies']])}\n")
        f.write(f"Provided concept AUCs: {', '.join([str(x) for x in state['provided_concept_accuracies']])}\n")
        f.write(f"Discovered concept AUCs: {', '.join([str(x) for x in state['discovered_concept_accuracies']])}\n")
        f.write(f"Discovered concept semantics: {', '.join(state['discovered_concept_semantics'])}")
    if wandb_enabled:
        semantics_data = list(zip(range(1, state['n_discovered_concepts'] + 1), state['discovered_concept_semantics']))
        wandb.log({
            "n_discovered_concepts": state['n_discovered_concepts'],
            "rejected_because_of_similarity": state['rejected_because_of_similarity'],
            "did_not_match_concept_bank": state['did_not_match_concept_bank'],
            "concept_semantics": wandb.Table(data=semantics_data, columns=["Discovered concept", "Meaning"])
        }, commit=False)

    with open(os.path.join(save_path, "discovered_concept_test_ground_truth.pickle"), "wb") as f:
        pickle.dump(state["discovered_concept_test_ground_truth"], f)

    return model, model_0, state["n_discovered_concepts"], state["discovered_concept_test_ground_truth"]
