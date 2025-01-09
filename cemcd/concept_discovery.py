import numpy as np
import lightning
import pickle
from cemcd.training import train_cem
from pathlib import Path
from cemcd.models.cem import ConceptEmbeddingModel 
import torch
import wandb
import yaml
import cemcd.turtle as turtle
import sklearn
from tqdm import trange

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

def get_accuracies(test_results, n_provided_concepts):
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

def discover_concepts(config, save_path, datasets):
    save_path = Path(save_path)
    results = {}

    train_dataset_size = len(datasets[0].train_dl().dataset)
    val_dataset_size = len(datasets[0].val_dl().dataset)
    test_dataset_size = len(datasets[0].test_dl().dataset)

    predictions = []
    embeddings = []
    for dataset in datasets:
        model, test_results = train_cem(
            n_concepts=dataset.n_concepts,
            n_tasks=dataset.n_tasks,
            pre_concept_model=None,
            latent_representation_size=dataset.latent_representation_size,
            train_dl=dataset.train_dl(),
            val_dl=dataset.val_dl(),
            test_dl=dataset.test_dl(),
            save_path=save_path / f"initial_{dataset.foundation_model}cem.pth",
            max_epochs=config["max_epochs"]
        )
        results.update({k + f"_{dataset.foundation_model}cem": v for k, v in test_results.items()})
        if config["wandb"]:
            wandb.log({k + f"_{dataset.foundation_model}cem": v for k, v in test_results.items()})
        c_pred, c_embs, _ = calculate_embeddings(model, datasets.train_dl())
        predictions.append(c_pred)
        embeddings.append(c_embs)
    predictions = np.stack(predictions, axis=0)

    n_discovered_concepts = 0
    did_not_match_concept_bank = 0
    duplicates = 0

    discovered_concept_labels = np.zeros((train_dataset_size, 0))
    discovered_concept_test_ground_truth = np.zeros((test_dataset_size, 0))
    discovered_concept_semantics = []

    for concept_idx in trange(datasets[0].n_concepts):
        for warm_start in (True, False):
            for concept_on in (True, False):
                if concept_on:
                    sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] > 0.5, axis=0)
                else:
                    sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] < 0.5, axis=0)

                dino_embeddings = dino_c_embs[:, concept_idx][sample_filter]
                clip_embeddings = clip_c_embs[:, concept_idx][sample_filter]

                cluster_labels, _ = turtle.run_turtle(
                    Zs=[dino_embeddings, clip_embeddings], k=config["n_clusters"], warm_start=warm_start)

                clusters, counts = np.unique(cluster_labels, return_counts=True)

                for cluster_idx, cluster in enumerate(clusters):
                    labels = np.repeat(np.nan, dino_c_embs.shape[0])
                    labels[sample_filter] = cluster_labels == cluster

                    if np.sum(labels == 1) < 30 or np.sum(labels == 0) < 30:
                        continue

                    best_roc_auc = 0
                    best_roc_auc_idx = None
                    for i in range(dino_datasets.concept_bank.shape[1]):
                        if np.all(dino_datasets.concept_bank[:, i][sample_filter] == 0) or np.all(dino_datasets.concept_bank[:, i][sample_filter] == 1):
                            continue
                        auc = sklearn.metrics.roc_auc_score(
                            dino_datasets.concept_bank[:, i][sample_filter],
                            labels[sample_filter],
                        )
                        if auc > best_roc_auc:
                            best_roc_auc = auc
                            best_roc_auc_idx = i
                    
                    discovered_concept_labels = np.concatenate(
                        (discovered_concept_labels, np.expand_dims(labels, axis=1)),
                        axis=1
                    )
                    discovered_concept_test_ground_truth = np.concatenate(
                        (discovered_concept_test_ground_truth, np.expand_dims(dino_datasets.concept_test_ground_truth[:, best_roc_auc_idx], axis=1)),
                    axis=1
                    )
                    discovered_concept_semantics.append(dino_datasets.concept_names[best_roc_auc_idx])
                    wandb.log({"best_roc_auc": best_roc_auc, "semantics": dino_datasets.concept_names[best_roc_auc_idx]})
                    n_discovered_concepts += 1

    with open("discovered_concepts.pkl", "wb") as f:
        pickle.dump((
            discovered_concept_labels,
            discovered_concept_test_ground_truth,
            discovered_concept_semantics,
            n_discovered_concepts), f)

    task_accuracy, \
    provided_concept_accuracy, \
    discovered_concept_accuracy, \
    provided_concept_auc, \
    discovered_concept_auc = _get_accuracies(test_results, datasets.n_concepts)



        # if best_roc_auc < 0.65:
        #     print("Concept does not seem to match any in concept bank.")
        #     state["did_not_match_concept_bank"] += 1
        #     state["concepts_left_to_try"].remove((concept_idx, concept_on))
        #     with state_path.open("wb") as f:
        #         pickle.dump(state, f)
        #     continue



        # [test_results] = trainer.test(model, datasets.test_dl(state["discovered_concept_test_ground_truth"]))
        # task_accuracy, \
        # provided_concept_accuracy, \
        # discovered_concept_accuracy, \
        # provided_concept_auc, \
        # discovered_concept_auc = _get_accuracies(test_results, datasets.n_concepts)
        # state["task_accuracies"].append(task_accuracy)
        # state["provided_concept_accuracies"].append(provided_concept_accuracy)
        # state["discovered_concept_accuracies"].append(discovered_concept_accuracy)
        # state["provided_concept_aucs"].append(provided_concept_auc)
        # state["discovered_concept_aucs"].append(discovered_concept_auc)



    # results = {
    #     "n_discovered_concepts": state["n_discovered_concepts"],
    #     "rejected_because_of_similarity": state["rejected_because_of_similarity"],
    #     "did_not_match_concept_bank": state["did_not_match_concept_bank"],
    #     "discovered_concept_semantics": state["discovered_concept_semantics"],
    #     "task_accuracies": state["task_accuracies"],
    #     "provided_concept_accuracies": list(map(float, state["provided_concept_accuracies"])),
    #     "provided_concept_aucs": list(map(float, state["provided_concept_aucs"])),
    #     "discovered_concept_accuracies": list(map(float, state["discovered_concept_accuracies"])),
    #     "discovered_concept_aucs": list(map(float, state["discovered_concept_aucs"]))
    # }
    # with (save_path / "results.yaml").open("w") as f:
    #     yaml.safe_dump(results, f)

    # if config["use_wandb"]:
    #     semantics_data = list(zip(range(1, state['n_discovered_concepts'] + 1), state['discovered_concept_semantics']))
    #     wandb.log({
    #         "n_discovered_concepts": state['n_discovered_concepts'],
    #         "rejected_because_of_similarity": state['rejected_because_of_similarity'],
    #         "did_not_match_concept_bank": state['did_not_match_concept_bank'],
    #         "concept_semantics": wandb.Table(data=semantics_data, columns=["Discovered concept", "Meaning"])
    #     }, commit=False)

    # return model, model_0, state["n_discovered_concepts"], state["discovered_concept_test_ground_truth"]
