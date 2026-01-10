import argparse
from tqdm import tqdm, trange
from pathlib import Path
import wandb
import yaml
import numpy as np
import torch
import lightning
import sklearn.metrics
from cemcd.training import train_cem, train_hicem, load_hicem
from cemcd.data import get_latent_representation_size
import cemcd.concept_discovery
from experiment_utils import load_config, load_datasets, train_initial_cems, load_initial_cems, get_intervention_accuracies

def get_accuracies(test_results, n_provided_concepts, model_name):
    task_accuracy = round(test_results['test_y_accuracy'], 4)
    provided_concept_accuracies = []
    discovered_concept_accuracies = []
    provided_concept_aucs = []
    discovered_concept_aucs = []
    for key, value in test_results.items():
        if key[:3] == "con":
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

    results = {
        f"{model_name}_task_accuracy": float(task_accuracy),
        f"{model_name}_provided_concept_accuracy": float(provided_concept_accuracy),
        f"{model_name}_provided_concept_accuracies": list(map(lambda x: round(float(x), 4), provided_concept_accuracies)),
        f"{model_name}_provided_concept_auc": float(provided_concept_auc),
        f"{model_name}_provided_concept_aucs": list(map(lambda x: round(float(x), 4), provided_concept_aucs))}
    if len(discovered_concept_accuracies) > 0:
        results.update({
            f"{model_name}_discovered_concept_accuracy": float(discovered_concept_accuracy),
            f"{model_name}_discovered_concept_accuracies": list(map(lambda x: round(float(x), 4), discovered_concept_accuracies)),
            f"{model_name}_discovered_concept_auc": float(discovered_concept_auc),
            f"{model_name}_discovered_concept_aucs": list(map(lambda x: round(float(x), 4), discovered_concept_aucs))})
    return results

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "command",
        choices=[
            "train-initial-models",
            "discover-concepts",
            "match-concepts",
            "train-hicems",
            "evaluate-interventions",
            "run-baselines"
        ],
        help="Sub-command to run."
    )

    parser.add_argument(
        "-c", "--config", 
        type=Path,
        required=True,
        help="Path to the experiment config file.")
    
    parser.add_argument(
        "-r", "--run-dir", 
        type=Path,
        required=True,
        help="Path to the run directory.")

    return parser.parse_args()

def get_provided_and_discovered_intervention_accuracies(models, foundation_models, datasets, discovered_concept_test_ground_truth, model_name_prefix, model_type="hicem"):
    n_provided_concepts = datasets.n_concepts
    n_discovered_concepts = discovered_concept_test_ground_truth.shape[1]
    all_concepts = range(n_provided_concepts + n_discovered_concepts)
    provided_concepts = range(n_provided_concepts)
    discovered_concepts = list(range(n_provided_concepts, n_provided_concepts + n_discovered_concepts))

    results = {}
    for foundation_model, model in zip(foundation_models, models):
        model_name = f"{model_name_prefix}_{foundation_model}_{model_type}"

        results[f"{model_name}_discovered_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model, additional_concepts=discovered_concept_test_ground_truth),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=False)
        results[f"{model_name}_discovered_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model, additional_concepts=discovered_concept_test_ground_truth),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=True)

        results[f"{model_name}_all_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model, additional_concepts=discovered_concept_test_ground_truth),
            concepts_to_intervene=all_concepts,
            one_at_a_time=False)
        results[f"{model_name}_provided_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model, additional_concepts=discovered_concept_test_ground_truth),
            concepts_to_intervene=provided_concepts,
            one_at_a_time=True)

    return results

def test_concept_interventions(
        initial_models,
        models_with_discovered_concepts,
        foundation_models,
        datasets,
        discovered_concept_test_ground_truth):
    results = {}

    for foundation_model, model in zip(foundation_models, initial_models):
        model_name = foundation_model + "_cem"
        intervention_accuracies_cumulative = get_intervention_accuracies(
            model=model,
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model),
            concepts_to_intervene=range(model.n_concepts),
            one_at_a_time=False)
        intervention_accuracies_one_at_a_time = get_intervention_accuracies(
            model=model,
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model),
            concepts_to_intervene=range(model.n_concepts),
            one_at_a_time=True)
        results[f"initial_{model_name}_interventions_cumulative"] = intervention_accuracies_cumulative
        results[f"initial_{model_name}_interventions_one_at_a_time"] = intervention_accuracies_one_at_a_time

    results.update(
        get_provided_and_discovered_intervention_accuracies(
            models=models_with_discovered_concepts,
            foundation_models=foundation_models,
            datasets=datasets,
            discovered_concept_test_ground_truth=discovered_concept_test_ground_truth,
            model_name_prefix="enhanced"
        )
    )

    return results

def log_results(config, run_dir, results):
    with (run_dir / "results.yaml").open("a") as f:
        yaml.safe_dump(results, f)

    if config["use_wandb"]:
        wandb.log(results)

def match_discovered_concepts_to_concept_bank(discovered_concept_labels, n_discovered_sub_concepts, datasets):
    matched_discovered_concepts = []
    discovered_concept_train_ground_truth = []
    discovered_concept_test_ground_truth = []
    discovered_concept_semantics = []
    discovered_concept_roc_aucs = []
    n_matched_sub_concepts = [0] * datasets.n_concepts

    for top_concept_idx, sub_concepts in enumerate(datasets.concept_bank):
        first_discovered_concept_to_check = sum(n_discovered_sub_concepts[:top_concept_idx])
        n_discovered_concepts_to_check = n_discovered_sub_concepts[top_concept_idx]

        for sub_concept in sub_concepts:
            true_sub_concept_labels = sub_concept["train_labels"]
            best_roc_auc = 0.0
            best_discovered_concept_idx = None
            for discovered_concept_idx in range(first_discovered_concept_to_check, first_discovered_concept_to_check + n_discovered_concepts_to_check):
                labels = discovered_concept_labels[:, discovered_concept_idx]
                score = sklearn.metrics.roc_auc_score(true_sub_concept_labels, labels)
                if score > best_roc_auc:
                    best_roc_auc = score
                    best_discovered_concept_idx = discovered_concept_idx
            if best_roc_auc > 0.7:
                matched_discovered_concepts.append(best_discovered_concept_idx)
                discovered_concept_train_ground_truth.append(true_sub_concept_labels)
                discovered_concept_test_ground_truth.append(sub_concept["test_labels"])
                discovered_concept_semantics.append(sub_concept["name"])
                discovered_concept_roc_aucs.append(best_roc_auc)
                n_matched_sub_concepts[top_concept_idx] += 1

    return (matched_discovered_concepts,
            np.stack(discovered_concept_train_ground_truth, axis=-1),
            np.stack(discovered_concept_test_ground_truth, axis=-1),
            discovered_concept_semantics,
            discovered_concept_roc_aucs,
            n_matched_sub_concepts)

def match_hisae_discovered_concepts_to_concept_bank(hisae_config, active_concepts, datasets):
    discovered_concept_labels = []
    discovered_concept_train_ground_truth = []
    discovered_concept_test_ground_truth = []
    hicem_concepts = []
    n_matched_concepts = 0

    for top_concept_idx, sub_concepts in tqdm(enumerate(datasets.concept_bank)):
        matched_sub_concepts = []
        for sub_concept in tqdm(sub_concepts, leave=False):
            true_sub_concept_labels = sub_concept["train_labels"]
            best_avg_roc_auc = 0.0
            best_latent_idx = None
            best_sub_latent_idxs = None
            for latent_idx in trange(hisae_config["dictionary_size"], leave=False):
                latent_activations = np.any(active_concepts[:, top_concept_idx] == latent_idx, axis=1)
                if np.all(latent_activations == 0):
                    continue

                sub_concept_roc_auc = sklearn.metrics.roc_auc_score(true_sub_concept_labels, latent_activations)

                first_sub_latent_idx = hisae_config["dictionary_size"] + latent_idx * hisae_config["sub_dictionary_size"]
                end_of_sub_latents = first_sub_latent_idx + hisae_config["sub_dictionary_size"]

                sub_sub_concept_roc_aucs = []
                sub_latent_idxs = []
                for sub_sub_concept in tqdm(sub_concept["sub_sub_concepts"], leave=False):
                    true_sub_sub_concept_labels = sub_sub_concept["train_labels"]
                    best_sub_roc_auc = 0.0
                    best_sub_latent_idx = None
                    for sub_latent_idx in range(first_sub_latent_idx, end_of_sub_latents):
                        sub_latent_activations = np.any(active_concepts[:, top_concept_idx] == sub_latent_idx, axis=1)
                        roc_auc = sklearn.metrics.roc_auc_score(true_sub_sub_concept_labels, sub_latent_activations)
                        if roc_auc > best_sub_roc_auc:
                            best_sub_roc_auc = roc_auc
                            best_sub_latent_idx = sub_latent_idx
                    sub_sub_concept_roc_aucs.append(best_sub_roc_auc)
                    if best_sub_roc_auc > 0.7:
                        sub_latent_idxs.append(best_sub_latent_idx)
                    else:
                        sub_latent_idxs.append(None)

                avg_roc_auc = np.mean([sub_concept_roc_auc] + sub_sub_concept_roc_aucs)

                if avg_roc_auc > best_avg_roc_auc:
                    best_avg_roc_auc = avg_roc_auc
                    best_latent_idx = latent_idx
                    best_sub_latent_idxs = sub_latent_idxs

            if best_avg_roc_auc > 0.7:
                matched_sub_sub_concepts = []
                for true_sub_sub_concept_idx, sub_sub_concept in enumerate(sub_concept["sub_sub_concepts"]):
                    if best_sub_latent_idxs[true_sub_sub_concept_idx] is not None:
                        sub_sub_concept_labels = np.any(active_concepts[:, top_concept_idx] == best_sub_latent_idxs[true_sub_sub_concept_idx], axis=1)
                        matched_sub_sub_concepts.append({
                            "name": sub_sub_concept["name"],
                            "idx": datasets.n_concepts + n_matched_concepts,
                            "roc_auc": sklearn.metrics.roc_auc_score(sub_sub_concept["train_labels"], sub_sub_concept_labels),
                            "positive_sub_concepts": [],
                            "negative_sub_concepts": []
                        })
                        discovered_concept_labels.append(sub_sub_concept_labels)
                        discovered_concept_train_ground_truth.append(sub_sub_concept["train_labels"])
                        discovered_concept_test_ground_truth.append(sub_sub_concept["test_labels"])
                        n_matched_concepts += 1

                sub_concept_labels = np.any(active_concepts[:, top_concept_idx] == best_latent_idx, axis=1)
                matched_sub_concepts.append({
                    "name": sub_concept["name"],
                    "idx": datasets.n_concepts + n_matched_concepts,
                    "roc_auc": sklearn.metrics.roc_auc_score(sub_concept["train_labels"], sub_concept_labels),
                    "positive_sub_concepts": matched_sub_sub_concepts,
                    "negative_sub_concepts": []
                })
                discovered_concept_labels.append(sub_concept_labels)
                discovered_concept_train_ground_truth.append(sub_concept["train_labels"])
                discovered_concept_test_ground_truth.append(sub_concept["test_labels"])
                n_matched_concepts += 1

        hicem_concepts.append({
            "name": datasets.concept_names[top_concept_idx],
            "idx": top_concept_idx,
            "positive_sub_concepts": matched_sub_concepts,
            "negative_sub_concepts": []
        })

    return (np.stack(discovered_concept_labels, axis=-1),
        np.stack(discovered_concept_train_ground_truth, axis=-1),
        np.stack(discovered_concept_test_ground_truth, axis=-1),
        hicem_concepts,
        n_matched_concepts)

def run_unlabelled_concepts_baseline(run_dir, config, datasets, sub_concepts):
    n_top_concepts = len(sub_concepts)
    n_discovered_sub_concepts = sum(map(sum, sub_concepts))
    trainer = lightning.Trainer()

    train_dataset_size = len(datasets.data["train"])
    val_dataset_size = len(datasets.data["val"])
    test_dataset_size = len(datasets.data["test"])

    unlabelled_results = {}
    for foundation_model in config["foundation_models"]:
        train_dl = datasets.get_dataloader(
            split="train",
            foundation_model=foundation_model,
            additional_concepts=np.full((train_dataset_size, n_discovered_sub_concepts), np.nan))
        val_dl = datasets.get_dataloader(
            split="val",
            foundation_model=foundation_model,
            additional_concepts=np.full((val_dataset_size, n_discovered_sub_concepts), np.nan))
        test_dl = datasets.get_dataloader(
            split="test",
            foundation_model=foundation_model,
            additional_concepts=np.full((test_dataset_size, n_discovered_sub_concepts), np.nan))
        model, _ = train_hicem(
            sub_concepts=sub_concepts,
            n_tasks=datasets.n_tasks,
            latent_representation_size=get_latent_representation_size(foundation_model),
            embedding_size=config["hicem_embedding_size"],
            concept_loss_weight=config["hicem_concept_loss_weight"],
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            save_path=run_dir / f"unlabelled_{foundation_model}_hicem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])

        c_pred, _, _ = cemcd.concept_discovery.calculate_embeddings(model, train_dl)

        discovered_concept_test_ground_truth = np.zeros((test_dataset_size, 0))

        sub_concept_idx = n_top_concepts
        for top_concept_idx in range(n_top_concepts):
            for _ in range(sum(sub_concepts[top_concept_idx])):
                if config["only_match_subconcepts"]:
                    _, matching_concept_idx = match_single_concept_to_concept_bank(c_pred[:, sub_concept_idx], datasets, datasets.sub_concept_map[top_concept_idx])
                else:
                    _, matching_concept_idx = match_single_concept_to_concept_bank(c_pred[:, sub_concept_idx], datasets)
                discovered_concept_test_ground_truth = np.concatenate(
                    (discovered_concept_test_ground_truth, np.expand_dims(datasets.concept_test_ground_truth[:, matching_concept_idx], axis=1)),
                    axis=1)
                sub_concept_idx += 1

        [test_results] = trainer.test(model, dataset.test_dl(discovered_concept_test_ground_truth))
        model_results = get_accuracies(test_results, n_top_concepts, f"unlabelled_{dataset.foundation_model}_hicem")
        unlabelled_results.update(model_results)

        unlabelled_results.update(get_provided_and_discovered_intervention_accuracies(
            models=[model],
            datasets=[dataset],
            discovered_concept_test_ground_truth=discovered_concept_test_ground_truth,
            model_name_prefix="unlabelled"
        ))
    
    return unlabelled_results

def train_initial_models(run_dir, config, datasets):
    log = lambda results: log_results(config, run_dir, results)

    if not config["use_foundation_model_representations_instead_of_concept_embeddings"]:
        _, test_results = train_initial_cems(config, datasets, run_dir)

        for foundation_model, test_result in zip(config["foundation_models"], test_results):
            model_results = get_accuracies(test_result, datasets.n_concepts, f"initial_{foundation_model}_cem")
            log(model_results)

def discover_concepts(run_dir, config, datasets):
    log = lambda results: log_results(config, run_dir, results)

    initial_models = load_initial_cems(run_dir, config, datasets)

    results = cemcd.concept_discovery.split_concepts(
        config=config,
        initial_models=initial_models,
        datasets=datasets)
    
    if config["sub_concept_extraction_method"] == "hisae":
        active_concepts = results["active_concepts"]
        np.save(run_dir / "active_concepts", active_concepts)
    else:
        discovered_concept_labels = results["discovered_concept_labels"]
        n_discovered_sub_concepts = results["n_discovered_sub_concepts"]
        np.save(run_dir / "discovered_concept_labels", discovered_concept_labels)
        log({"n_discovered_sub_concepts": n_discovered_sub_concepts})


def match_concepts(run_dir, config, datasets):
    log = lambda results: log_results(config, run_dir, results)

    if not config["match_to_concept_bank_and_train_hicem"]:
        return

    if config["sub_concept_extraction_method"] == "hisae":
        active_concepts = np.load(run_dir / "active_concepts.npy")

        (matched_discovered_concept_labels,
         discovered_concept_train_ground_truth,
         discovered_concept_test_ground_truth,
         hicem_concepts,
         n_matched_discovered_concepts) = match_hisae_discovered_concepts_to_concept_bank(
            hisae_config=config["hisae_config"],
            active_concepts=active_concepts,
            datasets=datasets
         )
    else:
        discovered_concept_labels = np.load(run_dir / "discovered_concept_labels.npy")
        with (run_dir / "results.yaml").open("r") as f:
            results_yaml = yaml.safe_load(f)
            n_discovered_sub_concepts = results_yaml["n_discovered_sub_concepts"]

        (matched_discovered_concepts,
         discovered_concept_train_ground_truth,
         discovered_concept_test_ground_truth,
         discovered_concept_semantics,
         discovered_concept_roc_aucs,
         n_matched_sub_concepts) = match_discovered_concepts_to_concept_bank(
            discovered_concept_labels=discovered_concept_labels,
            n_discovered_sub_concepts=n_discovered_sub_concepts,
            concept_bank=datasets.concept_bank)
        
        matched_discovered_concept_labels = discovered_concept_labels[:, matched_discovered_concepts]

        hicem_concepts = []
        next_discovered_concept_idx = datasets.n_concepts
        for top_concept_idx, n_matched in enumerate(n_matched_sub_concepts):
            positive_sub_concepts = []
            for _ in range(n_matched):
                positive_sub_concepts.append({
                    "name": f"{discovered_concept_semantics[next_discovered_concept_idx - datasets.n_concepts]}",
                    "idx": next_discovered_concept_idx,
                    "roc_auc": float(discovered_concept_roc_aucs[next_discovered_concept_idx - datasets.n_concepts]),
                    "positive_sub_concepts": [],
                    "negative_sub_concepts": []
                })
                next_discovered_concept_idx += 1
            hicem_concepts.append({
                "name": datasets.concept_names[top_concept_idx],
                "idx": top_concept_idx,
                "positive_sub_concepts": positive_sub_concepts,
                "negative_sub_concepts": []
            })

            n_matched_discovered_concepts = sum(n_matched_sub_concepts)


    log({"n_matched_discovered_concepts": n_matched_discovered_concepts,
         "hicem_concepts": hicem_concepts})

    np.savez(run_dir / "matched_discovered_concepts.npz",
        matched_discovered_concept_labels=matched_discovered_concept_labels,
        discovered_concept_train_ground_truth=discovered_concept_train_ground_truth,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)

def train_hicems(run_dir, config, datasets):
    if not config["match_to_concept_bank_and_train_hicem"]:
        return
    
    log = lambda results: log_results(config, run_dir, results)

    val_dataset_size = len(datasets.data["val"])

    with (run_dir / "results.yaml").open("r") as f:
        results_yaml = yaml.safe_load(f)
        hicem_concepts = results_yaml["hicem_concepts"]
        n_matched_discovered_concepts = results_yaml["n_matched_discovered_concepts"]

    loaded_arrays = np.load(run_dir / "matched_discovered_concepts.npz")
    matched_discovered_concept_labels = loaded_arrays["matched_discovered_concept_labels"]
    discovered_concept_test_ground_truth = loaded_arrays["discovered_concept_test_ground_truth"]

    for foundation_model in config["foundation_models"]:
        _, test_results = train_hicem(
            concepts=hicem_concepts,
            n_tasks=datasets.n_tasks,
            latent_representation_size=get_latent_representation_size(foundation_model),
            embedding_size=config["hicem_embedding_size"],
            concept_loss_weight=config["hicem_concept_loss_weight"],
            train_dl=datasets.get_dataloader("train", foundation_model=foundation_model, additional_concepts=matched_discovered_concept_labels),
            val_dl=datasets.get_dataloader("val", foundation_model=foundation_model, additional_concepts=np.full((val_dataset_size, n_matched_discovered_concepts), np.nan)),
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model, additional_concepts=discovered_concept_test_ground_truth),
            save_path=run_dir / f"enhanced_{foundation_model}_hicem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        log(get_accuracies(test_results, datasets.n_concepts, f"enhanced_{foundation_model}_hicem"))

def load_hicems(run_dir, config, datasets):
    hicems = []

    with (run_dir / "results.yaml").open("r") as f:
        results_yaml = yaml.safe_load(f)
        hicem_concepts = results_yaml["hicem_concepts"]

    loaded_arrays = np.load(run_dir / "matched_discovered_concepts.npz")
    matched_discovered_concept_labels = loaded_arrays["matched_discovered_concept_labels"]

    for foundation_model in config["foundation_models"]:
        model = load_hicem(
            concepts=hicem_concepts,
            n_tasks=datasets.n_tasks,
            latent_representation_size=get_latent_representation_size(foundation_model),
            embedding_size=config["hicem_embedding_size"],
            concept_loss_weight=config["hicem_concept_loss_weight"],
            train_dl=datasets.get_dataloader("train", foundation_model=foundation_model, additional_concepts=matched_discovered_concept_labels),
            path=run_dir / f"enhanced_{foundation_model}_hicem.pth",
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        
        hicems.append(model)

    return hicems

def evaluate_interventions(run_dir, config, datasets):
    if not config["evaluate_interventions"]:
        return
    
    log = lambda results: log_results(config, run_dir, results)

    initial_models = load_initial_cems(run_dir, config, datasets)
    models_with_discovered_concepts = load_hicems(run_dir, config, datasets)

    loaded_arrays = np.load(run_dir / "matched_discovered_concepts.npz")
    discovered_concept_test_ground_truth = loaded_arrays["discovered_concept_test_ground_truth"]

    intervention_results = test_concept_interventions(
        initial_models=initial_models,
        models_with_discovered_concepts=models_with_discovered_concepts,
        foundation_models=config["foundation_models"],
        datasets=datasets,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)
    log(intervention_results)

def run_baselines(run_dir, config, datasets):
    # TODO: implementation needs to be updated
    log = lambda results: log_results(config, run_dir, results)

    if not config["evaluate_cems_with_discovered_concepts"]:
        cems_with_discovered_concepts = []
        for dataset in datasets:
            model, test_results = train_cem(
                n_concepts=dataset.n_concepts + n_matched_discovered_concepts,
                n_tasks=dataset.n_tasks,
                latent_representation_size=dataset.latent_representation_size,
                embedding_size=config["cem_embedding_size"],
                concept_loss_weight=config["cem_concept_loss_weight"],
                train_dl=dataset.train_dl(matched_discovered_concept_labels),
                val_dl=dataset.val_dl(np.full((val_dataset_size, n_matched_discovered_concepts), np.nan)),
                test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
                save_path=run_dir / f"enhanced_{dataset.foundation_model}_cem.pth",
                max_epochs=config["max_epochs"],
                use_task_class_weights=config["use_task_class_weights"],
                use_concept_loss_weights=config["use_concept_loss_weights"])
            log(get_accuracies(test_results, dataset.n_concepts, f"enhanced_{dataset.foundation_model}_cem"))
            cems_with_discovered_concepts.append(model)

        log(get_provided_and_discovered_intervention_accuracies(
            models=cems_with_discovered_concepts,
            datasets=datasets,
            discovered_concept_test_ground_truth=discovered_concept_test_ground_truth,
            model_name_prefix="enhanced",
            model_type="cem"
        ))

    if config["evaluate_models_with_perfect_discovered_concepts"]:
        models_with_perfect_discovered_concepts = []
        for dataset in datasets:
            model, test_results = train_hicem(
                concepts=hicem_concepts,
                n_tasks=dataset.n_tasks,
                latent_representation_size=dataset.latent_representation_size,
                embedding_size=config["hicem_embedding_size"],
                concept_loss_weight=config["hicem_concept_loss_weight"],
                train_dl=dataset.train_dl(discovered_concept_train_ground_truth),
                val_dl=dataset.val_dl(np.full((val_dataset_size, n_matched_discovered_concepts), np.nan)),
                test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
                save_path=run_dir / f"ground_truth_baseline_{dataset.foundation_model}_cem.pth",
                max_epochs=config["max_epochs"],
                use_task_class_weights=config["use_task_class_weights"],
                use_concept_loss_weights=config["use_concept_loss_weights"])
            model_results = get_accuracies(test_results, 0, f"ground_truth_baseline_{dataset.foundation_model}_cem")
            log(model_results)
            models_with_perfect_discovered_concepts.append(model)

        int_baseline_results = get_provided_and_discovered_intervention_accuracies(
            models=models_with_perfect_discovered_concepts,
            datasets=datasets,
            discovered_concept_test_ground_truth=discovered_concept_test_ground_truth,
            model_name_prefix="ground_truth_baseline"
        )
        log(int_baseline_results)

    if config["evaluate_unlabelled_concepts_baseline"]:
        log(run_unlabelled_concepts_baseline(run_dir, config, datasets, sub_concepts))

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    config = load_config(args.config)
    run_dir = args.run_dir

    assert len(config["foundation_models"]) == 1 or config["sub_concept_extraction_method"] == "clustering", "Only one foundation model can be used unless clustering is used for sub-concept extraction."

    datasets = load_datasets(config)

    if args.command == "train-initial-models":
        train_initial_models(run_dir, config, datasets)
    elif args.command == "discover-concepts":
        discover_concepts(run_dir, config, datasets)
    elif args.command == "match-concepts":
        match_concepts(run_dir, config, datasets)
    elif args.command == "train-hicems":
        train_hicems(run_dir, config, datasets)
    elif args.command == "evaluate-interventions":
        evaluate_interventions(run_dir, config, datasets)
    elif args.command == "run-baselines":
        run_baselines(run_dir, config, datasets)
