import argparse
import os
from pathlib import Path
import wandb
import yaml
import numpy as np
import torch
import lightning
import sklearn.metrics
from cemcd.training import train_cem, train_hicem
import cemcd.concept_discovery
from experiment_utils import load_config, load_datasets, train_initial_cems, get_intervention_accuracies

ALPHABET = [
    "ALPHA",
    "BRAVO",
    "CHARLIE",
    "DELTA",
    "ECHO",
    "FOXTROT",
    "GOLF",
    "HOTEL",
    "INDIA",
    "JULIET",
    "KILO",
    "LIMA",
    "MIKE",
    "NOVEMBER",
    "OSCAR",
    "PAPA",
    "QUEBEC",
    "ROMEO",
    "SIERRA",
    "TANGO",
    "UNIFORM",
    "VICTOR",
    "WHISKEY",
    "XRAY",
    "YANKEE",
    "ZULU"
]

def get_accuracies(test_results, n_provided_concepts, model_name):
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
        "-c", "--config", 
        type=str,
        required=True,
        help="Path to the experiment config file.")
    parser.add_argument(
        "-r", "--repeats",
        type=int,
        default=1,
        help="Number of times to run the experiment.")
    return parser.parse_args()

def create_run_name(results_dir, dataset):
    for word1 in ALPHABET:
        for word2 in ALPHABET:
            for word3 in ALPHABET:
                run_name = f"{dataset}-{word1}-{word2}-{word3}"
                if not (Path(results_dir) / run_name).exists():
                    return run_name
    raise RuntimeError("All run names have been used.")

def get_provided_and_discovered_intervention_accuracies(models, datasets, discovered_concept_test_ground_truth, model_name_prefix, model_type="hicem"):
    n_provided_concepts = datasets[0].n_concepts
    n_discovered_concepts = discovered_concept_test_ground_truth.shape[1]
    all_concepts = range(n_provided_concepts + n_discovered_concepts)
    provided_concepts = range(n_provided_concepts)
    discovered_concepts = list(range(n_provided_concepts, n_provided_concepts + n_discovered_concepts))

    results = {}
    for dataset, model in zip(datasets, models):
        model_name = f"{model_name_prefix}_{(dataset.foundation_model)}_{model_type}"

        results[f"{model_name}_discovered_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=False)
        results[f"{model_name}_discovered_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=True)

        results[f"{model_name}_all_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=all_concepts,
            one_at_a_time=False)
        results[f"{model_name}_provided_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=provided_concepts,
            one_at_a_time=True)

    return results

def test_concept_interventions(
        initial_models,
        models_with_discovered_concepts,
        datasets,
        discovered_concept_test_ground_truth):
    results = {}

    for dataset, model in zip(datasets, initial_models):
        model_name = (dataset.foundation_model) + "_cem"
        intervention_accuracies_cumulative = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(),
            concepts_to_intervene=range(model.n_concepts),
            one_at_a_time=False)
        intervention_accuracies_one_at_a_time = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(),
            concepts_to_intervene=range(model.n_concepts),
            one_at_a_time=True)
        results[f"initial_{model_name}_interventions_cumulative"] = intervention_accuracies_cumulative
        results[f"initial_{model_name}_interventions_one_at_a_time"] = intervention_accuracies_one_at_a_time

    results.update(
        get_provided_and_discovered_intervention_accuracies(
            models=models_with_discovered_concepts,
            datasets=datasets,
            discovered_concept_test_ground_truth=discovered_concept_test_ground_truth,
            model_name_prefix="enhanced"
        )
    )

    return results

def log_results(config, run_dir, results):
    with (Path(run_dir) / "results.yaml").open("a") as f:
        yaml.safe_dump(results, f)

    if config["use_wandb"]:
        wandb.log(results)

def match_single_concept_to_concept_bank(labels, dataset, chosen_concept_bank_idxs=None):
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

def match_discovered_concepts_to_concept_bank(
        discovered_concept_labels,
        n_discovered_sub_concepts, 
        concept_bank,
        concept_test_ground_truth,
        concept_names,
        sub_concept_map=None):
    test_datset_size = concept_test_ground_truth.shape[0]
    n_concepts_discovered = discovered_concept_labels.shape[1]
    discovered_concept_train_ground_truth = np.full_like(discovered_concept_labels, np.nan)
    discovered_concept_test_ground_truth = np.full((test_datset_size, n_concepts_discovered), np.nan)
    discovered_concept_semantics = [None] * n_concepts_discovered
    discovered_concept_roc_aucs = [0] * n_concepts_discovered

    for true_concept_idx in range(concept_bank.shape[1]):
        first_discovered_concept_to_check = 0
        n_discovered_concepts_to_check = n_concepts_discovered
        if sub_concept_map is not None:
            for top_concept_idx, n_sub_concepts in enumerate(n_discovered_sub_concepts):
                if true_concept_idx in sub_concept_map[top_concept_idx]:
                    first_discovered_concept_to_check = sum(n_discovered_sub_concepts[:top_concept_idx])
                    n_discovered_concepts_to_check = n_sub_concepts
                    break

        true_concept_labels = concept_bank[:, true_concept_idx]
        best_roc_auc = 0
        best_discovered_concept_idx = None
        for discovered_concept_idx in range(first_discovered_concept_to_check, first_discovered_concept_to_check + n_discovered_concepts_to_check):
            labels = discovered_concept_labels[:, discovered_concept_idx]
            score = sklearn.metrics.roc_auc_score(true_concept_labels, labels)
            if score > best_roc_auc:
                best_roc_auc = score
                best_discovered_concept_idx = discovered_concept_idx
        if best_roc_auc > 0.7 and best_roc_auc > discovered_concept_roc_aucs[best_discovered_concept_idx]:
            discovered_concept_train_ground_truth[:, best_discovered_concept_idx] = true_concept_labels
            discovered_concept_test_ground_truth[:, best_discovered_concept_idx] = concept_test_ground_truth[:, true_concept_idx]
            discovered_concept_semantics[best_discovered_concept_idx] = concept_names[true_concept_idx]
            discovered_concept_roc_aucs[best_discovered_concept_idx] = best_roc_auc

    return (discovered_concept_train_ground_truth,
            discovered_concept_test_ground_truth,
            discovered_concept_semantics,
            discovered_concept_roc_aucs)

def run_unlabelled_concepts_baseline(run_dir, config, datasets, sub_concepts):
    n_top_concepts = len(sub_concepts)
    n_discovered_sub_concepts = sum(map(sum, sub_concepts))
    trainer = lightning.Trainer()

    unlabelled_results = {}
    for dataset in datasets:
        train_dataset_size = len(datasets[0].train_dl().dataset)
        val_dataset_size = len(datasets[0].val_dl().dataset)
        test_dataset_size = len(datasets[0].test_dl().dataset)

        model, _ = train_hicem(
            sub_concepts=sub_concepts,
            n_tasks=dataset.n_tasks,
            latent_representation_size=dataset.latent_representation_size,
            embedding_size=config["hicem_embedding_size"],
            concept_loss_weight=config["hicem_concept_loss_weight"],
            train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_sub_concepts), np.nan)),
            val_dl=dataset.val_dl(np.full((val_dataset_size, n_discovered_sub_concepts), np.nan)),
            test_dl=dataset.test_dl(np.full((test_dataset_size, n_discovered_sub_concepts), np.nan)),
            save_path=run_dir / f"unlabelled_{dataset.foundation_model}_hicem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])

        c_pred, _, _ = cemcd.concept_discovery.calculate_embeddings(model, dataset.train_dl(np.full((train_dataset_size, n_discovered_sub_concepts), np.nan)))

        discovered_concept_test_ground_truth = np.zeros((test_dataset_size, 0))

        sub_concept_idx = n_top_concepts
        for top_concept_idx in range(n_top_concepts):
            for _ in range(sum(sub_concepts[top_concept_idx])):
                if config["only_match_subconcepts"]:
                    _, matching_concept_idx = match_single_concept_to_concept_bank(c_pred[:, sub_concept_idx], dataset, dataset.sub_concept_map[top_concept_idx])
                else:
                    _, matching_concept_idx = match_single_concept_to_concept_bank(c_pred[:, sub_concept_idx], dataset)
                discovered_concept_test_ground_truth = np.concatenate(
                    (discovered_concept_test_ground_truth, np.expand_dims(dataset.concept_test_ground_truth[:, matching_concept_idx], axis=1)),
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

def run_experiment(run_dir, config):
    assert len(config["foundation_models"]) == 1 or config["sub_concept_extraction_method"] == "clustering", "Only one foundation model can be used unless clustering is used for sub-concept extraction."

    run_dir = Path(run_dir)
    datasets = load_datasets(config)
    val_dataset_size = len(datasets[0].val_dl().dataset)
    log = lambda results: log_results(config, run_dir, results)

    if not config["use_foundation_model_representations_instead_of_concept_embeddings"]:
        initial_models, test_results = train_initial_cems(config, datasets, run_dir)

        for dataset, test_result in zip(datasets, test_results):
            model_results = get_accuracies(test_result, dataset.n_concepts, f"initial_{dataset.foundation_model}_cem")
            log(model_results)
    else:
        initial_models = []

    (discovered_concept_labels,
     n_discovered_sub_concepts) = cemcd.concept_discovery.split_concepts(
        config=config,
        initial_models=initial_models,
        datasets=datasets,
        concepts_to_split=range(config["n_concepts_to_split"]))

    (discovered_concept_train_ground_truth,
     discovered_concept_test_ground_truth,
     discovered_concept_semantics,
     discovered_concept_roc_aucs) = match_discovered_concepts_to_concept_bank(
        discovered_concept_labels=discovered_concept_labels,
        n_discovered_sub_concepts=n_discovered_sub_concepts,
        concept_bank=datasets[0].concept_bank,
        concept_test_ground_truth=datasets[0].concept_test_ground_truth,
        concept_names=datasets[0].concept_names,
        sub_concept_map=datasets[0].sub_concept_map if config["only_match_subconcepts"] else None)
    
    log({"n_discovered_sub_concepts": n_discovered_sub_concepts,
         "discovered_concept_semantics": list(map(str, discovered_concept_semantics)),
         "discovered_concept_roc_aucs": list(map(float, discovered_concept_roc_aucs)),})

    np.savez(run_dir / "discovered_concepts.npz",
        discovered_concept_labels=discovered_concept_labels,
        discovered_concept_train_ground_truth=discovered_concept_train_ground_truth,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)

    matched_mask = np.array(discovered_concept_roc_aucs) > 0
    discovered_concept_labels = discovered_concept_labels[:, matched_mask]
    discovered_concept_train_ground_truth = discovered_concept_train_ground_truth[:, matched_mask]
    discovered_concept_test_ground_truth = discovered_concept_test_ground_truth[:, matched_mask]
    n_interpreted_sub_concepts = []
    for n in n_discovered_sub_concepts:
        n_interpreted_sub_concepts.append(np.sum(matched_mask[:n]))
        matched_mask = matched_mask[n:]

    sub_concepts = list(map(lambda n: (n, 0), n_interpreted_sub_concepts)) # We don't split negative embeddings
    models_with_discovered_concepts = []
    for dataset in datasets:
        model, test_results = train_hicem(
            sub_concepts=sub_concepts,
            n_tasks=dataset.n_tasks,
            latent_representation_size=dataset.latent_representation_size,
            embedding_size=config["hicem_embedding_size"],
            concept_loss_weight=config["hicem_concept_loss_weight"],
            train_dl=dataset.train_dl(discovered_concept_labels),
            val_dl=dataset.val_dl(np.full((val_dataset_size, sum(n_interpreted_sub_concepts)), np.nan)),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            save_path=run_dir / f"enhanced_{dataset.foundation_model}_hicem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        log(get_accuracies(test_results, dataset.n_concepts, f"enhanced_{dataset.foundation_model}_hicem"))
        models_with_discovered_concepts.append(model)

    if config["evaluate_interventions"]:
        intervention_results = test_concept_interventions(
            initial_models=initial_models,
            models_with_discovered_concepts=models_with_discovered_concepts,
            datasets=datasets,
            discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)
        log(intervention_results)

    if config["evaluate_cems_with_discovered_concepts"]:
        cems_with_discovered_concepts = []
        for dataset in datasets:
            model, test_results = train_cem(
                n_concepts=dataset.n_concepts + sum(n_interpreted_sub_concepts),
                n_tasks=dataset.n_tasks,
                latent_representation_size=dataset.latent_representation_size,
                embedding_size=config["cem_embedding_size"],
                concept_loss_weight=config["cem_concept_loss_weight"],
                train_dl=dataset.train_dl(discovered_concept_labels),
                val_dl=dataset.val_dl(np.full((val_dataset_size, sum(n_interpreted_sub_concepts)), np.nan)),
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
                sub_concepts=sub_concepts,
                n_tasks=dataset.n_tasks,
                latent_representation_size=dataset.latent_representation_size,
                embedding_size=config["hicem_embedding_size"],
                concept_loss_weight=config["hicem_concept_loss_weight"],
                train_dl=dataset.train_dl(discovered_concept_train_ground_truth),
                val_dl=dataset.val_dl(np.full((val_dataset_size, sum(n_interpreted_sub_concepts)), np.nan)),
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

    repeats = args.repeats
    print(f"Running {repeats} times.")
    for _ in range(repeats):
        config = load_config(args.config)
        run_name = create_run_name(config["results_dir"], config["dataset"])
        print(f"RUN NAME: {run_name}\n")
        run_dir = Path(config["results_dir"]) / run_name
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(Path(args.config).read_text())
        if config["use_wandb"]:
            wandb.init(
                project="cem-concept-discovery",
                config=config,
                name=run_name,
                notes=config["description"])
        run_experiment(
            run_dir=run_dir,
            config=config)

        if config["use_wandb"]:
            wandb.save(os.path.join(run_dir, "*"))
            wandb.finish()
