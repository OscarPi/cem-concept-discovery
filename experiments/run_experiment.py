import argparse
import os
from pathlib import Path
import wandb
import yaml
import numpy as np
import torch
import lightning
from cemcd.training import train_cem, train_hicem
import cemcd.concept_discovery
from experiment_utils import load_config, load_datasets, make_pre_concept_model, get_initial_models, get_intervention_accuracies

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
        model_name = f"{model_name_prefix}_{(dataset.foundation_model or 'basic')}_{model_type}"

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
        model_name = (dataset.foundation_model or 'basic') + "_cem"
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
            pre_concept_model=None,
            latent_representation_size=dataset.latent_representation_size,
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
                if config["only_discover_subconcepts"]:
                    _, matching_concept_idx = cemcd.concept_discovery.match_to_concept_bank(c_pred[:, sub_concept_idx], dataset, dataset.sub_concept_map[top_concept_idx])
                else:
                    _, matching_concept_idx = cemcd.concept_discovery.match_to_concept_bank(c_pred[:, sub_concept_idx], dataset)
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
    run_dir = Path(run_dir)
    datasets = load_datasets(config)
    val_dataset_size = len(datasets[0].val_dl().dataset)
    log = lambda results: log_results(config, run_dir, results)

    pre_concept_model = make_pre_concept_model(config)

    if not config["cluster_representations"]:
        initial_models, test_results = get_initial_models(config, datasets, run_dir)

        for dataset, test_result in zip(datasets, test_results):
            model_results = get_accuracies(test_result, dataset.n_concepts, f"initial_{dataset.foundation_model or 'basic'}_cem")
            log(model_results)
    else:
        initial_models = []

    (discovered_concept_labels,
     discovered_concept_train_ground_truth,
     discovered_concept_test_ground_truth,
     discovered_concept_roc_aucs,
     n_discovered_sub_concepts) = cemcd.concept_discovery.split_concepts(
        config=config,
        save_path=run_dir,
        initial_models=initial_models,
        datasets=datasets,
        concepts_to_split=range(4))#range(datasets[0].n_concepts)) TODO
    n_discovered_concepts = discovered_concept_labels.shape[1]
    n_discovered_top_concepts = n_discovered_concepts - sum(n_discovered_sub_concepts)

    sub_concepts = list(map(lambda n: (n, 0), n_discovered_sub_concepts)) # We don't split negative embeddings
    for _ in range(n_discovered_top_concepts):
        sub_concepts.append((0, 0))
    models_with_discovered_concepts = []
    for dataset in datasets:
        model, test_results = train_hicem(
            sub_concepts=sub_concepts,
            n_tasks=dataset.n_tasks,
            pre_concept_model=pre_concept_model if dataset.foundation_model is None else None,
            latent_representation_size=dataset.latent_representation_size or list(pre_concept_model.modules())[-1].out_features,
            train_dl=dataset.train_dl(discovered_concept_labels),
            val_dl=dataset.val_dl(np.full((val_dataset_size, n_discovered_concepts), np.nan)),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            save_path=run_dir / f"enhanced_{dataset.foundation_model or 'basic'}_hicem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        log(get_accuracies(test_results, dataset.n_concepts, f"enhanced_{dataset.foundation_model or 'basic'}_hicem"))
        models_with_discovered_concepts.append(model)

    intervention_results = test_concept_interventions(
        initial_models=initial_models,
        models_with_discovered_concepts=models_with_discovered_concepts,
        datasets=datasets,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)
    log(intervention_results)

    cems_with_discovered_concepts = []
    for dataset in datasets:
        model, test_results = train_cem(
            n_concepts=dataset.n_concepts + n_discovered_concepts,
            n_tasks=dataset.n_tasks,
            pre_concept_model=pre_concept_model if dataset.foundation_model is None else None,
            latent_representation_size=dataset.latent_representation_size or list(pre_concept_model.modules())[-1].out_features,
            train_dl=dataset.train_dl(discovered_concept_labels),
            val_dl=dataset.val_dl(np.full((val_dataset_size, n_discovered_concepts), np.nan)),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            save_path=run_dir / f"enhanced_{dataset.foundation_model or 'basic'}_cem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        log(get_accuracies(test_results, dataset.n_concepts, f"enhanced_{dataset.foundation_model or 'basic'}_cem"))
        cems_with_discovered_concepts.append(model)

    log(get_provided_and_discovered_intervention_accuracies(
        models=cems_with_discovered_concepts,
        datasets=datasets,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth,
        model_name_prefix="enhanced",
        model_type="cem"
    ))

    models_with_perfect_discovered_concepts = []
    for dataset in datasets:
        model, test_results = train_hicem(
            sub_concepts=sub_concepts,
            n_tasks=dataset.n_tasks,
            pre_concept_model=pre_concept_model if dataset.foundation_model is None else None,
            latent_representation_size=dataset.latent_representation_size or list(pre_concept_model.modules())[-1].out_features,
            train_dl=dataset.train_dl(discovered_concept_train_ground_truth),
            val_dl=dataset.val_dl(np.full((val_dataset_size, n_discovered_concepts), np.nan)),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            save_path=run_dir / f"ground_truth_baseline_{dataset.foundation_model or 'basic'}_cem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        model_results = get_accuracies(test_results, 0, f"ground_truth_baseline_{dataset.foundation_model or 'basic'}_cem")
        log(model_results)
        models_with_perfect_discovered_concepts.append(model)

    int_baseline_results = get_provided_and_discovered_intervention_accuracies(
        models=models_with_perfect_discovered_concepts,
        datasets=datasets,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth,
        model_name_prefix="ground_truth_baseline"
    )
    log(int_baseline_results)

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
