import argparse
import os
from pathlib import Path
import wandb
import yaml
import numpy as np
import torch
import lightning
from cemcd.training import train_cem
import cemcd.concept_discovery
from experiment_utils import load_config, load_datasets, make_pre_concept_model, get_initial_models

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
        f"{model_name}_provided_concept_auc": float(provided_concept_auc),
        f"{model_name}_provided_concept_aucs": list(map(lambda x: round(float(x), 4), provided_concept_aucs))}
    if len(discovered_concept_accuracies) > 0:
        results.update({
            f"{model_name}_discovered_concept_accuracy": float(discovered_concept_accuracy),
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

def get_intervention_accuracies(model, test_dl, concepts_to_intervene, one_at_a_time):
    trainer = lightning.Trainer()
    intervention_accuracies = []
    model.intervention_mask = torch.tensor([0] * model.n_concepts)
    [test_results] = trainer.test(model, test_dl)
    initial_task_accuracy = test_results["test_y_accuracy"]
    for c in concepts_to_intervene:
        if one_at_a_time:
            model.intervention_mask = torch.tensor([0] * model.n_concepts)
        model.intervention_mask[c] = 1
        [test_results] = trainer.test(model, test_dl)
        accuracy_difference = round(test_results["test_y_accuracy"] - initial_task_accuracy, 4)
        intervention_accuracies.append(accuracy_difference)
    return intervention_accuracies

def test_concept_interventions(
        initial_models,
        models_with_discovered_concepts,
        datasets,
        discovered_concept_test_ground_truth):
    results = {}

    for dataset, model in zip(datasets, initial_models):
        model_name = (dataset.foundation_model or 'basic') + "cem"
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

    n_provided_concepts = datasets[0].n_concepts
    n_discovered_concepts = discovered_concept_test_ground_truth.shape[1]
    all_concepts = range(n_provided_concepts + n_discovered_concepts)
    provided_concepts = range(n_provided_concepts)
    discovered_concepts = range(n_provided_concepts, n_provided_concepts + n_discovered_concepts)

    for dataset, model in zip(datasets, models_with_discovered_concepts):
        model_name = (dataset.foundation_model or 'basic') + "cem"
        results[f"enhanced_{model_name}_all_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=all_concepts,
            one_at_a_time=False)
        results[f"enhanced_{model_name}_discovered_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=False)
        results[f"enhanced_{model_name}_provided_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=provided_concepts,
            one_at_a_time=True)
        results[f"enhanced_{model_name}_discovered_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=True)

    return results

def log_results(config, run_dir, results):
    with (Path(run_dir) / "results.yaml").open("a") as f:
        yaml.safe_dump(results, f)

    if config["use_wandb"]:
        wandb.log(results)

def run_experiment(run_dir, config):
    run_dir = Path(run_dir)
    datasets = load_datasets(config)
    val_dataset_size = len(datasets[0].val_dl().dataset)
    log = lambda results: log_results(config, run_dir, results)

    pre_concept_model = make_pre_concept_model(config)

    initial_models, test_results = get_initial_models(config, datasets, run_dir)

    for dataset, test_result in zip(datasets, test_results):
        model_results = get_accuracies(test_result, dataset.n_concepts, f"initial_{dataset.foundation_model or 'basic'}cem")
        log(model_results)

    discovered_concept_labels, discovered_concept_train_ground_truth, discovered_concept_test_ground_truth, discovered_concept_roc_aucs = cemcd.concept_discovery.discover_concepts(
        config=config,
        save_path=run_dir,
        initial_models=initial_models,
        datasets=datasets)
    n_discovered_concepts = len(discovered_concept_roc_aucs)

    models_with_discovered_concepts = []
    for dataset in datasets:
        model, test_results = train_cem(
            n_concepts=dataset.n_concepts + n_discovered_concepts,
            n_tasks=dataset.n_tasks,
            pre_concept_model=pre_concept_model if dataset.foundation_model is None else None,
            latent_representation_size=dataset.latent_representation_size or list(pre_concept_model.modules())[-1].out_features,
            train_dl=dataset.train_dl(discovered_concept_labels),
            val_dl=dataset.val_dl(np.full((val_dataset_size, n_discovered_concepts), np.nan)),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            save_path=run_dir / f"enhanced_{dataset.foundation_model or 'basic'}cem.pth",
            max_epochs=config["max_epochs"])
        model_results = get_accuracies(test_results, dataset.n_concepts, f"enhanced_{dataset.foundation_model or 'basic'}cem")
        log(model_results)
        models_with_discovered_concepts.append(model)

    intervention_results = test_concept_interventions(
        initial_models=initial_models,
        models_with_discovered_concepts=models_with_discovered_concepts,
        datasets=datasets,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth)
    log(intervention_results)

    models_with_perfect_discovered_concepts = []
    for dataset in datasets:
        model, test_results = train_cem(
            n_concepts=dataset.n_concepts + n_discovered_concepts,
            n_tasks=dataset.n_tasks,
            pre_concept_model=pre_concept_model if dataset.foundation_model is None else None,
            latent_representation_size=dataset.latent_representation_size or list(pre_concept_model.modules())[-1].out_features,
            train_dl=dataset.train_dl(discovered_concept_train_ground_truth),
            val_dl=dataset.val_dl(np.full((val_dataset_size, n_discovered_concepts), np.nan)),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            save_path=run_dir / f"ground_truth_baseline_{dataset.foundation_model or 'basic'}cem.pth",
            max_epochs=config["max_epochs"])
        model_results = get_accuracies(test_results, dataset.n_concepts, f"ground_truth_baseline_{dataset.foundation_model or 'basic'}cem")
        log(model_results)
        models_with_perfect_discovered_concepts.append(model)

    int_baseline_results = {}
    all_concepts = range(datasets[0].n_concepts + n_discovered_concepts)
    provided_concepts = range(datasets[0].n_concepts)
    discovered_concepts = range(datasets[0].n_concepts, datasets[0].n_concepts + n_discovered_concepts)
    for dataset, model in zip(datasets, models_with_perfect_discovered_concepts):
        model_name = (dataset.foundation_model or 'basic') + "cem"
        int_baseline_results[f"ground_truth_baseline_{model_name}_all_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=all_concepts,
            one_at_a_time=False)
        int_baseline_results[f"ground_truth_baseline_{model_name}_discovered_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=False)
        int_baseline_results[f"ground_truth_baseline_{model_name}_provided_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=provided_concepts,
            one_at_a_time=True)
        int_baseline_results[f"ground_truth_baseline_{model_name}_discovered_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=True)
    log(int_baseline_results)

  
    # concept_model = make_concept_model(config, datasets[0].n_concepts)
    # cbm, cbm_test_results = train_cbm(
    #     n_concepts=datasets[0].n_concepts,
    #     n_tasks=datasets[0].n_tasks,
    #     concept_model=concept_model,
    #     train_dl=datasets[0].train_dl(),
    #     val_dl=datasets[0].val_dl(),
    #     test_dl=datasets[0].test_dl(),
    #     black_box=False,
    #     save_path=run_dir / "cbm_baseline.pth",
    #     max_epochs=config["max_epochs"])
    # cbm_task_accuracy = round(cbm_test_results["test_y_accuracy"], 4)
    # cbm_concept_auc = round(cbm_test_results["test_c_auc"], 4)
    # cbm_intervention_accuracies = get_intervention_accuracies(
    #     model=cbm,
    #     test_dl=datasets[0].test_dl(),
    #     concepts_to_intervene=range(cbm.n_concepts))
    # log({
    #     "cbm_task_accuracy": cbm_task_accuracy,
    #     "cbm_concept_auc": cbm_concept_auc,
    #     "cbm_intervention_accuracies": cbm_intervention_accuracies
    # })

    # # CBM with concept loss weight of 0 is a black box
    # _, black_box_test_results = train_cbm(
    #     n_concepts=list(pre_concept_model.modules())[-1].out_features,
    #     n_tasks=datasets[0].n_tasks,
    #     concept_model=pre_concept_model,
    #     train_dl=datasets[0].train_dl(),
    #     val_dl=datasets[0].val_dl(),
    #     test_dl=datasets[0].test_dl(),
    #     black_box=True,
    #     save_path=run_dir / "black_box_baseline.pth",
    #     max_epochs=config["max_epochs"])
    # black_box_task_accuracy = round(black_box_test_results["test_y_accuracy"], 4)
    # log({"black_box_task_accuracy": black_box_task_accuracy})

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
