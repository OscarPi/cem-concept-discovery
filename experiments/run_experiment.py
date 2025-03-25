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

def calculate_c_pred_percentiles(model, model_name, train_dl):
    c_pred, _, _ = cemcd.concept_discovery.calculate_embeddings(model, train_dl)
    percentile_95 = np.percentile(c_pred, 95, axis=0).tolist()
    percentile_5 = np.percentile(c_pred, 5, axis=0).tolist()
    results = {
        f"{model_name}_c_pred_95_percentile": percentile_95,
        f"{model_name}_c_pred_5_percentile": percentile_5,
    }
    return results

def get_intervention_accuracies(model, train_dl, test_dl, concepts_to_intervene, one_at_a_time, model_name):
    trainer = lightning.Trainer()

    c_pred, _, _ = cemcd.concept_discovery.calculate_embeddings(model, train_dl)
    model.intervention_on_value = torch.from_numpy(np.percentile(c_pred, 95, axis=0))
    model.intervention_off_value = torch.from_numpy(np.percentile(c_pred, 5, axis=0))
    print(f"{model_name} interventions on value: {model.intervention_on_value}")
    print(f"{model_name} interventions off value: {model.intervention_off_value}")

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
        discovered_concept_test_ground_truth,
        provided_concepts_removed):
    results = {}

    for dataset, model in zip(datasets, initial_models):
        model_name = (dataset.foundation_model or 'basic') + "cem"
        intervention_accuracies_cumulative = get_intervention_accuracies(
            model=model,
            train_dl=dataset.train_dl(),
            test_dl=dataset.test_dl(),
            concepts_to_intervene=range(model.n_concepts),
            one_at_a_time=False,
            model_name=f"initial_{model_name}")
        intervention_accuracies_one_at_a_time = get_intervention_accuracies(
            model=model,
            train_dl=dataset.train_dl(),
            test_dl=dataset.test_dl(),
            concepts_to_intervene=range(model.n_concepts),
            one_at_a_time=True,
            model_name=f"initial_{model_name}")
        results[f"initial_{model_name}_interventions_cumulative"] = intervention_accuracies_cumulative
        results[f"initial_{model_name}_interventions_one_at_a_time"] = intervention_accuracies_one_at_a_time

    n_provided_concepts = datasets[0].n_concepts
    n_discovered_concepts = discovered_concept_test_ground_truth.shape[1]
    all_concepts = range(n_provided_concepts + n_discovered_concepts)
    provided_concepts = range(n_provided_concepts)
    if provided_concepts_removed:
        discovered_concepts = range(n_discovered_concepts)
    else:
        discovered_concepts = range(n_provided_concepts, n_provided_concepts + n_discovered_concepts)

    for dataset, model in zip(datasets, models_with_discovered_concepts):
        model_name = (dataset.foundation_model or 'basic') + "hicem"
        train_dataset_size = len(dataset.train_dl().dataset)

        results[f"enhanced_{model_name}_discovered_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=model,
            train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_concepts), np.nan), use_provided_concepts=not provided_concepts_removed),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth, use_provided_concepts=not provided_concepts_removed),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=False,
            model_name=f"enhanced_{model_name}")
        results[f"enhanced_{model_name}_discovered_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
            model=model,
            train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_concepts), np.nan), use_provided_concepts=not provided_concepts_removed),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth, use_provided_concepts=not provided_concepts_removed),
            concepts_to_intervene=discovered_concepts,
            one_at_a_time=True,
            model_name=f"enhanced_{model_name}")

        if not provided_concepts_removed:
            results[f"enhanced_{model_name}_all_concept_interventions_cumulative"] = get_intervention_accuracies(
                model=model,
                train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_concepts), np.nan)),
                test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
                concepts_to_intervene=all_concepts,
                one_at_a_time=False,
                model_name=f"enhanced_{model_name}")
            results[f"enhanced_{model_name}_provided_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
                model=model,
                train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_concepts), np.nan)),
                test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
                concepts_to_intervene=provided_concepts,
                one_at_a_time=True,
                model_name=f"enhanced_{model_name}")

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

    # for dataset, model in zip(datasets, initial_models):
    #     log(calculate_c_pred_percentiles(model, f"initial_{dataset.foundation_model or 'basic'}cem", dataset.train_dl()))

    (discovered_concept_labels,
     discovered_concept_train_ground_truth,
     discovered_concept_test_ground_truth,
     discovered_concept_roc_aucs,
     n_discovered_subconcepts) = cemcd.concept_discovery.split_concepts(
        config=config,
        save_path=run_dir,
        initial_models=initial_models,
        datasets=datasets,
        concepts_to_split=[0, 1, 2]) #range(datasets[0].n_concepts)) TODO: CHANGE
    n_discovered_concepts = sum(n_discovered_subconcepts)

    models_with_discovered_concepts = []
    for dataset in datasets:
        model, test_results = train_hicem(
            n_top_concepts=datasets[0].n_concepts,
            n_sub_concepts=n_discovered_subconcepts,
            n_tasks=dataset.n_tasks,
            pre_concept_model=pre_concept_model if dataset.foundation_model is None else None,
            latent_representation_size=dataset.latent_representation_size or list(pre_concept_model.modules())[-1].out_features,
            train_dl=dataset.train_dl(discovered_concept_labels),
            val_dl=dataset.val_dl(np.full((val_dataset_size, n_discovered_concepts), np.nan)),
            test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
            save_path=run_dir / f"enhanced_{dataset.foundation_model or 'basic'}cem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        model_results = get_accuracies(test_results, dataset.n_concepts, f"enhanced_{dataset.foundation_model or 'basic'}hicem")
        log(model_results)
        # log(calculate_c_pred_percentiles(model, f"enhanced_{dataset.foundation_model or 'basic'}cem", dataset.train_dl(discovered_concept_labels, use_provided_concepts=False)))
        models_with_discovered_concepts.append(model)

    intervention_results = test_concept_interventions(
        initial_models=initial_models,
        models_with_discovered_concepts=models_with_discovered_concepts,
        datasets=datasets,
        discovered_concept_test_ground_truth=discovered_concept_test_ground_truth,
        provided_concepts_removed=False)
    log(intervention_results)

    # models_with_perfect_discovered_concepts = []
    # for dataset in datasets:
    #     model, test_results = train_cem(
    #         n_concepts=n_discovered_concepts,
    #         n_tasks=dataset.n_tasks,
    #         pre_concept_model=pre_concept_model if dataset.foundation_model is None else None,
    #         latent_representation_size=dataset.latent_representation_size or list(pre_concept_model.modules())[-1].out_features,
    #         train_dl=dataset.train_dl(discovered_concept_train_ground_truth, use_provided_concepts=False),
    #         val_dl=dataset.val_dl(np.full((val_dataset_size, n_discovered_concepts), np.nan), use_provided_concepts=False),
    #         test_dl=dataset.test_dl(discovered_concept_test_ground_truth, use_provided_concepts=False),
    #         save_path=run_dir / f"ground_truth_baseline_{dataset.foundation_model or 'basic'}cem.pth",
    #         max_epochs=config["max_epochs"],
    #         use_task_class_weights=config["use_task_class_weights"],
    #         use_concept_loss_weights=config["use_concept_loss_weights"])
    #     model_results = get_accuracies(test_results, 0, f"ground_truth_baseline_{dataset.foundation_model or 'basic'}cem")
    #     log(model_results)
    #     log(calculate_c_pred_percentiles(model, f"ground_truth_baseline_{dataset.foundation_model or 'basic'}cem", dataset.train_dl(discovered_concept_train_ground_truth, use_provided_concepts=False)))
    #     models_with_perfect_discovered_concepts.append(model)

    # int_baseline_results = {}
    # # all_concepts = range(datasets[0].n_concepts + n_discovered_concepts)
    # # provided_concepts = range(datasets[0].n_concepts)
    # discovered_concepts = range(n_discovered_concepts)
    # for dataset, model in zip(datasets, models_with_perfect_discovered_concepts):
    #     train_dataset_size = len(dataset.train_dl().dataset)
    #     model_name = (dataset.foundation_model or 'basic') + "cem"
    #     # int_baseline_results[f"ground_truth_baseline_{model_name}_all_concept_interventions_cumulative"] = get_intervention_accuracies(
    #     #     model=model,
    #     #     train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_concepts), np.nan)),
    #     #     test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
    #     #     concepts_to_intervene=all_concepts,
    #     #     one_at_a_time=False,
    #     #     model_name=f"ground_truth_baseline_{model_name}")
    #     int_baseline_results[f"ground_truth_baseline_{model_name}_discovered_concept_interventions_cumulative"] = get_intervention_accuracies(
    #         model=model,
    #         train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_concepts), np.nan), use_provided_concepts=False),
    #         test_dl=dataset.test_dl(discovered_concept_test_ground_truth, use_provided_concepts=False),
    #         concepts_to_intervene=discovered_concepts,
    #         one_at_a_time=False,
    #         model_name=f"ground_truth_baseline_{model_name}")
    #     # int_baseline_results[f"ground_truth_baseline_{model_name}_provided_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
    #     #     model=model,
    #     #     train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_concepts), np.nan)),
    #     #     test_dl=dataset.test_dl(discovered_concept_test_ground_truth),
    #     #     concepts_to_intervene=provided_concepts,
    #     #     one_at_a_time=True,
    #     #     model_name=f"ground_truth_baseline_{model_name}")
    #     int_baseline_results[f"ground_truth_baseline_{model_name}_discovered_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
    #         model=model,
    #         train_dl=dataset.train_dl(np.full((train_dataset_size, n_discovered_concepts), np.nan), use_provided_concepts=False),
    #         test_dl=dataset.test_dl(discovered_concept_test_ground_truth, use_provided_concepts=False),
    #         concepts_to_intervene=discovered_concepts,
    #         one_at_a_time=True,
    #         model_name=f"ground_truth_baseline_{model_name}")
    # log(int_baseline_results)

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
