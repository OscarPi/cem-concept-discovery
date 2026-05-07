import argparse
from pathlib import Path
import yaml
import wandb
import torch
from cemcd.training import train_cbm, train_black_box
from cemcd.data import get_latent_representation_size
from experiment_utils import load_config, load_datasets, get_intervention_accuracies

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", 
        type=Path,
        required=True,
        help="Path to the experiment config file.")
    return parser.parse_args()

def run_baselines(config):
    results = {}

    datasets = load_datasets(config)

    results_dir = Path(config["results_dir"]) / f"{config['dataset']}_baselines"
    results_dir.mkdir(exist_ok=True)
    for run_number in range(1, 11):
        if (results_dir / f"baseline_results_{run_number}.yaml").exists():
            continue

        all_ok = True
        for foundation_model in config["foundation_models"]:
            if (results_dir / f"{foundation_model}_cbm_baseline_{run_number}.pth").exists():
                all_ok = False
                break
            if (results_dir / f"{foundation_model}_black_box_baseline_{run_number}.pth").exists():
                all_ok = False
                break

        if all_ok:
            break
    else:
        print("Could not create results files: too many already.")
        return

    for foundation_model in config["foundation_models"]:
        cbm, cbm_test_results = train_cbm(
            n_concepts=datasets.n_concepts,
            concept_names=datasets.concept_names,
            n_tasks=datasets.n_tasks,
            latent_representation_size=get_latent_representation_size(foundation_model),
            concept_loss_weight=config["cbm_concept_loss_weight"],
            train_dl=datasets.get_dataloader("train", foundation_model=foundation_model),
            val_dl=datasets.get_dataloader("val", foundation_model=foundation_model),
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model),
            save_path=results_dir / f"{foundation_model}_cbm_baseline_{run_number}.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        cbm_task_accuracy = round(cbm_test_results["test_y_accuracy"], 4)
        cbm_concept_auc = round(cbm_test_results["test_c_auc"], 4)
        # results[f"{foundation_model}_cbm_concept_interventions_one_at_a_time"] = get_intervention_accuracies(
        #     model=cbm,
        #     test_dl=datasets.get_dataloader("test", foundation_model=foundation_model),
        #     concepts_to_intervene=range(datasets.n_concepts),
        #     one_at_a_time=True
        # )
        results[f"{foundation_model}_cbm_concept_interventions_cumulative"] = get_intervention_accuracies(
            model=cbm,
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model),
            concepts_to_intervene=range(datasets.n_concepts),
            one_at_a_time=False
        )
        results.update({
            f"{foundation_model}_cbm_task_accuracy": float(cbm_task_accuracy),
            f"{foundation_model}_cbm_concept_auc": float(cbm_concept_auc)
        })

        _, black_box_test_results = train_black_box(
            n_concepts=datasets.n_concepts, # Determines the shape of the architecture, black box so no concept supervision is used.
            n_tasks=datasets.n_tasks,
            latent_representation_size=get_latent_representation_size(foundation_model),
            embedding_size=config["cem_embedding_size"],
            train_dl=datasets.get_dataloader("train", foundation_model=foundation_model),
            val_dl=datasets.get_dataloader("val", foundation_model=foundation_model),
            test_dl=datasets.get_dataloader("test", foundation_model=foundation_model),
            save_path=results_dir / f"{foundation_model}_black_box_baseline_{run_number}.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"])
        black_box_task_accuracy = round(black_box_test_results["test_y_accuracy"], 4)
        results.update({f"{foundation_model}_black_box_task_accuracy": float(black_box_task_accuracy)})

    with (results_dir / f"baseline_results_{run_number}.yaml").open("w") as f:
        yaml.safe_dump(results, f)
    if config["use_wandb"]:
        wandb.log(results)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    config = load_config(args.config)

    if config["use_wandb"]:
        wandb.init(
            project="cem-concept-discovery-baselines",
            config=config,
            notes="Baseline run")

    run_baselines(config)
