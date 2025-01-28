import argparse
from pathlib import Path
import yaml
import torch
from cemcd.training import train_cbm, train_cem, train_black_box
from cemcd.data import mnist, dsprites, cub, awa, celeba
from experiment_utils import load_config, load_datasets, make_concept_model, make_pre_concept_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", 
        type=str,
        required=True,
        help="Path to the experiment config file.")
    parser.add_argument(
        "-r", "--results-dir",
        type=Path,
        default=1,
        help="Path to directory to store results in.")
    return parser.parse_args()

def load_datasets(config):
    if config["dataset"] == "mnist_add":
        mnist_config = config["mnist_config"]
        vanilla_dataset = mnist.MNISTDatasets(
            n_digits=mnist_config["n_digits"],
            max_digit=mnist_config["max_digit"],
            foundation_model=None,
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
        dino_dataset = mnist.MNISTDatasets(
            n_digits=mnist_config["n_digits"],
            max_digit=mnist_config["max_digit"],
            foundation_model="dinov2",
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
    elif config["dataset"] == "dsprites":
        vanilla_dataset = dsprites.DSpritesDatasets(
            foundation_model=None,
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
        dino_dataset = dsprites.DSpritesDatasets(
            foundation_model="dinov2",
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
    elif config["dataset"] == "cub":
        vanilla_dataset = cub.CUBDatasets(
            foundation_model=None,
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
        dino_dataset = cub.CUBDatasets(
            foundation_model="dinov2",
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
    elif config["dataset"] == "awa":
        vanilla_dataset = awa.AwADatasets(
            foundation_model=None,
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
        dino_dataset = awa.AwADatasets(
            foundation_model="dinov2",
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
    elif config["dataset"] == "celeba":
        vanilla_dataset = celeba.CELEBADatasets(
            foundation_model=None,
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
        dino_dataset = celeba.CELEBADatasets(
            foundation_model="dinov2",
            dataset_dir=config["dataset_dir"],
            model_dir=config["model_dir"])
    else:
        raise ValueError(f"Unrecognised dataset: {config['dataset']}")
    
    return vanilla_dataset, dino_dataset

def run_baselines(results_dir, config):
    results_dir = Path(results_dir)
    results = {}

    vanilla_dataset, dino_dataset = load_datasets(config)

    pre_concept_model = make_pre_concept_model(config)
    _, cem_test_results = train_cem(
        n_concepts=vanilla_dataset.n_concepts,
        n_tasks=vanilla_dataset.n_tasks,
        pre_concept_model=pre_concept_model,
        latent_representation_size=list(pre_concept_model.modules())[-1].out_features,
        train_dl=vanilla_dataset.train_dl(),
        val_dl=vanilla_dataset.val_dl(),
        test_dl=vanilla_dataset.test_dl(),
        save_path=results_dir / "cem_baseline.pth",
        max_epochs=config["max_epochs"],
        use_task_class_weights=config["use_task_class_weights"],
        use_concept_loss_weights=config["use_concept_loss_weights"])
    cem_task_accuracy = round(cem_test_results["test_y_accuracy"], 4)
    cem_concept_auc = round(cem_test_results["test_c_auc"], 4)
    results.update({
        "cem_task_accuracy": float(cem_task_accuracy),
        "cem_concept_auc": float(cem_concept_auc)
    })

    concept_model = make_concept_model(config, vanilla_dataset.n_concepts)
    _, cbm_test_results = train_cbm(
        n_concepts=vanilla_dataset.n_concepts,
        n_tasks=vanilla_dataset.n_tasks,
        concept_model=concept_model,
        train_dl=vanilla_dataset.train_dl(),
        val_dl=vanilla_dataset.val_dl(),
        test_dl=vanilla_dataset.test_dl(),
        black_box=False,
        save_path=results_dir / "cbm_baseline.pth",
        max_epochs=config["max_epochs"],
        use_task_class_weights=config["use_task_class_weights"],
        use_concept_loss_weights=config["use_concept_loss_weights"])
    cbm_task_accuracy = round(cbm_test_results["test_y_accuracy"], 4)
    cbm_concept_auc = round(cbm_test_results["test_c_auc"], 4)
    results.update({
        "cbm_task_accuracy": float(cbm_task_accuracy),
        "cbm_concept_auc": float(cbm_concept_auc)
    })

    _, black_box_test_results = train_black_box(
        n_tasks=dino_dataset.n_tasks,
        latent_representation_size=dino_dataset.latent_representation_size,
        train_dl=dino_dataset.train_dl(),
        val_dl=dino_dataset.val_dl(),
        test_dl=dino_dataset.test_dl(),
        save_path=None,
        max_epochs=config["max_epochs"],
        use_task_class_weights=config["use_task_class_weights"])
    black_box_task_accuracy = round(black_box_test_results["test_y_accuracy"], 4)
    results.update({"black_box_task_accuracy": float(black_box_task_accuracy)})

    with (results_dir / "baseline_results.yaml").open("w") as f:
        yaml.safe_dump(results, f)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    config = load_config(args.config)
    run_baselines(args.results_dir, config)
