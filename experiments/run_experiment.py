import argparse
import os
from cemcd.training import train_cbm
from cemcd.models.pre_concept_models import get_pre_concept_model
import cemcd.data.mnist as mnist
import cemcd.data.dsprites as dsprites
import cemcd.data.cub as cub
import cemcd.data.awa as awa
import cemcd.concept_discovery
import numpy as np
import lightning
import torch
from torchvision.models import resnet34
from pathlib import Path
import wandb
import yaml

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-c', '--config', 
        type=str, 
        help="Path to the experiment config file."
    )
    group.add_argument(
        '--resume', 
        type=str, 
        metavar='RUN_DIR',
        help="Resume an experiment from the given run directory."
    )
    return parser.parse_args()

def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def load_datasets(config):
    if config["dataset"] == "mnist_add":
        mnist_config = config["mnist_config"]
        return mnist.MNISTDatasets(
            n_digits=mnist_config["n_digits"],
            max_digit=mnist_config["max_digit"],
            dataset_dir=config["dataset_dir"],
        )
    elif config["dataset"] == "dsprites":
        return dsprites.DSpritesDatasets(dataset_dir=config["dataset_dir"])
    elif config["dataset"] == "cub":
        return cub.CUBDatasets(dataset_dir=config["dataset_dir"])
    elif config["dataset"] == "awa":
        return awa.AwADatasets()

    raise ValueError(f"Unrecognised dataset: {config['dataset']}")

def create_run_name(results_dir):
    run_name = f"{np.random.choice(ALPHABET)}-{np.random.choice(ALPHABET)}"
    while (Path(results_dir) / run_name).exists():
        run_name = f"{np.random.choice(ALPHABET)}-{np.random.choice(ALPHABET)}"
    return run_name

def make_pre_concept_model(config):
    if config["pre_concept_model"] == "cnn":
        cnn_config = config["pre_concept_cnn_config"]
        return get_pre_concept_model(cnn_config["width"], cnn_config["height"], cnn_config["channels"])
    elif config["pre_concept_model"] == "resnet34":
        return resnet34(pretrained=True)

    raise ValueError(f"Unknown pre concept model: {config['pre_concept_model']}")

def make_concept_model(config, n_concepts):
    if config["pre_concept_model"] == "cnn":
        cnn_config = config["pre_concept_cnn_config"]
        return get_pre_concept_model(cnn_config["width"], cnn_config["height"], cnn_config["channels"], n_concepts)
    elif config["pre_concept_model"] == "resnet34":
        return torch.nn.Sequential(resnet34(pretrained=True), torch.nn.Linear(1000, n_concepts))

    raise ValueError(f"Unknown pre concept model: {config['pre_concept_model']}")

def run_experiment(resume, run_dir, config):
    run_dir = Path(run_dir)
    datasets = load_datasets(config)

    pre_concept_model = make_pre_concept_model(config)
    model, model_0, n_discovered_concepts, discovered_concept_test_ground_truth = cemcd.concept_discovery.discover_multiple_concepts(
        config=config,
        resume=resume,
        pre_concept_model=pre_concept_model,
        save_path=run_dir,
        datasets=datasets)

    trainer = lightning.Trainer()

    concept_intervention_accuracies = []
    model.intervention_mask = torch.tensor([0] * model.n_concepts)
    [test_results] = trainer.test(model, datasets.test_dl(discovered_concept_test_ground_truth))
    task_accuracy = round(test_results["test_y_accuracy"], 4)
    concept_intervention_accuracies.append(task_accuracy)
    for i in range(model.n_concepts):
        model.intervention_mask[i] = 1
        [test_results] = trainer.test(model, datasets.test_dl(discovered_concept_test_ground_truth))
        task_accuracy = round(test_results["test_y_accuracy"], 4)
        concept_intervention_accuracies.append(task_accuracy)

    discovered_concept_intervention_accuracies = []
    model.intervention_mask = torch.tensor([0] * model.n_concepts)
    [test_results] = trainer.test(model, datasets.test_dl(discovered_concept_test_ground_truth))
    task_accuracy = round(test_results["test_y_accuracy"], 4)
    discovered_concept_intervention_accuracies.append(task_accuracy)
    for i in range(n_discovered_concepts):
        model.intervention_mask[i + datasets.n_concepts] = 1
        [test_results] = trainer.test(model, datasets.test_dl(discovered_concept_test_ground_truth))
        task_accuracy = round(test_results["test_y_accuracy"], 4)
        discovered_concept_intervention_accuracies.append(task_accuracy)

    cem_intervention_accuracies = []
    for i in range(datasets.n_concepts + 1):
        model_0.intervention_mask = torch.tensor([1] * i + [0] * (model_0.n_concepts - i))
        [test_results] = trainer.test(model_0, datasets.test_dl())
        task_accuracy = round(test_results["test_y_accuracy"], 4)
        cem_intervention_accuracies.append(task_accuracy)

    concept_model = make_concept_model(config, datasets.n_concepts)
    cbm, cbm_test_results = train_cbm(
        n_concepts=datasets.n_concepts,
        n_tasks=datasets.n_tasks,
        concept_model=concept_model,
        train_dl=datasets.train_dl(),
        val_dl=datasets.val_dl(),
        test_dl=datasets.test_dl(),
        black_box=False,
        save_path=run_dir / "cbm_baseline.pth",
        max_epochs=config["max_epochs"])
    cbm_task_accuracy = round(cbm_test_results["test_y_accuracy"], 4)
    cbm_concept_auc = round(cbm_test_results["test_c_auc"], 4)

    cbm_intervention_accuracies = []
    for i in range(datasets.n_concepts + 1):
        cbm.intervention_mask = torch.tensor([1] * i + [0] * (cbm.n_concepts - i))
        [test_results] = trainer.test(cbm, datasets.test_dl())
        task_accuracy = round(test_results["test_y_accuracy"], 4)
        cbm_intervention_accuracies.append(task_accuracy)

    # CBM with concept loss weight of 0 is a black box
    _, black_box_test_results = train_cbm(
        n_concepts=list(pre_concept_model.modules())[-1].out_features,
        n_tasks=datasets.n_tasks,
        concept_model=pre_concept_model,
        train_dl=datasets.train_dl(),
        val_dl=datasets.val_dl(),
        test_dl=datasets.test_dl(),
        black_box=True,
        save_path=run_dir / "black_box_baseline.pth",
        max_epochs=config["max_epochs"]
    )
    black_box_task_accuracy = round(black_box_test_results["test_y_accuracy"], 4)

    results = {
        "all_concept_intervention_accuracies": concept_intervention_accuracies,
        "discovered_concept_intervention_accuracies": discovered_concept_intervention_accuracies,
        "cem_intervention_accuracies": cem_intervention_accuracies,
        "cbm_intervention_accuracies": cbm_intervention_accuracies,
        "cbm_task_accuracy": cbm_task_accuracy,
        "cbm_concept_auc": cbm_concept_auc,
        "black_box_task_accuracy": black_box_task_accuracy
    }
    with (run_dir / "results.yaml").open("a") as f:
        yaml.safe_dump(results, f)

    if config["use_wandb"]:
        wandb.log({
            "concept_intervention_accuracies": wandb.plot.line_series(
                xs=list(range(len(concept_intervention_accuracies))), 
                ys=[concept_intervention_accuracies, discovered_concept_intervention_accuracies, cem_intervention_accuracies, cbm_intervention_accuracies],
                keys=["All concepts intervened", "Discovered concepts intervened", "Regular CEM interventions", "CBM interventions"],
                title="Concept interventions",
                xname="Concepts intervened"),
            "cbm_task_accuracy": cbm_task_accuracy,
            "cbm_concept_auc": cbm_concept_auc,
            "black_box_task_accuracy": black_box_task_accuracy
        }, commit=False)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    if args.config:
        config = load_config(args.config)
        run_name = create_run_name(config["results_dir"])
        print(f"RUN NAME: {run_name}\n")
        run_dir = Path(config["results_dir"]) / run_name
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(Path(args.config).read_text())
        if config["use_wandb"]:
            wandb.init(
                project="cem-concept-discovery",
                config=config,
                name=run_name,
                notes=config["description"],
                id=run_name
            )
        run_experiment(
            resume=False,
            run_dir = run_dir,
            config=config
        )

    elif args.resume:
        run_dir = Path(args.resume)
        if (run_dir / "results.yaml").exists():
            raise RuntimeError("Cannot resume a run that has completed.")
        run_name = run_dir.name
        config = load_config(run_dir / "config.yaml")
        if config["use_wandb"]:
            wandb.init(
                project="cem-concept-discovery",
                id=run_name,
                resume="must"
            )
        run_experiment(
            resume=True,
            run_dir=run_dir,
            config=config
        )
    
    if config["use_wandb"]:
        wandb.save(os.path.join(run_dir, "*"))
