import argparse
import os
from pathlib import Path
import wandb
import torch
from experiment_utils import load_config, load_datasets
from run import train_initial_models, discover_and_match_concepts, train_hicems, evaluate_interventions, run_baselines

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
    parser.add_argument(
        "-c", "--config", 
        type=Path,
        required=True,
        help="Path to the experiment config file.")

    return parser.parse_args()

def create_run_name(results_dir, dataset):
    for word1 in ALPHABET:
        for word2 in ALPHABET:
            for word3 in ALPHABET:
                run_name = f"{dataset}-{word1}-{word2}-{word3}"
                if not (Path(results_dir) / run_name).exists():
                    return run_name
    raise RuntimeError("All run names have been used.")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    config = load_config(args.config)
    run_name = create_run_name(config["results_dir"], config["dataset"])
    print(f"RUN NAME: {run_name}\n")
    run_dir = Path(config["results_dir"]) / run_name
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(Path(args.config).read_text())
    if config["use_wandb"]:
        wandb.init(
            project="cem-concept-discovery-sae",
            config=config,
            name=run_name,
            notes=config["description"])

    datasets = load_datasets(config)

    train_initial_models(run_dir, config, datasets)
    discover_and_match_concepts(run_dir, config, datasets)
    train_hicems(run_dir, config, datasets)
    evaluate_interventions(run_dir, config, datasets)
    run_baselines(run_dir, config, datasets)


    if config["use_wandb"]:
        wandb.save(os.path.join(run_dir, "*"))
        wandb.finish()
