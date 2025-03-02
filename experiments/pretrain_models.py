import argparse
from pathlib import Path
import torch
from cemcd.training import train_cem
from experiment_utils import load_config, load_datasets, make_pre_concept_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", 
        type=str,
        required=True,
        help="Path to the experiment config file.")
    return parser.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    config = load_config(args.config)
    cache_dir = Path(config["cache_dir"])
#    cache_dir.mkdir()

    datasets = load_datasets(config)

    pre_concept_model = make_pre_concept_model(config)

    for dataset in datasets:
        train_cem(
            n_concepts=dataset.n_concepts,
            n_tasks=dataset.n_tasks,
            pre_concept_model=pre_concept_model if dataset.foundation_model is None else None,
            latent_representation_size=dataset.latent_representation_size or list(pre_concept_model.modules())[-1].out_features,
            train_dl=dataset.train_dl(),
            val_dl=dataset.val_dl(),
            test_dl=dataset.test_dl(),
            save_path=cache_dir / f"initial_{dataset.foundation_model or 'basic'}cem.pth",
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
