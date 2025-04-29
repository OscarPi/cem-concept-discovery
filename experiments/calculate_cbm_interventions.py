import argparse
from pathlib import Path
import torch
from cemcd.training import load_cbm
from experiment_utils import get_intervention_accuracies
from cemcd.data import cub, mnist, shapes

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", 
        type=Path,
        required=True,
        help="Path to the model.")
    parser.add_argument(
        "-d", "--dataset", 
        type=str,
        required=True,
        help="Dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    if args.dataset == "mnist":
        dataset = mnist.MNISTDatasets(
            n_digits=2,
            max_digit=6,
            foundation_model="dinov2",
            dataset_dir="/homes/ogh22/datasets",
            model_dir="/homes/ogh22/checkpoints")
    elif args.dataset == "shapes":
        dataset = shapes.ShapesDatasets(
            foundation_model="dinov2",
            dataset_dir="/homes/ogh22/datasets",
            model_dir="/homes/ogh22/checkpoints")
    elif args.dataset == "cub":
        dataset = cub.CUBDatasets(
            foundation_model="dinov2",
            dataset_dir="/homes/ogh22/datasets",
            model_dir="/homes/ogh22/checkpoints")
    else:
        raise ValueError("Unknown dataset")

    cbm, _ = load_cbm(
        n_concepts=dataset.n_concepts,
        n_tasks=dataset.n_tasks,
        latent_representation_size=dataset.latent_representation_size,
        train_dl=dataset.train_dl(),
        test_dl=dataset.test_dl(),
        path=args.model,
        use_task_class_weights=False,
        use_concept_loss_weights=False
    )

    results = get_intervention_accuracies(
        model=cbm,
        test_dl=dataset.test_dl(),
        concepts_to_intervene=range(dataset.n_concepts),
        one_at_a_time=False
    )

    print(f"{dataset.foundation_model}_cbm_concept_interventions_cumulative")
    print(",".join(map(str, results)))
