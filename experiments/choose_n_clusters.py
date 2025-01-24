import argparse
from pathlib import Path
import yaml
from tqdm import tqdm, trange
import sklearn.metrics
import numpy as np
import torch
from cemcd.concept_discovery import calculate_embeddings
from experiment_utils import load_config, load_datasets, get_initial_models
from cemcd import turtle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", 
        type=str,
        required=True,
        help="Path to the experiment config file.")
    parser.add_argument(
        "--ns-to-test",
        nargs="+",
        type=int,
        required=True,
        help="Numbers of clusters to calculate silhouette scores for."
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="File to store results in."
    )
    return parser.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()

    config = load_config(args.config)
    datasets = load_datasets(config)

    models, _ = get_initial_models(config, datasets, run_dir=None)

    predictions = []
    embeddings = []
    for dataset, model in zip(datasets, models):
        c_pred, c_embs, _ = calculate_embeddings(model, dataset.train_dl())
        predictions.append(c_pred)
        embeddings.append(c_embs)
    predictions = np.stack(predictions, axis=0)

    results = {}
    for n_clusters in tqdm(args.ns_to_test):
        score = 0
        for concept_idx in trange(models[0].n_concepts, leave=False):
            for concept_on in (True, False):
                if concept_on:
                    sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] > 0.5, axis=0)
                else:
                    sample_filter = np.logical_and.reduce(predictions[:, :, concept_idx] < 0.5, axis=0)

                Zs = []
                for e in embeddings:
                    Zs.append(e[:, concept_idx][sample_filter])

                cluster_labels, _ = turtle.run_turtle(
                    Zs=Zs, k=n_clusters, warm_start=config["warm_start"], epochs=1000)
                for Z in Zs:
                    score += sklearn.metrics.silhouette_score(Z, cluster_labels)
        print(f"n_clusters={n_clusters}, silhouette_score={score}")
        results[n_clusters] = float(score)
    
    if args.results_file is not None:
        with args.results_file.open("w") as f:
            yaml.safe_dump(results, f)
