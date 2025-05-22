# Adapted from https://github.com/Trustworthy-ML-Lab/Label-free-CBM

import torch
import numpy as np
import os
from tqdm import tqdm
import argparse

from label_free import cbm, utils
from label_free.data_utils import get_target_model, get_data
from label_free.train_cbm import train_cbm_and_save
from cemcd.data import cub, awa, shapes
from cemcd.metrics import calculate_concept_accuracies

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=str)
    parser.add_argument("--results-dir", required=True, type=str)
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    return parser.parse_args()

def evaluate_cbm(load_dir, dataset_dir, model_dir, dataset, device):
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().splitlines()

    if dataset == "cub":
        c_matches = {
            "A red eye": 147,
            "Glossy black wings": 20,
            "a black back": 69,
            "a black beak": 289,
            "a black breast": 116,
            "a black throat": 131,
            "a bright orange breast": 115,
            "a brownish back": 59,
            "a greenish back": 66,
            "a grey back": 63,
            "a large bill": 150,
            "a red belly": 210,
            "a red breast": 118,
            "a red throat": 133,
            "a streaked back": 238,
            "a streaked breast": 56,
            "a white belly": 209,
            "a white breast": 117,
            "a white underside": 51,
            "a yellow beak": 284,
            "a yellow crown": 299,
            "a yellow eye": 140,
            "all black coloration": 259,
            "black eyes": 145,
            "blue upperparts": 24,
            "blue wings": 9,
            "a duck-like bird": 225
        }
        concept_matches = {}
        for k, v in c_matches.items():
            if k in concepts:
                concept_matches[concepts.index(k)] = v
        cub_data = cub.CUBDatasets(dataset_dir=dataset_dir)
        concept_test_ground_truth = cub_data.concept_test_ground_truth
    elif dataset == "awa":
        concept_matches = {i: awa.SELECTED_CONCEPT_SEMANTICS.index(c).index(c) for i, c in enumerate(concepts)}
        awa_data = awa.AwADatasets(dataset_dir=dataset_dir)
        concept_test_ground_truth = awa_data.labelfree_concept_test_ground_truth
    elif dataset == "shapes":
        concept_matches = {i: shapes.CONCEPT_NAMES.index(c) for i, c in enumerate(concepts)}
        shapes_data = shapes.ShapesDatasets(dataset_dir=dataset_dir)
        concept_test_ground_truth = shapes_data.concept_test_ground_truth

    _, target_preprocess = get_target_model(device=device, model_dir=model_dir)
    model = cbm.load_cbm(load_dir, model_dir, device)

    test_d_probe = dataset + "_test"

    test_data_t = get_data(test_d_probe, dataset_dir=dataset_dir, preprocess=target_preprocess)

    accuracy = utils.get_accuracy_cbm(model, test_data_t, device)
    print("Task Accuracy: " + str(float(accuracy)))

    train_d_probe = dataset + "_train"
    train_data_t = get_data(train_d_probe, dataset_dir=dataset_dir, preprocess=target_preprocess)
    c_pred_train = np.zeros((0, len(concepts)))
    for images, _ in tqdm(train_data_t):
        with torch.no_grad():
            _, c = model(images.to(device))
            c_pred_train = np.concatenate((c_pred_train, c.cpu().detach().numpy()))


    c_pred = np.zeros((0, len(concepts)))
    for images, _ in tqdm(test_data_t):
        with torch.no_grad():
            _, c = model(images.to(device))
            c_pred = np.concatenate((c_pred, c.cpu().detach().numpy()))

    c_pred_used = np.zeros((concept_test_ground_truth.shape[0], 0))
    c_true = np.zeros((concept_test_ground_truth.shape[0], 0))
    for i, j in concept_matches.items():
        if np.all(concept_test_ground_truth[:, j:j+1] == 0) or np.all(concept_test_ground_truth[:, j:j+1] == 1):
            print("discarding " + str(i) + ":" + str(j))
            continue
        c_pred_used = np.concatenate((c_pred_used, c_pred[:, i:i+1]), axis=1)
        c_true = np.concatenate((c_true, concept_test_ground_truth[:, j:j+1]), axis=1)

    print("Concept AUCs:")
    print(calculate_concept_accuracies(c_pred_used, c_true)[2])

if __name__=='__main__':
    torch.set_float32_matmul_precision("high")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_arguments()

    load_dir = train_cbm_and_save(args.results_dir, args.dataset_dir, args.model_dir, args.dataset, device)
    evaluate_cbm(load_dir, args.dataset_dir, args.model_dir, args.dataset, device)
