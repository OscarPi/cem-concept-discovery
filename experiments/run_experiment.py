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
import time
import torch
from torchvision.models import resnet34
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = os.environ.get("RESULTS_DIR", 'results/')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', choices=['mnist_add', 'dsprites', 'cub', 'awa'],
                        help='Name of the experiment: "mnist_add", "dsprites", "cub", or "awa"')
    parser.add_argument('--resume', type=int, help="Resume the given run")
    parser.add_argument("--max-concepts-to-discover", type=int, help="The maximum number of concepts to discover.", default=10)
    parser.add_argument('--reuse-model', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()

def get_results_directory(experiment_name, resume):
    experiment_dir = os.path.join(RESULTS_DIR, experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    if resume is not None:
        run_num = resume
    else:
        run_dirs = [d for d in os.listdir(experiment_dir)]
        run_num = len(run_dirs) + 1

    run_dir = os.path.join(experiment_dir, f'run_{run_num}')
    if not resume:
        os.makedirs(run_dir)

    return run_dir

def run_experiment(
    resume,
    n_concepts,
    n_tasks,
    train_dl_getter,
    val_dl_getter,
    test_dl_getter,
    concept_bank,
    concept_test_ground_truth,
    concept_names,
    run_dir,
    pre_concept_model,
    concept_model,
    random_state,
    chi=True,
    max_concepts_to_discover=10,
    max_epochs=300,
    reuse=False,):

    model, model_0, n_discovered_concepts, discovered_concept_test_ground_truth = cemcd.concept_discovery.discover_multiple_concepts(
        resume=resume,
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        pre_concept_model=pre_concept_model,
        save_path=run_dir,
        train_dl_getter=train_dl_getter,
        val_dl_getter=val_dl_getter,
        test_dl_getter=test_dl_getter,
        concept_bank=concept_bank,
        concept_test_ground_truth=concept_test_ground_truth,
        concept_names=concept_names,
        max_concepts_to_discover=max_concepts_to_discover,
        random_state=random_state,
        chi=chi,
        max_epochs=max_epochs,
        reuse=reuse)

    trainer = lightning.Trainer()

    concept_intervention_accuracies = []
    model.intervention_mask = torch.tensor([0] * model.n_concepts)
    [test_results] = trainer.test(model, test_dl_getter(discovered_concept_test_ground_truth))
    task_accuracy = round(test_results["test_y_accuracy"], 4)
    concept_intervention_accuracies.append(task_accuracy)
    for i in range(model.n_concepts):
        model.intervention_mask[i] = 1
        [test_results] = trainer.test(model, test_dl_getter(discovered_concept_test_ground_truth))
        task_accuracy = round(test_results["test_y_accuracy"], 4)
        concept_intervention_accuracies.append(task_accuracy)

    discovered_concept_intervention_accuracies = []
    model.intervention_mask = torch.tensor([0] * model.n_concepts)
    [test_results] = trainer.test(model, test_dl_getter(discovered_concept_test_ground_truth))
    task_accuracy = round(test_results["test_y_accuracy"], 4)
    discovered_concept_intervention_accuracies.append(task_accuracy)
    for i in range(n_discovered_concepts):
        model.intervention_mask[i + n_concepts] = 1
        [test_results] = trainer.test(model, test_dl_getter(discovered_concept_test_ground_truth))
        task_accuracy = round(test_results["test_y_accuracy"], 4)
        discovered_concept_intervention_accuracies.append(task_accuracy)

    cem_intervention_accuracies = []
    for i in range(n_concepts + 1):
        model_0.intervention_mask = torch.tensor([1] * i + [0] * (model_0.n_concepts - i))
        [test_results] = trainer.test(model_0, test_dl_getter(None))
        task_accuracy = round(test_results["test_y_accuracy"], 4)
        cem_intervention_accuracies.append(task_accuracy)

    if os.path.exists(os.path.join(run_dir, "cbm_baseline.pth")):
        Path(os.path.join(run_dir, "cbm_baseline.pth")).unlink()
    cbm, cbm_test_results = train_cbm(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        concept_model=concept_model,
        train_dl=train_dl_getter(None),
        val_dl=val_dl_getter(None),
        test_dl=test_dl_getter(None),
        black_box=False,
        save_path=os.path.join(run_dir, "cbm_baseline.pth"),
        max_epochs=max_epochs)
    cbm_task_accuracy = round(cbm_test_results["test_y_accuracy"], 4)
    cbm_concept_auc = round(cbm_test_results["test_c_auc"], 4)

    cbm_intervention_accuracies = []
    for i in range(n_concepts + 1):
        cbm.intervention_mask = torch.tensor([1] * i + [0] * (cbm.n_concepts - i))
        [test_results] = trainer.test(cbm, test_dl_getter(None))
        task_accuracy = round(test_results["test_y_accuracy"], 4)
        cbm_intervention_accuracies.append(task_accuracy)

    if os.path.exists(os.path.join(run_dir, "black_box_baseline.pth")):
        Path(os.path.join(run_dir, "black_box_baseline.pth")).unlink()
    # CBM with concept loss weight of 0 is a black box
    _, black_box_test_results = train_cbm(
        n_concepts=list(pre_concept_model().modules())[-1].out_features,
        n_tasks=n_tasks,
        concept_model=pre_concept_model,
        train_dl=train_dl_getter(None),
        val_dl=val_dl_getter(None),
        test_dl=test_dl_getter(None),
        black_box=True,
        save_path=os.path.join(run_dir, "black_box_baseline.pth"),
        max_epochs=max_epochs
    )
    black_box_task_accuracy = round(black_box_test_results["test_y_accuracy"], 4)

    with open(os.path.join(run_dir, "concept_intervention_results.txt"), 'w') as f:
        f.write(f"All concept intervention accuracies: {', '.join([str(x) for x in concept_intervention_accuracies])}\n")
        f.write(f"Discovered concept intervention accuracies: {', '.join([str(x) for x in discovered_concept_intervention_accuracies])}\n")
        f.write(f"CEM intervention accuracies: {', '.join([str(x) for x in cem_intervention_accuracies])}\n")
        f.write(f"CBM intervention accuracies: {', '.join([str(x) for x in cbm_intervention_accuracies])}\n")

    with open(os.path.join(run_dir, "baseline_results.txt"), 'w') as f:
        f.write(f"CBM task accuracy: {cbm_task_accuracy}\n")
        f.write(f"CBM concept AUC: {cbm_concept_auc}\n")
        f.write(f"Black box task accuracy: {black_box_task_accuracy}\n")


def run_mnist(run_dir, random_state, resume, reuse, max_concepts_to_discover):
    datasets = mnist.MNISTDatasets(2, selected_digits=(0, 1, 2, 3, 4, 5, 6))

    concept_bank = np.stack((
        datasets.train_labels[:, 0] == 0,
        datasets.train_labels[:, 0] == 1,
        datasets.train_labels[:, 0] == 2,
        datasets.train_labels[:, 0] == 3,
        datasets.train_labels[:, 0] == 4,
        datasets.train_labels[:, 0] == 5,
        datasets.train_labels[:, 0] == 6,
        datasets.train_labels[:, 1] == 0,
        datasets.train_labels[:, 1] == 1,
        datasets.train_labels[:, 1] == 2,
        datasets.train_labels[:, 1] == 3,
        datasets.train_labels[:, 1] == 4,
        datasets.train_labels[:, 1] == 5,
        datasets.train_labels[:, 1] == 6
    ), axis=1)
    concept_test_ground_truth = np.stack((
        datasets.test_labels[:, 0] == 0,
        datasets.test_labels[:, 0] == 1,
        datasets.test_labels[:, 0] == 2,
        datasets.test_labels[:, 0] == 3,
        datasets.test_labels[:, 0] == 4,
        datasets.test_labels[:, 0] == 5,
        datasets.test_labels[:, 0] == 6,
        datasets.test_labels[:, 1] == 0,
        datasets.test_labels[:, 1] == 1,
        datasets.test_labels[:, 1] == 2,
        datasets.test_labels[:, 1] == 3,
        datasets.test_labels[:, 1] == 4,
        datasets.test_labels[:, 1] == 5,
        datasets.test_labels[:, 1] == 6
    ), axis=1)
    concept_names = [
        "First digit 0",
        "First digit 1",
        "First digit 2",
        "First digit 3",
        "First digit 4",
        "First digit 5",
        "First digit 6",
        "Second digit 0",
        "Second digit 1",
        "Second digit 2",
        "Second digit 3",
        "Second digit 4",
        "Second digit 5",
        "Second digit 6",
    ]

    run_experiment(
        resume=resume,
        n_concepts=2,
        n_tasks=13,
        train_dl_getter=lambda additional_concepts: datasets.train_dl(concept_generator=lambda labels: labels > 3, additional_concepts=additional_concepts),
        val_dl_getter=lambda additional_concepts: datasets.val_dl(concept_generator=lambda labels: labels > 3, additional_concepts=additional_concepts),
        test_dl_getter=lambda additional_concepts: datasets.test_dl(concept_generator=lambda labels: labels > 3, additional_concepts=additional_concepts),
        concept_bank=concept_bank,
        concept_test_ground_truth=concept_test_ground_truth,
        concept_names=concept_names,
        run_dir=run_dir,
        pre_concept_model=lambda: get_pre_concept_model(28, 28, 2),
        concept_model=lambda: get_pre_concept_model(28, 28, 2, 2),
        random_state=random_state,
        max_concepts_to_discover=max_concepts_to_discover,
        reuse=reuse
    )

def run_dsprites(run_dir, random_state, resume, reuse, max_concepts_to_discover):
    datasets = dsprites.DSpritesDatasets()

    concept_bank = np.stack((
        datasets.scale_train == 0,
        datasets.scale_train == 1,
        datasets.scale_train == 2,
        datasets.scale_train == 3,
        datasets.scale_train == 4,
        datasets.scale_train == 5,
        datasets.shape_train == 0,
        datasets.shape_train == 1,
        datasets.shape_train == 2,
        datasets.quadrant_train == 0,
        datasets.quadrant_train == 1,
        datasets.quadrant_train == 2,
        datasets.quadrant_train == 3
    ), axis=1)
    concept_test_ground_truth = np.stack((
        datasets.scale_test == 0,
        datasets.scale_test == 1,
        datasets.scale_test == 2,
        datasets.scale_test == 3,
        datasets.scale_test == 4,
        datasets.scale_test == 5,
        datasets.shape_test == 0,
        datasets.shape_test == 1,
        datasets.shape_test == 2,
        datasets.quadrant_test == 0,
        datasets.quadrant_test == 1,
        datasets.quadrant_test == 2,
        datasets.quadrant_test == 3
    ), axis=1)
    concept_names = [
        "Scale 0",
        "Scale 1",
        "Scale 2",
        "Scale 3",
        "Scale 4",
        "Scale 5",
        "Shape 0",
        "Shape 1",
        "Shape 2",
        "Quadrant 0",
        "Quadrant 1",
        "Quadrant 2",
        "Quadrant 3"
    ]

    run_experiment(
        resume=resume,
        n_concepts=3,
        n_tasks=11,
        train_dl_getter=lambda additional_concepts: datasets.train_dl(additional_concepts=additional_concepts),
        val_dl_getter=lambda additional_concepts: datasets.val_dl(additional_concepts=additional_concepts),
        test_dl_getter=lambda additional_concepts: datasets.test_dl(additional_concepts=additional_concepts),
        concept_bank=concept_bank,
        concept_test_ground_truth=concept_test_ground_truth,
        concept_names=concept_names,
        run_dir=run_dir,
        pre_concept_model=lambda: get_pre_concept_model(64, 64, 1),
        concept_model=lambda: get_pre_concept_model(64, 64, 1, 3),
        random_state=random_state,
        max_concepts_to_discover=max_concepts_to_discover,
        reuse=reuse
    )

CUB_COMPRESSED_CONCEPT_SEMANTICS = ['has_wing_color::light', 'has_wing_color::dark', 'has_upperparts_color::light', 'has_upperparts_color::dark', 'has_underparts_color::light', 'has_underparts_color::dark', 'has_back_color::light', 'has_back_color::dark', 'has_upper_tail_color::light', 'has_upper_tail_color::dark', 'has_breast_color::light', 'has_breast_color::dark', 'has_throat_color::light', 'has_throat_color::dark', 'has_forehead_color::light', 'has_forehead_color::dark', 'has_under_tail_color::light', 'has_under_tail_color::dark', 'has_nape_color::light', 'has_nape_color::dark', 'has_belly_color::light', 'has_belly_color::dark', 'has_primary_color::light', 'has_primary_color::dark', 'has_leg_color::light', 'has_leg_color::dark', 'has_bill_color::light', 'has_bill_color::dark', 'has_crown_color::light', 'has_crown_color::dark']

CUB_GROUPS = [
    "has_wing_color",
    "has_upperparts_color",
    "has_underparts_color",
    "has_back_color",
    "has_upper_tail_color",
    "has_breast_color",
    "has_throat_color",
    "has_forehead_color",
    "has_under_tail_color",
    "has_nape_color",
    "has_belly_color",
    "has_primary_color",
    "has_leg_color",
    "has_bill_color",
    "has_crown_color"]

def compress_colour_concepts(concepts):
    light = defaultdict(bool)
    dark = defaultdict(bool)

    light_colours = ["white", "yellow", "blue", "buff"]
    dark_colours = ["grey", "black", "brown"]

    compressed_concepts = []
    for idx, concept in enumerate(cub.SELECTED_CONCEPT_SEMANTICS):
        group = concept[:concept.find("::")]
        if group in CUB_GROUPS:
            colour = concept[concept.find("::")+2:]
            if concepts[idx] == 1:
                if colour in light_colours:
                    light[group] = True
                elif colour in dark_colours:
                    dark[group] = True
                else:
                    print(colour)
                    raise RuntimeError("unrecognised colour")

    for group in CUB_GROUPS:
        if light[group]:
            compressed_concepts.append(1)
        else:
            compressed_concepts.append(0)
        if dark[group]:
            compressed_concepts.append(1)
        else:
            compressed_concepts.append(0)

    return compressed_concepts

CUB_SELECTED_CLASSES = [140, 139, 38, 187, 167, 147, 25, 16, 119, 101, 184, 186, 90, 159, 17, 142, 154, 80, 131, 100]

def run_cub(run_dir, random_state, resume, reuse, max_concepts_to_discover):
    cub_datasets = cub.CUBDatasets(selected_classes=CUB_SELECTED_CLASSES)

    concept_bank = np.array(list(map(lambda d: d["attribute_label"], cub_datasets.train_data)))
    concept_test_ground_truth = np.array(list(map(lambda d: d["attribute_label"], cub_datasets.test_data)))
    concept_names = cub.SELECTED_CONCEPT_SEMANTICS

    run_experiment(
        resume=resume,
        n_concepts=len(CUB_COMPRESSED_CONCEPT_SEMANTICS),
        n_tasks=20,
        train_dl_getter=lambda additional_concepts: cub_datasets.train_dl(compress_colour_concepts, additional_concepts=additional_concepts),
        val_dl_getter=lambda additional_concepts: cub_datasets.val_dl(compress_colour_concepts, additional_concepts=additional_concepts),
        test_dl_getter=lambda additional_concepts: cub_datasets.test_dl(compress_colour_concepts, additional_concepts=additional_concepts),
        concept_bank=concept_bank,
        concept_test_ground_truth=concept_test_ground_truth,
        concept_names=concept_names,
        run_dir=run_dir,
        pre_concept_model=lambda: resnet34(pretrained=True),
        concept_model=lambda: torch.nn.Sequential(resnet34(pretrained=True), torch.nn.Linear(1000, len(CUB_COMPRESSED_CONCEPT_SEMANTICS))),
        random_state=random_state, 
        max_concepts_to_discover=max_concepts_to_discover,
        max_epochs=150,
        reuse=reuse
    )

def run_awa(run_dir, random_state, resume, reuse, max_concepts_to_discover):
    awa_datasets = awa.AwADatasets(selected_classes=[41, 5, 45, 6, 12, 28, 1, 48, 43, 30])

    concept_bank = np.array(list(map(lambda d: d["attribute_label"], awa_datasets.train_data)))
    concept_test_ground_truth = np.array(list(map(lambda d: d["attribute_label"], awa_datasets.test_data)))
    concept_names = awa.SELECTED_CONCEPT_SEMANTICS

    run_experiment(
        resume=resume,
        n_concepts=5,
        n_tasks=10,
        train_dl_getter=lambda additional_concepts: awa_datasets.train_dl(lambda c: c[:5], additional_concepts=additional_concepts),
        val_dl_getter=lambda additional_concepts: awa_datasets.val_dl(lambda c: c[:5], additional_concepts=additional_concepts),
        test_dl_getter=lambda additional_concepts: awa_datasets.test_dl(lambda c: c[:5], additional_concepts=additional_concepts),
        concept_bank=concept_bank,
        concept_test_ground_truth=concept_test_ground_truth,
        concept_names=concept_names,
        run_dir=run_dir,
        pre_concept_model=lambda: resnet34(pretrained=True),
        concept_model=lambda: torch.nn.Sequential(resnet34(pretrained=True), torch.nn.Linear(1000, 5)),
        random_state=random_state,
        chi=False,
        max_concepts_to_discover=max_concepts_to_discover,
        max_epochs=150,
        reuse=reuse
    )

if __name__ == "__main__":
    start_time = time.time()
    torch.set_float32_matmul_precision("high")
    args = parse_arguments()
    experiment_name = args.experiment_name
    run_dir = get_results_directory(experiment_name, args.resume)
    resume = True
    if args.resume is None:
        resume = False
        random_state = np.random.randint(0, 1000)
        with open(os.path.join(run_dir, "stats.txt"), "w") as f:
            f.write(f"random_state: {str(random_state)}\n")
            f.write(f"reuse_model: {str(args.reuse_model)}\n")
            f.write(f"max_concepts_to_discover: {str(args.max_concepts_to_discover)}\n")
    else:
        with open(os.path.join(run_dir, "random_state.txt"), "r") as f:
            random_state = int(f.read())

    if experiment_name == "mnist_add":
        run_mnist(run_dir, random_state, resume, args.reuse_model, args.max_concepts_to_discover)
    elif experiment_name == "dsprites":
        run_dsprites(run_dir, random_state, resume, args.reuse_model, args.max_concepts_to_discover)
    elif experiment_name == "cub":
        run_cub(run_dir, random_state, resume, args.reuse_model, args.max_concepts_to_discover)
    elif experiment_name == "awa":
        run_awa(run_dir, random_state, resume, args.reuse_model, args.max_concepts_to_discover)

    with open(os.path.join(run_dir, "stats.txt"), "a") as f:
        f.write(f"time: {time.time() - start_time}s\n")
