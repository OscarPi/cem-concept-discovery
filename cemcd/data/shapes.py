from pathlib import Path
import pickle
from PIL import Image, ImageDraw
import numpy as np
import torch
from cemcd.data.base import Datasets

SHAPES = ["square", "circle", "triangle", "hexagon"]
COLOURS = ["red", "green", "blue", "purple"]

CONCEPT_NAMES = [
    "Shape is a square", # 0
    "Shape is a circle", # 1
    "Shape is a triangle", # 2
    "Shape is a hexagon", # 3
    "Shape is red", # 4
    "Shape is green", # 5
    "Shape is blue", # 6
    "Shape is purple", # 7
    "Background is red", # 8
    "Background is green", # 9
    "Background is blue", # 10
    "Background is purple" # 11
]

def render_image(example):
    im = Image.new(mode="RGB", size=(256, 256), color=example["background_colour"])
    draw = ImageDraw.Draw(im)

    if example["shape"] == "square":
        draw.regular_polygon(
            bounding_circle=((128, 128), 100),
            n_sides=4,
            fill=example["shape_colour"],
            outline=example["shape_colour"])
    elif example["shape"] == "circle":
        draw.circle(xy=(128, 128), radius=100, fill=example["shape_colour"], outline=example["shape_colour"])
    elif example["shape"] == "triangle":
        draw.regular_polygon(
            bounding_circle=((128, 128), 100),
            n_sides=3,
            fill=example["shape_colour"],
            outline=example["shape_colour"])
    elif example["shape"] == "hexagon":
        draw.regular_polygon(
            bounding_circle=((128, 128), 100),
            n_sides=6,
            fill=example["shape_colour"],
            outline=example["shape_colour"])
    else:
        raise ValueError(f"Unexpected shape: {example['shape']}")

    for xy, rotation in zip(example["obstructions_centres"], example["obstructions_rotations"]):
        draw.regular_polygon(
            bounding_circle=(xy, example["obstructions_radius"]),
            n_sides=example["obstructions_n_sides"],
            rotation=rotation,
            fill="black",
            outline="black")

    return im

def calculate_label(example):
    label = 0
    label += SHAPES.index(example["shape"])
    label += 4 * COLOURS.index(example["shape_colour"])
    label += 16 * [c for c in COLOURS if c != example["shape_colour"]].index(example["background_colour"])
    return label

def calculate_concepts(example):
    concept_labels = [
        example["shape"] in ["square", "triangle", "hexagon"],
        example["shape_colour"] in ["red", "green"],
        example["shape_colour"] in ["blue", "purple"],
        example["background_colour"] in ["red", "green"],
        example["background_colour"] in ["blue", "purple"],
    ]
    return torch.tensor(concept_labels, dtype=torch.float32)

def make_concept_bank(examples):
    concept_bank = np.zeros((len(examples), 12))

    for idx, example in enumerate(examples):
        if example["shape"] == "square":
            concept_bank[idx, 0] = 1
        if example["shape"] == "circle":
            concept_bank[idx, 1] = 1
        if example["shape"] == "triangle":
            concept_bank[idx, 2] = 1
        if example["shape"] == "hexagon":
            concept_bank[idx, 3] = 1
        if example["shape_colour"] == "red":
            concept_bank[idx, 4] = 1
        if example["shape_colour"] == "green":
            concept_bank[idx, 5] = 1
        if example["shape_colour"] == "blue":
            concept_bank[idx, 6] = 1
        if example["shape_colour"] == "purple":
            concept_bank[idx, 7] = 1
        if example["background_colour"] == "red":
            concept_bank[idx, 8] = 1
        if example["background_colour"] == "green":
            concept_bank[idx, 9] = 1
        if example["background_colour"] == "blue":
            concept_bank[idx, 10] = 1
        if example["background_colour"] == "purple":
            concept_bank[idx, 11] = 1
    
    return concept_bank

class ShapesDatasets(Datasets):
    def __init__(
            self,
            foundation_model=None,
            dataset_dir="/datasets",
            model_dir="/checkpoints",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        with (Path(dataset_dir) / "shapes" / "shapes_dataset.pkl").open("rb") as f:
            dataset = pickle.load(f)

        def data_getter(examples):
            def getter(idx):
                ex = examples[idx]
                return render_image(ex), calculate_label(ex), calculate_concepts(ex)

            getter.length = len(examples)
            return getter

        super().__init__(
            train_getter=data_getter(dataset["train"]),
            val_getter=data_getter(dataset["val"]),
            test_getter=data_getter(dataset["test"]),
            foundation_model=foundation_model,
            train_img_transform=None,
            val_test_img_transform=None,
            representation_cache_dir=Path(dataset_dir) / "shapes",
            model_dir=model_dir,
            device=device)
        
        self.concept_bank = make_concept_bank(dataset["train"])

        self.concept_test_ground_truth = make_concept_bank(dataset["test"])

        self.sub_concept_map = [
            [0, 2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11]
        ]
    
        self.concept_names = CONCEPT_NAMES

        self.n_concepts = 5
        self.n_tasks = 48
