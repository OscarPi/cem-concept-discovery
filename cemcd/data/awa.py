"""
As with other files in the repository, based on code from https://github.com/mateoespinosa/cem
Which was adapted from: https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/cub_loader.py
"""
from pathlib import Path
import pickle
import numpy as np
import torch
from PIL import Image
from cemcd.data import transforms
from cemcd.data.base import Datasets

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 50

#########################################################
## CONCEPT INFORMATION REGARDING AwA2
#########################################################

# AwA2 Class names

CLASS_NAMES = [
    "antelope",
    "grizzly+bear",
    "killer+whale",
    "beaver",
    "dalmatian",
    "persian+cat",
    "horse",
    "german+shepherd",
    "blue+whale",
    "siamese+cat",
    "skunk",
    "mole",
    "tiger",
    "hippopotamus",
    "leopard",
    "moose",
    "spider+monkey",
    "humpback+whale",
    "elephant",
    "gorilla",
    "ox",
    "fox",
    "sheep",
    "seal",
    "chimpanzee",
    "hamster",
    "squirrel",
    "rhinoceros",
    "rabbit",
    "bat",
    "giraffe",
    "wolf",
    "chihuahua",
    "rat",
    "weasel",
    "otter",
    "buffalo",
    "zebra",
    "giant+panda",
    "deer",
    "bobcat",
    "pig",
    "lion",
    "mouse",
    "polar+bear",
    "collie",
    "walrus",
    "raccoon",
    "cow",
    "dolphin",
]

# Names of all selected AwA attributes
SELECTED_CONCEPT_SEMANTICS = [
    "black",
    "white",
    "blue",
    "brown",
    "gray",
    "orange",
    "red",
    "yellow",
    "patches",
    "spots",
    "stripes",
    "furry",
    "hairless",
    "toughskin",
    "big",
    "small",
    "bulbous",
    "lean",
    "flippers",
    "hands",
    "hooves",
    "pads",
    "paws",
    "longleg",
    "longneck",
    "tail",
    "chewteeth",
    "meatteeth",
    "buckteeth",
    "strainteeth",
    "horns",
    "claws",
    "tusks",
    "muscle",
    "bipedal",
    "quadrapedal"
]

class AwADatasets(Datasets):
    def __init__(
            self,
            foundation_model=None,
            dataset_dir="/datasets",
            model_dir="/checkpoints",
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        dataset_dir = Path(dataset_dir)
        with (dataset_dir / "AwA2" / "train.pkl").open("rb") as f:
            train_data = pickle.load(f)
        with (dataset_dir / "AwA2" / "val.pkl").open("rb") as f:
            val_data = pickle.load(f)
        with (dataset_dir / "AwA2" / "test.pkl").open("rb") as f:
            test_data = pickle.load(f)

        def data_getter(data):
            def getter(idx):
                example = data[idx]
                image_path = dataset_dir / example["img_path"]
                image = Image.open(image_path).convert("RGB")
                class_label = example["class_label"]
                attr_label = example["attribute_label"]
                attr_label = attr_label[:5]

                return image, class_label, torch.tensor(attr_label, dtype=torch.float32)
            getter.length = len(data)
            return getter

        train_img_transform = None
        val_test_img_transform = None
        if foundation_model is None:
            train_img_transform = transforms.resnet_train
            val_test_img_transform = transforms.resnet_val_test

        super().__init__(
            train_getter=data_getter(train_data),
            val_getter=data_getter(val_data),
            test_getter=data_getter(test_data),
            foundation_model=foundation_model,
            train_img_transform=train_img_transform,
            val_test_img_transform=val_test_img_transform,
            dataset_dir=dataset_dir / "AwA2",
            model_dir=model_dir,
            device=device
        )

        train_concepts = np.array(list(map(lambda d: d["attribute_label"], train_data)))
        self.concept_bank = np.concatenate((train_concepts, np.logical_not(train_concepts)), axis=1)
        test_concepts = np.array(list(map(lambda d: d["attribute_label"], test_data)))
        self.concept_test_ground_truth = np.concatenate((test_concepts, np.logical_not(test_concepts)), axis=1)
        self.concept_names = SELECTED_CONCEPT_SEMANTICS + list(map(lambda s: "NOT " + s, SELECTED_CONCEPT_SEMANTICS))

        self.n_concepts = 5
        self.n_tasks = N_CLASSES
