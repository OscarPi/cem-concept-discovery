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
from cemcd.data.base import Datasets, DataGetterWrapper

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

SUPER_CONCEPT_NAMES = [
    "patterned",
    "distal_limb",
    "teeth",
    "weapons",
]

SUPER_CONCEPTS = {
    "patches": "patterned",
    "spots": "patterned",
    "stripes": "patterned",
    "flippers": "distal_limb",
    "hands": "distal_limb",
    "hooves": "distal_limb",
    "pads": "distal_limb",
    "paws": "distal_limb",
    "chewteeth": "teeth",
    "meatteeth": "teeth",
    "buckteeth": "teeth",
    "strainteeth": "teeth",
    "horns": "weapons",
    "claws": "weapons",
    "tusks": "weapons",
}

SUB_CONCEPT_NAMES = sorted(SUPER_CONCEPTS.keys())
SUB_CONCEPT_INDICES = [SELECTED_CONCEPT_SEMANTICS.index(c) for c in SUB_CONCEPT_NAMES]

SUB_CONCEPT_MAP = []
for _ in range(len(SUPER_CONCEPT_NAMES)):
    SUB_CONCEPT_MAP.append([])
for sub_concept_name, super_concept_name in SUPER_CONCEPTS.items():
    super_concept_index = SUPER_CONCEPT_NAMES.index(super_concept_name)
    SUB_CONCEPT_MAP[super_concept_index].append(SUB_CONCEPT_NAMES.index(sub_concept_name))

class AwADatasets(Datasets):
    def __init__(self, dataset_dir="/datasets", model_dir="/checkpoints"):
        super().__init__(
            n_concepts=len(SELECTED_CONCEPT_SEMANTICS) - len(SUB_CONCEPT_NAMES) + len(SUPER_CONCEPT_NAMES),
            n_tasks=N_CLASSES,
            representation_cache_dir=dataset_dir / "AwA2",
            model_dir=model_dir
        )

        dataset_dir = Path(dataset_dir)
        with (dataset_dir / "AwA2" / "train.pkl").open("rb") as f:
            self.train_data = pickle.load(f)
        with (dataset_dir / "AwA2" / "val.pkl").open("rb") as f:
            self.val_data = pickle.load(f)
        with (dataset_dir / "AwA2" / "test.pkl").open("rb") as f:
            self.test_data = pickle.load(f)

        def data_getter(data):
            def getter(idx):
                example = data[idx]
                image_path = dataset_dir / example["img_path"]
                image = Image.open(image_path).convert("RGB")
                class_label = example["class_label"]
                attr_label = example["attribute_label"]
                super_concepts = {
                    "patterned": 0,
                    "distal_limb": 0,
                    "teeth": 0,
                    "weapons": 0
                }
                non_super_concepts = []
                for i, concept_name in enumerate(SELECTED_CONCEPT_SEMANTICS):
                    if concept_name in SUPER_CONCEPTS:
                        super_concept_name = SUPER_CONCEPTS[concept_name]
                        if attr_label[i] == 1:
                            super_concepts[super_concept_name] = 1
                    else:
                        non_super_concepts.append(attr_label[i])
                concept_annotations = [
                    super_concepts["patterned"],
                    super_concepts["distal_limb"],
                    super_concepts["teeth"],
                    super_concepts["weapons"]
                ] + non_super_concepts

                return image, class_label, torch.tensor(concept_annotations, dtype=torch.float32)
            return getter

        self.data = {
            "train": DataGetterWrapper(data_getter(self.train_data), len(self.train_data)),
            "val": DataGetterWrapper(data_getter(self.val_data), len(self.val_data)),
            "test": DataGetterWrapper(data_getter(self.test_data), len(self.test_data)),
        }

        self.concept_bank = np.stack(list(map(lambda d: np.array(d["attribute_label"], dtype=bool)[SUB_CONCEPT_INDICES], self.train_data)))
        self.concept_test_ground_truth = np.stack(list(map(lambda d: np.array(d["attribute_label"])[SUB_CONCEPT_INDICES], self.test_data)))
        self.labelfree_concept_test_ground_truth = np.stack(list(map(lambda d: np.array(d["attribute_label"]), self.test_data)))

        self.concept_names = SUB_CONCEPT_NAMES

        self.sub_concept_map = SUB_CONCEPT_MAP
