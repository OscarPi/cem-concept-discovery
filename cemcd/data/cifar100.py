from pathlib import Path
from tqdm import trange
import sklearn.model_selection
import numpy as np
import torch
import torchvision
import clip
from cemcd.data.base import Datasets
from cemcd.data import transforms

CLASS_NAMES = [
    "aquarium fish",
    "apple",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "computer keyboard",
    "lamp",
    "lawn-mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak tree",
    "orange",
    "orchid",
    "otter",
    "palm tree",
    "pear",
    "pickup truck",
    "pine tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow tree",
    "wolf",
    "woman",
    "worm"
]

SUPER_CLASS_NAMES = [
    "aquatic mammals",
    "fish",
    "flowers",
    "food containers",
    "fruit and vegetables",
    "household electrical devices",
    "household furniture",
    "insects",
    "large carnivores",
    "large man-made outdoor things",
    "large natural outdoor scenes",
    "large omnivores and herbivores",
    "medium-sized mammals",
    "non-insect invertebrates",
    "people",
    "reptiles",
    "small mammals",
    "trees",
    "vehicles 1",
    "vehicles 2"
]

SUPER_SUPER_CLASS_NAMES = [
    "animals",
    "man-made",
    "natural"
]

SUPER_CLASSES = {
    "beaver": "aquatic mammals",
    "dolphin": "aquatic mammals",
    "otter": "aquatic mammals",
    "seal": "aquatic mammals",
    "whale": "aquatic mammals",
    "aquarium fish": "fish",
    "flatfish": "fish",
    "ray": "fish",
    "shark": "fish",
    "trout": "fish",
    "orchid": "flowers",
    "poppy": "flowers",
    "rose": "flowers",
    "sunflower": "flowers",
    "tulip": "flowers",
    "bottle": "food containers",
    "bowl": "food containers",
    "can": "food containers",
    "cup": "food containers",
    "plate": "food containers",
    "apple": "fruit and vegetables",
    "mushroom": "fruit and vegetables",
    "orange": "fruit and vegetables",
    "pear": "fruit and vegetables",
    "sweet pepper": "fruit and vegetables",
    "clock": "household electrical devices",
    "computer keyboard": "household electrical devices",
    "lamp": "household electrical devices",
    "telephone": "household electrical devices",
    "television": "household electrical devices",
    "bed": "household furniture",
    "chair": "household furniture",
    "couch": "household furniture",
    "table": "household furniture",
    "wardrobe": "household furniture",
    "bee": "insects",
    "beetle": "insects",
    "butterfly": "insects",
    "caterpillar": "insects",
    "cockroach": "insects",
    "bear": "large carnivores",
    "leopard": "large carnivores",
    "lion": "large carnivores",
    "tiger": "large carnivores",
    "wolf": "large carnivores",
    "bridge": "large man-made outdoor things",
    "castle": "large man-made outdoor things",
    "house": "large man-made outdoor things",
    "road": "large man-made outdoor things",
    "skyscraper": "large man-made outdoor things",
    "cloud": "large natural outdoor scenes",
    "forest": "large natural outdoor scenes",
    "mountain": "large natural outdoor scenes",
    "plain": "large natural outdoor scenes",
    "sea": "large natural outdoor scenes",
    "camel": "large omnivores and herbivores",
    "cattle": "large omnivores and herbivores",
    "chimpanzee": "large omnivores and herbivores",
    "elephant": "large omnivores and herbivores",
    "kangaroo": "large omnivores and herbivores",
    "fox": "medium-sized mammals",
    "porcupine": "medium-sized mammals",
    "possum": "medium-sized mammals",
    "raccoon": "medium-sized mammals",
    "skunk": "medium-sized mammals",
    "crab": "non-insect invertebrates",
    "lobster": "non-insect invertebrates",
    "snail": "non-insect invertebrates",
    "spider": "non-insect invertebrates",
    "worm": "non-insect invertebrates",
    "baby": "people",
    "boy": "people",
    "girl": "people",
    "man": "people",
    "woman": "people",
    "crocodile": "reptiles",
    "dinosaur": "reptiles",
    "lizard": "reptiles",
    "snake": "reptiles",
    "turtle": "reptiles",
    "hamster": "small mammals",
    "mouse": "small mammals",
    "rabbit": "small mammals",
    "shrew": "small mammals",
    "squirrel": "small mammals",
    "maple tree": "trees",
    "oak tree": "trees",
    "palm tree": "trees",
    "pine tree": "trees",
    "willow tree": "trees",
    "bicycle": "vehicles 1",
    "bus": "vehicles 1",
    "motorcycle": "vehicles 1",
    "pickup truck": "vehicles 1",
    "train": "vehicles 1",
    "lawn-mower": "vehicles 2",
    "rocket": "vehicles 2",
    "streetcar": "vehicles 2",
    "tank": "vehicles 2",
    "tractor": "vehicles 2"
}

SUPER_SUPER_CLASSES = {
    "aquatic mammals": "animals",
    "fish": "animals",
    "flowers": "natural",
    "food containers": "man-made",
    "fruit and vegetables": "natural",
    "household electrical devices": "man-made",
    "household furniture": "man-made",
    "insects": "animals",
    "large carnivores": "animals",
    "large man-made outdoor things": "man-made",
    "large natural outdoor scenes": "natural",
    "large omnivores and herbivores": "animals",
    "medium-sized mammals": "animals",
    "non-insect invertebrates": "animals",
    "people": "animals",
    "reptiles": "animals",
    "small mammals": "animals",
    "trees": "natural",
    "vehicles 1": "man-made",
    "vehicles 2": "man-made"
}

SUB_CONCEPT_MAP = [list(range(len(SUPER_CLASS_NAMES)))]
# for super_class, super_class_name in enumerate(SUPER_CLASS_NAMES):
#     super_super_class_name = SUPER_SUPER_CLASSES[super_class_name]
#     super_super_class = SUPER_SUPER_CLASS_NAMES.index(super_super_class_name)
#     SUB_CONCEPT_MAP[super_super_class].append(super_class)

class CIFARDatasets(Datasets):
    def __init__(
            self,
            foundation_model=None,
            dataset_dir="/datasets",
            model_dir="/checkpoints",
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        train_val_data = torchvision.datasets.CIFAR100(dataset_dir, train=True, download=True)
        self.train_data, self.val_data = sklearn.model_selection.train_test_split(
            train_val_data, test_size=0.2, random_state=42
        )
        self.test_data = torchvision.datasets.CIFAR100(dataset_dir, train=False, download=True)

        def data_getter(data):
            def getter(idx):
                img = data[idx][0]
                label = data[idx][1]
                #class_name = CLASS_NAMES[label]
                #super_class_name = SUPER_CLASSES[class_name]
                #super_super_class_name = SUPER_SUPER_CLASSES[super_class_name]
                #super_super_class = SUPER_SUPER_CLASS_NAMES.index(super_super_class_name)
                concept_labels = [1]
                #concept_labels[super_super_class] = 1
                return img, label, torch.tensor(concept_labels, dtype=torch.float32)
            getter.length = len(data)
            return getter

        train_img_transform = None
        val_test_img_transform = None
        if foundation_model is None:
            train_img_transform = transforms.resnet_train
            val_test_img_transform = transforms.resnet_val_test

        super().__init__(
            train_getter=data_getter(self.train_data),
            val_getter=data_getter(self.val_data),
            test_getter=data_getter(self.test_data),
            foundation_model=foundation_model,
            train_img_transform=train_img_transform,
            val_test_img_transform=val_test_img_transform,
            dataset_dir=Path(dataset_dir) / "cifar-100-python",
            model_dir=model_dir,
            device=device
        )

        train_concepts = np.zeros((len(self.train_data), len(SUPER_CLASS_NAMES)))
        for idx, (_, label) in enumerate(self.train_data):
            class_name = CLASS_NAMES[label]
            super_class_name = SUPER_CLASSES[class_name]
            super_class = SUPER_CLASS_NAMES.index(super_class_name)
            train_concepts[idx, super_class] = 1
        self.concept_bank = np.concatenate((train_concepts, np.logical_not(train_concepts)), axis=1)

        test_concepts = np.zeros((len(self.test_data), len(SUPER_CLASS_NAMES)))
        for idx, (_, label) in enumerate(self.test_data):
            class_name = CLASS_NAMES[label]
            super_class_name = SUPER_CLASSES[class_name]
            super_class = SUPER_CLASS_NAMES.index(super_class_name)
            test_concepts[idx, super_class] = 1
        self.concept_test_ground_truth = np.concatenate((test_concepts, np.logical_not(test_concepts)), axis=1)

        self.concept_names = SUPER_CLASS_NAMES + list(map(lambda s: "NOT " + s, SUPER_CLASS_NAMES))

        self.n_concepts = 1
        self.n_tasks = 100

        self.sub_concept_map = SUB_CONCEPT_MAP
