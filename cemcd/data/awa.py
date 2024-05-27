"""
As with other files in the repository, based on code from https://github.com/mateoespinosa/cem
Which was adapted from: https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/cub_loader.py
"""
import os
import torch
import pickle
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 50


# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY AwA2 EXPERIMENTS!!
#                The dataset can be downloaded from here: https://cvml.ista.ac.at/AwA2/
# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR
DATASET_DIR = os.environ.get("DATASET_DIR", 'datasets/Animals_with_Attributes2/')

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

class AwADataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the AwA dataset
    """

    def __init__(self, data, transform=None, concept_transform=None, additional_concepts=None):
        self.data = data
        self.transform = transform
        self.concept_transform = concept_transform
        self.additional_concepts = additional_concepts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx]
        image_path = image_data["img_path"]
        image = Image.open(image_path).convert("RGB")

        class_label = image_data["class_label"]
        if self.transform:
            image = self.transform(image)

        attr_label = image_data["attribute_label"]
        if self.concept_transform is not None:
            attr_label = self.concept_transform(attr_label)

        if self.additional_concepts is not None:
            for i in self.additional_concepts[idx]:
                attr_label.append(i)
        
        return image, class_label, torch.FloatTensor(attr_label)

class AwADatasets:
    def __init__(self, root_dir=DATASET_DIR, selected_classes=None):
        with open(os.path.join(root_dir, "train.pickle"), "rb") as f:
            train_data = pickle.load(f)
        with open(os.path.join(root_dir, "val.pickle"), "rb") as f:
            val_data = pickle.load(f)
        with open(os.path.join(root_dir, "test.pickle"), "rb") as f:
            test_data = pickle.load(f)

        self.selected_classes = selected_classes
        if selected_classes is not None:
            self.train_data = self.filter_data(train_data, selected_classes)
            self.val_data = self.filter_data(val_data, selected_classes)
            self.test_data = self.filter_data(test_data, selected_classes)
        else:
            self.train_data = train_data
            self.val_data = val_data
            self.test_data =  test_data

    def filter_data(self, data, selected_classes):
        filtered_data = []
        for sample in data:
            if sample["class_label"] in selected_classes:
                sample["class_label"] = selected_classes.index(sample["class_label"])
                filtered_data.append(sample)
        return filtered_data

    def train_dl(self, concept_transform=None, additional_concepts=None):
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
        return DataLoader(
            AwADataset(self.train_data, transform, concept_transform, additional_concepts=additional_concepts),
            batch_size=128,
            num_workers=7
        )

    def val_dl(self, concept_transform=None, additional_concepts=None):
        transform = transforms.Compose([
            transforms.CenterCrop(299),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
        return DataLoader(
            AwADataset(self.val_data, transform, concept_transform, additional_concepts=additional_concepts),
            batch_size=128,
            num_workers=7
        )

    def test_dl(self, concept_transform=None, additional_concepts=None):
        transform = transforms.Compose([
            transforms.CenterCrop(299),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
        return DataLoader(
            AwADataset(self.test_data, transform, concept_transform, additional_concepts=additional_concepts),
            batch_size=128,
            num_workers=7
        )
