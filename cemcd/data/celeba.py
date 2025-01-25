"""
Adapted from https://github.com/mateoespinosa/cem
"""
import os
import torch
import torchvision

from pytorch_lightning import seed_everything
from torchvision import transforms


from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cemcd.data import transforms
from cemcd.data.base import Datasets


#########################################################
## CONCEPT INFORMATION REGARDING CelebA
#########################################################


SELECTED_CONCEPTS = [
    2,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    39,
]

CONCEPT_SEMANTICS = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
]

##########################################################
## SIMPLIFIED LOADER FUNCTION FOR STANDARDIZATION
##########################################################

def load_celeba(dataset_dir):
    def _binarize(concepts, selected):
        binary_repr = []
        concepts = concepts[selected]
        for i in range(0, concepts.shape[-1]):
            binary_repr.append(
                str(int(np.sum(concepts[i : i + 1]) > 0)))
        return int("".join(binary_repr), 2)

    celeba_train_data = torchvision.datasets.CelebA(
        root=dataset_dir,
        split="all",
        download=True,
        target_transform=lambda x: x[0].long() - 1,
        target_type=["attr"])

    concept_freq = np.sum(
        celeba_train_data.attr.cpu().detach().numpy(),
        axis=0
    ) / celeba_train_data.attr.shape[0]
    sorted_concepts = list(map(
        lambda x: x[0],
        sorted(enumerate(np.abs(concept_freq - 0.5)), key=lambda x: x[1])))
    num_concepts = 2
    concept_idxs = sorted_concepts[:num_concepts]
    concept_idxs = sorted(concept_idxs)
    num_hidden = 6
    hidden_concepts = sorted(
        sorted_concepts[
            num_concepts:min(
                (num_concepts + num_hidden),
                len(sorted_concepts))])

    celeba_train_data = torchvision.datasets.CelebA(
        root=dataset_dir,
        split="all",
        download=True,
        target_transform=lambda x: [
            torch.tensor(
                _binarize(
                    x[1].cpu().detach().numpy(),
                    selected=(concept_idxs + hidden_concepts)),
                dtype=torch.long),
            x[1][concept_idxs].float()],
        target_type=["identity", "attr"])
    label_remap = {}
    vals = np.unique(
        list(map(
            lambda x: _binarize(
                x.cpu().detach().numpy(),
                selected=(concept_idxs + hidden_concepts)),
            celeba_train_data.attr)))
    for i, label in enumerate(vals):
        label_remap[label] = i

    celeba_train_data = torchvision.datasets.CelebA(
        root=dataset_dir,
        split="all",
        download=True,
        target_transform=lambda x: [
            torch.tensor(
                label_remap[_binarize(
                    x[1].cpu().detach().numpy(),
                    selected=(concept_idxs + hidden_concepts))],
                dtype=torch.long),
            x[1][concept_idxs].float(),
            x[1].float()
        ],
        target_type=["identity", "attr"])
    n_tasks = len(label_remap)

    # # And subsample to reduce its massive size
    # factor = 12
    # if factor != 1:
    #     train_idxs = np.random.default_rng(seed=42).choice(
    #         np.arange(0, len(celeba_train_data)),
    #         replace=False,
    #         size=len(celeba_train_data)//factor)
    #     celeba_train_data = torch.utils.data.Subset(
    #         celeba_train_data,
    #         train_idxs)
   
    total_samples = len(celeba_train_data)
    train_samples = int(0.7 * total_samples)
    test_samples = int(0.2 * total_samples)
    val_samples = total_samples - test_samples - train_samples
    generator = torch.Generator().manual_seed(42)
    celeba_train_data, celeba_test_data, celeba_val_data = torch.utils.data.random_split(
        celeba_train_data,
        [train_samples, test_samples, val_samples],
        generator=generator)

    concept_mask = np.repeat(True, len(sorted_concepts))
    concept_mask[concept_idxs] = False
    concept_bank = []
    for _, (_, _, c) in celeba_train_data:
        c = c.detach().cpu().numpy()
        concept_bank.append(c[concept_mask])
    concept_bank = np.stack(concept_bank)
    concept_bank = np.concatenate((concept_bank, np.logical_not(concept_bank)), axis=1)

    concept_test_ground_truth = []
    for _, (_, _, c) in celeba_test_data:
        c = c.detach().cpu().numpy()
        concept_test_ground_truth.append(c[concept_mask])
    concept_test_ground_truth = np.stack(concept_test_ground_truth)
    concept_test_ground_truth = np.concatenate((concept_test_ground_truth, np.logical_not(concept_test_ground_truth)), axis=1)

    concept_names = []
    for i, name in enumerate(CONCEPT_SEMANTICS):
        if i not in concept_idxs:
            concept_names.append(name)

    return {
        "train_data": celeba_train_data,
        "val_data": celeba_val_data,
        "test_data": celeba_test_data,
        "concept_bank": concept_bank,
        "concept_test_ground_truth": concept_test_ground_truth,
        "concept_names": concept_names,
        "n_concepts": num_concepts,
        "n_tasks": n_tasks
    }

class CELEBADatasets(Datasets):
    def __init__(
            self,
            foundation_model=None,
            dataset_dir="/datasets",
            cache_dir=None,
            model_dir="/checkpoints",
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        dataset = load_celeba(dataset_dir)   
        train_data = dataset["train_data"]
        val_data = dataset["val_data"]
        test_data = dataset["test_data"]

        def data_getter(data):
            def getter(idx):
                (image, [y, c, _]) = data[idx]

                return image, y, c
            getter.length = len(data)
            return getter

        super().__init__(
            train_getter=data_getter(train_data),
            val_getter=data_getter(val_data),
            test_getter=data_getter(test_data),
            foundation_model=foundation_model,
            train_img_transform=None,
            val_test_img_transform=None,
            cache_dir=cache_dir,
            model_dir=model_dir,
            device=device
        )

        self.concept_bank = dataset["concept_bank"]
        self.concept_test_ground_truth = dataset["concept_test_ground_truth"]
        self.concept_names = dataset["concept_names"]
        self.n_concepts = dataset["n_concepts"]
        self.n_tasks = dataset["n_tasks"]
