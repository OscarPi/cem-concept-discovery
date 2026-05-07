from pathlib import Path
import numpy as np
import torch
from PIL import Image
import OpenEXR
import struct
import json
from cemcd.data.transforms import safe_to_tensor
from cemcd.data.base import Datasets

# FRUIT_VEG = [
#     "Fruit",
#     "Vegetables"
# ]

# OTHER = [
#     "Rice",
#     "Milk",
#     "Yoghurt",
#     "Cheese",
#     "Butter",
#     "Egg",
#     "Mince",
#     "Meat",
#     "Pasta",
#     "Flour",
#     "Sugar",
#     "Oil",
#     "Spice",
#     "Garlic",
#     "Chilli",
#     "Chocolate",
#     "Syrup",
#     "Tin Tomatoes"
# ]

# TOP_LEVEL_CONCEPTS = [FRUIT_VEG] + [[item] for item in OTHER]

SUB_SUB_CONCEPTS = {
    "Apple": ["Apple 1", "Apple 2", "Apple 3", "Apple 4", "Apple 5"],
    "Potato": ["Potato 1", "Potato 2", "Potato 3", "Potato 4", "Potato 5"],
    "Pepper": ["Pepper 1", "Pepper 2", "Pepper 3"]
}

def hex_str_to_id(id_hex_string):
    packed = struct.Struct("=I").pack(int(id_hex_string, 16))
    return struct.Struct("=f").unpack(packed)[0]

def image_contains_ingredient(channel1, channel2, manifest, dataset_info, ingredient):
    for i in range(1, dataset_info["object_counts"]["ingredient_counts"][ingredient] + 1):
        if f"{ingredient} {i}" in manifest:
            id = hex_str_to_id(manifest[f"{ingredient} {i}"])
            if np.any(channel1 == id) or np.any(channel2 == id):
                return True
    return False

class KitchensDataset:
    def __init__(self, dataset_info, directory, concept_names, number_of_examples):
        self._dataset_info = dataset_info
        self._directory = directory
        self._concept_names = concept_names
        self._length = number_of_examples

    def __getitem__(self, idx):
        image_path = self._directory / (str(idx + 1).zfill(len(str(self._length))) + ".png")
        image = Image.open(image_path).convert("RGB")

        with (self._directory / (str(idx + 1).zfill(len(str(self._length))) + ".json")).open() as f:
            instance_info = json.load(f)
            class_label = instance_info["recipe_idx"]

        with OpenEXR.File(str(self._directory / (str(idx + 1).zfill(len(str(self._length))) + ".exr"))) as exrfile:
            manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
            channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
            channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

        concept_annotations = []

        for concept_name in self._concept_names:
            concept_annotation = 0
            if concept_name in self._dataset_info["ingredient_groups"]:
                for ingredient in self._dataset_info["ingredient_groups"][concept_name]:
                    if image_contains_ingredient(channel1, channel2, manifest, self._dataset_info, ingredient):
                        concept_annotation = 1
                        break
            else:
                if image_contains_ingredient(channel1, channel2, manifest, self._dataset_info, concept_name):
                    concept_annotation = 1
            concept_annotations.append(concept_annotation)

        return image, class_label, torch.tensor(concept_annotations, dtype=torch.float32)


    def __len__(self):
        return self._length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class KitchensDatasets(Datasets):
    def __init__(self, dataset_dir="/datasets", model_dir="/checkpoints"):
        dataset_dir = Path(dataset_dir) / "pseudokitchens_V2"

        with (dataset_dir / "info.json").open() as f:
            dataset_info = json.load(f)

        ingredient_groups = sorted(dataset_info["ingredient_groups"].keys())
        grouped_ingredients = [ing for group in dataset_info["ingredient_groups"].values() for ing in group]
        top_level_concepts = ingredient_groups + sorted([i for i in dataset_info["object_counts"]["ingredient_counts"].keys() if i not in grouped_ingredients])

        super().__init__(
            n_concepts=len(top_level_concepts),
            n_tasks=len(dataset_info["recipes"]),
            representation_cache_dir=dataset_dir,
            model_dir=model_dir
        )

        self.dataset_info = dataset_info
        self.dataset_dir = dataset_dir

        self.data = {
            "train": KitchensDataset(dataset_info, dataset_dir / "train", top_level_concepts, dataset_info["train_size"]),
            "val": KitchensDataset(dataset_info, dataset_dir / "val", top_level_concepts, dataset_info["val_size"]),
            "test": KitchensDataset(dataset_info, dataset_dir / "test", top_level_concepts, dataset_info["test_size"]),
        }

        self.concept_names = top_level_concepts

        all_sub_concept_and_sub_sub_concept_names = []
        for concept in top_level_concepts:
            if concept in self.dataset_info["ingredient_groups"]:
                for sub_concept_name in self.dataset_info["ingredient_groups"][concept]:
                    all_sub_concept_and_sub_sub_concept_names.append(sub_concept_name)
                    if sub_concept_name in SUB_SUB_CONCEPTS:
                        for sub_sub_concept_name in SUB_SUB_CONCEPTS[sub_concept_name]:
                            all_sub_concept_and_sub_sub_concept_names.append(sub_sub_concept_name)
        all_sub_concept_and_sub_sub_concept_train_labels = self.create_concept_bank(all_sub_concept_and_sub_sub_concept_names, "train")
        all_sub_concept_and_sub_sub_concept_test_labels = self.create_concept_bank(all_sub_concept_and_sub_sub_concept_names, "test")

        self.concept_bank = []
        for concept in top_level_concepts:
            sub_concepts = []
            if concept in self.dataset_info["ingredient_groups"]:
                for sub_concept_name in self.dataset_info["ingredient_groups"][concept]:
                    sub_sub_concepts = []
                    if sub_concept_name in SUB_SUB_CONCEPTS:
                        for sub_sub_concept_name in SUB_SUB_CONCEPTS[sub_concept_name]:
                            i = all_sub_concept_and_sub_sub_concept_names.index(sub_sub_concept_name)
                            sub_sub_concepts.append({
                                "name": sub_sub_concept_name,
                                "train_labels": all_sub_concept_and_sub_sub_concept_train_labels[:, i],
                                "test_labels": all_sub_concept_and_sub_sub_concept_test_labels[:, i],
                            })
                    i = all_sub_concept_and_sub_sub_concept_names.index(sub_concept_name)
                    sub_concepts.append({
                        "name": sub_concept_name,
                        "train_labels": all_sub_concept_and_sub_sub_concept_train_labels[:, i],
                        "test_labels": all_sub_concept_and_sub_sub_concept_test_labels[:, i],
                        "sub_sub_concepts": sub_sub_concepts
                    })

            self.concept_bank.append(sub_concepts)

        self.ingredients = sorted(self.dataset_info["object_counts"]["ingredient_counts"].keys())
        # labelfree_test_concepts = []
        # for i in range(len(self.ingredients)):
        #     labelfree_test_concepts.append([])
        # for i in range(dataset_info["test_size"]):
        #     with OpenEXR.File(str(self.dataset_dir / "test" / (str(i + 1).zfill(len(str(dataset_info["test_size"]))) + ".exr"))) as exrfile:
        #         manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
        #         channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
        #         channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

        #     for concept_idx, concept_name in enumerate(self.ingredients):
        #         concept_annotation = 0
        #         if image_contains_ingredient(channel1, channel2, manifest, dataset_info, concept_name):
        #             concept_annotation = 1
        #         labelfree_test_concepts[concept_idx].append(concept_annotation)
        # self.labelfree_concept_test_ground_truth = np.stack(labelfree_test_concepts, axis=1)

    def create_concept_bank(self, concept_names, split):
        concept_bank = []
        for _ in range(len(concept_names)):
            concept_bank.append([])

        for i in range(self.dataset_info[f"{split}_size"]):
            with OpenEXR.File(str(self.dataset_dir / split / (str(i + 1).zfill(len(str(self.dataset_info[f"{split}_size"]))) + ".exr"))) as exrfile:
                manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
                channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
                channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

            for concept_idx, concept_name in enumerate(concept_names):
                concept_annotation = False
                if concept_name in self.dataset_info["ingredient_groups"]:
                    for ingredient in self.dataset_info["ingredient_groups"][concept_name]:
                        if image_contains_ingredient(channel1, channel2, manifest, self.dataset_info, ingredient):
                            concept_annotation = True
                            break
                elif concept_name in self.dataset_info["object_counts"]["ingredient_counts"].keys():
                    if image_contains_ingredient(channel1, channel2, manifest, self.dataset_info, concept_name):
                        concept_annotation = True
                else:
                    if concept_name in manifest:
                        id = hex_str_to_id(manifest[concept_name])
                        if np.any(channel1 == id) or np.any(channel2 == id):
                            concept_annotation = True

                concept_bank[concept_idx].append(concept_annotation)

        return np.stack(concept_bank, axis=1, dtype=bool)
