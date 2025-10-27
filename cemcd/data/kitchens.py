from pathlib import Path
import numpy as np
import torch
from PIL import Image
import OpenEXR
import struct
import json
from cemcd.data.base import Datasets, DataGetterWrapper

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

class KitchensDatasets(Datasets):
    def __init__(self, dataset_dir="/datasets", model_dir="/checkpoints"):
        dataset_dir = Path(dataset_dir) / "pseudokitchens_V2"

        with (dataset_dir / "info.json").open() as f:
            dataset_info = json.load(f)

        ingredient_groups = sorted(dataset_info["ingredient_groups"].keys())
        top_level_concepts = ingredient_groups + sorted([i for i in dataset_info["object_counts"]["ingredient_counts"].keys() if i not in ingredient_groups])

        super().__init__(
            n_concepts=len(top_level_concepts),
            n_tasks=len(dataset_info["recipes"]),
            representation_cache_dir=dataset_dir,
            model_dir=model_dir
        )

        self.dataset_info = dataset_info
        self.dataset_dir = dataset_dir

        def data_getter(directory, number_of_examples):
            def getter(idx):
                image_path = directory / (str(idx + 1).zfill(len(str(number_of_examples))) + ".png")
                image = Image.open(image_path).convert("RGB")

                with (directory / (str(idx + 1).zfill(len(str(number_of_examples))) + ".json")).open() as f:
                    instance_info = json.load(f)
                    class_label = instance_info["recipe_idx"]

                with OpenEXR.File(str(directory / (str(idx + 1).zfill(len(str(number_of_examples))) + ".exr"))) as exrfile:
                    manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
                    channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
                    channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

                concept_annotations = []

                for concept_name in top_level_concepts:
                    concept_annotation = 0
                    if concept_name in dataset_info["ingredient_groups"]:
                        for ingredient in dataset_info["ingredient_groups"][concept_name]:
                            if image_contains_ingredient(channel1, channel2, manifest, dataset_info, ingredient):
                                concept_annotation = 1
                                break
                    else:
                        if image_contains_ingredient(channel1, channel2, manifest, dataset_info, concept_name):
                            concept_annotation = 1
                    concept_annotations.append(concept_annotation)

                return image, class_label, torch.tensor(concept_annotations, dtype=torch.float32)
            return getter

        self.data = {
            "train": DataGetterWrapper(data_getter(dataset_dir / "train", dataset_info["train_size"]), dataset_info["train_size"]),
            "val": DataGetterWrapper(data_getter(dataset_dir / "val", dataset_info["val_size"]), dataset_info["val_size"]),
            "test": DataGetterWrapper(data_getter(dataset_dir / "test", dataset_info["test_size"]), dataset_info["test_size"]),
        }

        self.sub_concept_names = []
        self.sub_concept_map = []
        for concept in top_level_concepts:
            sub_concepts = []
            if concept in self.dataset_info["ingredient_groups"]:
                sub_concepts = self.dataset_info["ingredient_groups"][concept]

            self.sub_concept_names.extend(sub_concepts)
            sub_concept_indices = []
            for sub_concept in sub_concepts:
                sub_concept_indices.append(self.sub_concept_names.index(sub_concept))
            self.sub_concept_map.append(sub_concept_indices)

        # self.sub_sub_concept_names = []
        # self.sub_sub_concept_map = []
        # for sub_concept in self.sub_concept_names:
        #     sub_sub_concept_indices = []
        #     if sub_concept in self.dataset_info["ingredient_groups"]:
        #         sub_sub_concepts = self.dataset_info["ingredient_groups"][sub_concept]
        #         self.sub_sub_concept_names.extend(sub_sub_concepts)
        #         for sub_sub_concept in sub_sub_concepts:
        #             sub_sub_concept_indices.append(self.sub_sub_concept_names.index(sub_sub_concept))

        #     self.sub_sub_concept_map.append(sub_sub_concept_indices)

        self.sub_concept_bank = self.create_concept_bank(self.sub_concept_names, "train")
        self.sub_concept_test_ground_truth = self.create_concept_bank(self.sub_concept_names, "test")
        # self.sub_sub_concept_bank = self.create_concept_bank(self.sub_sub_concept_names, "train")
        # self.sub_sub_concept_test_ground_truth = self.create_concept_bank(self.sub_sub_concept_names, "test")

        # self.ingredients = sorted(dataset_info["object_counts"]["ingredient_counts"].keys())
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
                elif image_contains_ingredient(channel1, channel2, manifest, self.dataset_info, concept_name):
                    concept_annotation = True
                concept_bank[concept_idx].append(concept_annotation)

        return np.stack(concept_bank, axis=1, dtype=bool)
