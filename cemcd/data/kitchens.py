from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import OpenEXR
import struct
import json
from cemcd.data import transforms
from cemcd.data.base import Datasets, CEMDataset

# dry_ingredients = [
#     "Flour",
#     "Macaroni",
#     "Rice",
#     "Spaghetti",
#     "Spice",
#     "Sugar",
# ]

# wet_ingredients = [
#     "Apple",
#     "Banana",
#     "Butter",
#     "Carrot",
#     "Cheese",
#     "Chilli",
#     "Courgette",
#     "Egg",
#     "Meat",
#     "Milk",
#     "Mince",
#     "Oil",
#     "Onion",
#     "Orange",
#     "Pear",
#     "Pepper",
#     "Pineapple",
#     "Potato",
#     "Garlic",
#     "Syrup",
#     "Tin Tomatoes",
#     "Yoghurt",
#     "Chocolate",
# ]

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
    def __init__(
            self,
            foundation_model=None,
            dataset_dir="/datasets",
            model_dir="/checkpoints",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        dataset_dir = Path(dataset_dir) / "kitchens"

        with (dataset_dir / "info.json").open() as f:
            dataset_info = json.load(f)

        # concept_names = []
        # grouped_ingredients = []
        # for group, ingredients in dataset_info["ingredient_groups"].items():
        #     concept_names.append(group)
        #     grouped_ingredients.extend(ingredients)
        # for ingredient in dataset_info["object_counts"]["ingredient_counts"].keys():
        #     if ingredient not in grouped_ingredients:
        #         concept_names.append(ingredient)

        self.ingredients = sorted(dataset_info["object_counts"]["ingredient_counts"].keys())

        def data_getter(directory, number_of_examples):
            def getter(idx):
                image_path = directory / (str(idx + 1).zfill(len(str(number_of_examples))) + ".png")
                image = Image.open(image_path).convert("RGB")

                with (directory / (str(idx + 1).zfill(len(str(number_of_examples))) + ".json")).open() as f:
                    instance_info = json.load(f)
                    if "recipe_idx" in instance_info:
                        class_label = instance_info["recipe_idx"]
                    else:
                        class_label = 0
                with OpenEXR.File(str(directory / (str(idx + 1).zfill(len(str(number_of_examples))) + ".exr"))) as exrfile:
                    manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
                    channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
                    channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

                # concept_annotations = []

                # for concept_name in concept_names:
                #     concept_annotation = 0
                #     if concept_name in dataset_info["ingredient_groups"]:
                #         for ingredient in dataset_info["ingredient_groups"][concept_name]:
                #             if image_contains_ingredient(channel1, channel2, manifest, dataset_info, ingredient):
                #                 concept_annotation = 1
                #                 break
                #     else:
                #         if image_contains_ingredient(channel1, channel2, manifest, dataset_info, concept_name):
                #             concept_annotation = 1
                #     concept_annotations.append(concept_annotation)

                concept_annotations = []

                for ingredient in self.ingredients:
                    concept_annotation = 0
                    if image_contains_ingredient(channel1, channel2, manifest, dataset_info, ingredient):
                        concept_annotation = 1
                    concept_annotations.append(concept_annotation)

                return image, class_label, torch.tensor(concept_annotations, dtype=torch.float32)
            getter.length = number_of_examples
            return getter

        train_img_transform = None
        val_test_img_transform = None
        if foundation_model is None:
            train_img_transform = transforms.resnet_train
            val_test_img_transform = transforms.resnet_val_test

        super().__init__(
            train_getter=data_getter(dataset_dir / "train", dataset_info["train_size"]),
            val_getter=data_getter(dataset_dir / "val", dataset_info["val_size"]),
            test_getter=data_getter(dataset_dir / "test", dataset_info["test_size"]),
            foundation_model=foundation_model,
            train_img_transform=train_img_transform,
            val_test_img_transform=val_test_img_transform,
            representation_cache_dir=dataset_dir,
            model_dir=model_dir,
            device=device
        )

        if self.foundation_model is not None:
            cache_file = Path(dataset_dir) / f"{self.foundation_model}_concept_test.pt"
            if cache_file.exists():
                data = torch.load(cache_file, weights_only=True)
                self.concept_test_x = data["concept_test_x"]
                self.concept_test_y = data["concept_test_y"]
                self.concept_test_c = data["concept_test_c"]
            else:
                self.concept_test_x, self.concept_test_y, self.concept_test_c = self.run_foundation_model(val_test_img_transform, model_dir, data_getter(dataset_dir / "concept_test", dataset_info["concept_test_size"]), device)
                data = {
                    "concept_test_x": self.concept_test_x,
                    "concept_test_y": self.concept_test_y,
                    "concept_test_c": self.concept_test_c
                }
                torch.save(data, cache_file)


        # self.labelfree_concept_test_ground_truth = np.stack(list(map(lambda d: np.array(d["attribute_label"]), test_data)))

        #self.n_concepts = len(concept_names)
        self.n_concepts = len(self.ingredients)
        self.n_tasks = len(dataset_info["recipes"])

        # concept_bank_concept_names = []
        # self.sub_concept_map = []
        # for concept_name in concept_names:
        #     sub_concepts = []
        #     if concept_name in dataset_info["ingredient_groups"]:
        #         for ingredient in dataset_info["ingredient_groups"][concept_name]:
        #             sub_concepts.append(len(concept_bank_concept_names))
        #             concept_bank_concept_names.append(ingredient)
        #     self.sub_concept_map.append(sub_concepts)

        # self.concept_names = concept_bank_concept_names

        # train_concepts = []
        # for i in range(len(self.concept_names)):
        #     train_concepts.append([])
        # for i in range(dataset_info["train_size"]):
        #     with OpenEXR.File(str(dataset_dir / "train" / (str(i + 1).zfill(len(str(dataset_info["train_size"]))) + ".exr"))) as exrfile:
        #         manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
        #         channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
        #         channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

        #     for concept_idx, concept_name in enumerate(self.concept_names):
        #         concept_annotation = 0
        #         if image_contains_ingredient(channel1, channel2, manifest, dataset_info, concept_name):
        #             concept_annotation = 1
        #         train_concepts[concept_idx].append(concept_annotation)
        # self.concept_bank = np.stack(train_concepts, axis=1)

        # test_concepts = []
        # for i in range(len(self.concept_names)):
        #     test_concepts.append([])
        # for i in range(dataset_info["test_size"]):
        #     with OpenEXR.File(str(dataset_dir / "test" / (str(i + 1).zfill(len(str(dataset_info["test_size"]))) + ".exr"))) as exrfile:
        #         manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
        #         channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
        #         channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

        #     for concept_idx, concept_name in enumerate(self.concept_names):
        #         concept_annotation = 0
        #         if image_contains_ingredient(channel1, channel2, manifest, dataset_info, concept_name):
        #             concept_annotation = 1
        #         test_concepts[concept_idx].append(concept_annotation)
        # self.concept_test_ground_truth = np.stack(test_concepts, axis=1)

    def concept_test_dl(self):
        if self.foundation_model is not None:
            dataset = TensorDataset(self.concept_test_x, self.concept_test_y, self.concept_test_c)
        else:
            raise NotImplementedError()

        return DataLoader(
                dataset,
                batch_size=256,
                num_workers=7)
