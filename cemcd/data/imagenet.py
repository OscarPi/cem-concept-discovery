from pathlib import Path
from nltk.corpus import wordnet as wn
import pickle
import torch
from torch.utils.data import DataLoader
from cemcd.data.base import Datasets, DataGetterWrapper, CEMDataset

CONCEPT_SYNSETS = [
    "n13086908", # Plant part
    "n04194289", # Ship
    "n01861778", # Mammal
    "n09287968", # Geological formation
    "n02778669", # Ball
    "n03051540", # Clothing
    "n02316707", # Echinoderm
    "n04285146", # Sports equipment
    "n03614007", # Keyboard
    "n03743902", # Memorial
    "n03736970", # Mechanical device
    "n04576211", # Wheeled vehicle
    "n02159955", # Insect
    "n12992868", # Fungus
    "n02913152", # Building
    "n04451818", # Tool
    "n02898711", # Bridge
    "n01976957", # Crab
    "n01503061", # Bird
    "n03800933", # Musical instrument
    "n01661091", # Reptile
    "n04524313", # Vehicle
    "n04202417", # Shop
    "n07555863", # Food
    "n03094503", # Container
    "n02512053", # Fish
    "n02796623", # Barrier
    "n03739693", # Medical instrument
    "n04341686", # Structure
    "n03682487", # Lock
    "n04447443", # Toiletry
    "n03278248", # Electronic equipment
    "n03122748", # Covering
    "n03926148", # Photographic equipment
    "n03405265", # Furnishing
    "n03269401", # Electrical device
    "n03699975", # Machine
    "n01905661", # Invertebrate
    "n02924116", # Bus
    "n01627424", # Amphibian
    "n04263760", # Source of illumination
    "n04147495", # Scientific instrument
    "n00007846", # Person
    "n03183080", # Device
    "n00021265", # Food
    "n04516672", # Utensil
    "n03563967", # Implement
    "n03528263", # Home appliance
    "n03309808", # Fabric
    "n03294048", # Equipment
    "n03078287", # Communication system
    "n00017222", # Plant
    "n04019101", # Public transport
    "n03964744", # Plaything
    "n03100490", # Conveyance
]

GROUP_SIZE = 50000  # number of examples in each file when loading precomputed features

def imagenet_id_to_synset(imagenet_id):
    """
    Convert an ImageNet-style ID to an NLTK Synset.
    """
    pos = imagenet_id[0]
    offset = int(imagenet_id[1:])
    return wn.synset_from_pos_and_offset(pos, offset)

def all_hypernyms(synset):
    """
    Return the full set of hypernyms (direct and indirect) of a WordNet synset.
    """
    visited = set()

    def recurse(s):
        for h in s.hypernyms():
            if h not in visited:
                visited.add(h)
                recurse(h)

    recurse(synset)
    visited.add(synset)
    return visited

# def load_precomputed_features_for_split(split_name, split_size, foundation_model, dir):
#     if foundation_model == "dinov2_vitg14":
#         representation_size = 1536
#     elif foundation_model == "clip_vitl14":
#         representation_size = 768

#     n_groups = (split_size + GROUP_SIZE - 1) // GROUP_SIZE

#     x = torch.zeros((split_size, representation_size), dtype=torch.float32)

#     for g in range(n_groups):
#         t = torch.load(dir / f"{split_name}_{foundation_model}_features_group_{g+1}.pt")
#         start = g * GROUP_SIZE
#         end = min(start + GROUP_SIZE, split_size)
#         x[start:end] = t

#     return x

def calculate_concepts(synset_ids):
    synset_id_to_concepts = {}
    for synset_id in synset_ids:
        synset = imagenet_id_to_synset(synset_id)
        hypernyms = all_hypernyms(synset)
        concepts = torch.zeros(len(CONCEPT_SYNSETS))
        for idx, concept_synset_id in enumerate(CONCEPT_SYNSETS):
            concept_synset = imagenet_id_to_synset(concept_synset_id)
            if concept_synset in hypernyms:
                concepts[idx] = 1.0

        synset_id_to_concepts[synset_id] = concepts

    return synset_id_to_concepts

class ImageNetDatasets(Datasets):
    def __init__(
            self,
            dataset_dir="/datasets"):
        dataset_dir = Path(dataset_dir)
        splits_file = dataset_dir / "imagenet" / "splits.pkl"

        if not splits_file.exists():
            raise FileNotFoundError(f"Cannot find splits file at {splits_file}. Run the splitting script first.")

        with splits_file.open("rb") as f:
            splits = pickle.load(f)

        train_synset_ids = sorted(set(item["synset_id"] for item in splits["train"]))
        synset_id_to_idx = {s: i for i, s in enumerate(train_synset_ids)}

        synset_id_to_concepts = calculate_concepts(train_synset_ids)

        # if foundation_model is not None:
        #     train_x = load_precomputed_features_for_split("train", len(splits["train"]), foundation_model, dataset_dir / "imagenet")
        #     val_x = load_precomputed_features_for_split("val", len(splits["val"]), foundation_model, dataset_dir / "imagenet")
        #     test_x = load_precomputed_features_for_split("test", len(splits["test"]), foundation_model, dataset_dir / "imagenet")
        # else:
        foundation_model = "clip_vitl14"

        super().__init__(
            n_concepts=len(CONCEPT_SYNSETS),
            n_tasks=len(train_synset_ids),
        )

        def data_getter(data, split_name):
            currently_loaded_chunk = None
            currently_loaded_chunk_idx = None
            def getter(idx):
                nonlocal currently_loaded_chunk, currently_loaded_chunk_idx

                synset_id = data[idx]["synset_id"]

                chunk_idx = idx // GROUP_SIZE
                within_chunk_idx = idx % GROUP_SIZE

                if chunk_idx != currently_loaded_chunk_idx:
                    file_path = dataset_dir / "imagenet" / f"{split_name}_{foundation_model}_features_group_{chunk_idx+1}.pt"
                    currently_loaded_chunk = torch.load(file_path, weights_only=True)
                    currently_loaded_chunk_idx = chunk_idx

                return currently_loaded_chunk[within_chunk_idx], synset_id_to_idx[synset_id], synset_id_to_concepts[synset_id]
            return getter
        
        self.data = {
            "train": DataGetterWrapper(data_getter(splits["train"], "train"), len(splits["train"])),
            "val": DataGetterWrapper(data_getter(splits["val"], "val"), len(splits["val"])),
            "test": DataGetterWrapper(data_getter(splits["test"], "test"), len(splits["test"])),
        }

        self.concept_names = []
        for synset_id in CONCEPT_SYNSETS:
            synset = imagenet_id_to_synset(synset_id)
            concept_name = synset.name().split(".")[0].replace('_',' ')
            self.concept_names.append(concept_name)


    def get_dataloader(self, split, foundation_model=None, transform=None, use_concepts=True, additional_concepts=None, include_provided_concepts=True):
        assert foundation_model == "clip_vitl14", "Only 'clip_vitl14' is supported for ImageNet."

        dataset = CEMDataset(
            data=self.data[split],
            transform=transform,
            use_concepts=use_concepts,
            additional_concepts=additional_concepts,
            include_provided_concepts=include_provided_concepts)

        return DataLoader(
            dataset,
            batch_size=256,
            num_workers=7)
