from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import clip
from cemcd.data import transforms

class CEMDataset(Dataset):
    def __init__(self, data, transform=None, use_concepts=True, additional_concepts=None, include_provided_concepts=True):
        self.data = data
        self.transform = transform
        self.use_concepts = use_concepts
        self.additional_concepts = additional_concepts
        self.include_provided_concepts = include_provided_concepts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, c = self.data[idx]

        if self.transform:
            x = self.transform(x)

        if not self.include_provided_concepts:
            c = torch.tensor([], dtype=torch.float32)

        if self.additional_concepts is not None:
            c = torch.concat((c, torch.from_numpy(self.additional_concepts[idx].astype(np.float32))))

        if self.use_concepts:
            return x, y, c
        else:
            return x, y

class Datasets:
    def __init__(
            self,
            n_concepts,
            n_tasks,
            representation_cache_dir=None,
            model_dir="/checkpoints"):
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self._representation_cache_dir = Path(representation_cache_dir)
        self._model_dir = Path(model_dir)
        self._representation_cache = {}
        self._label_and_concept_cache = {}

    def _run_foundation_model(self, foundation_model, split, custom_transform=None):
        if custom_transform is None and foundation_model in self._representation_cache:
            if split in self._representation_cache[foundation_model]:
                return self._representation_cache[foundation_model][split]

        if custom_transform is None:
            cache_file = self._representation_cache_dir / foundation_model / f"{split}.pt"
            if cache_file.exists(): 
                print(f"Loading representations from {cache_file}.")
                if foundation_model not in self._representation_cache:
                    self._representation_cache[foundation_model] = {}
                self._representation_cache[foundation_model][split] = torch.load(cache_file, weights_only=True)
                return self._representation_cache[foundation_model][split]

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        if foundation_model == "dinov2_vitg14":
            torch.hub.set_dir(self._model_dir / "dinov2")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
            model.eval()
            transform = transforms.dino_transforms
        elif foundation_model == "clip_vitl14":
            ckpt_dir = self._model_dir / "clip"
            model, transform = clip.load("ViT-L/14", device=device, download_root=ckpt_dir)
            model.eval()
            model = model.encode_image
            transform.transforms[2] = transforms._convert_image_to_rgb
            transform.transforms[3] = transforms._safe_to_tensor
        else:
            raise ValueError(f"Unrecognised foundation model: {self.foundation_model}.")

        if custom_transform is not None:
            transform = custom_transform

        data = self.data[split]
        xs = []
        with torch.no_grad():
            for img, _, _ in tqdm(data):
                img = transform(img)

                img = img[torch.newaxis, ...].to(device)
                x = model(img).detach().cpu().squeeze().float()

                xs.append(x)
        xs = torch.stack(xs)

        if custom_transform is None:
            cache_dir = self._representation_cache_dir / foundation_model
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"{split}.pt"
            torch.save(xs, cache_file)
            if foundation_model not in self._representation_cache:
                self._representation_cache[foundation_model] = {}
            self._representation_cache[foundation_model][split] = xs

        return xs

    def get_labels_and_concepts(self, split):
        if split in self._label_and_concept_cache:
            return self._label_and_concept_cache[split]

        ys = []
        cs = []
        for _, y, c in tqdm(self.data[split]):
            ys.append(y)
            cs.append(c)

        self._label_and_concept_cache[split] = (torch.tensor(ys), torch.stack(cs))
        return self._label_and_concept_cache[split]

    def get_dataloader(self, split, foundation_model=None, transform=None, use_concepts=True, additional_concepts=None, include_provided_concepts=True):
        if foundation_model is None:
            dataset = CEMDataset(
                data=self.data[split],
                transform=transform,
                use_concepts=use_concepts,
                additional_concepts=additional_concepts,
                include_provided_concepts=include_provided_concepts)
        else:
            x = self._run_foundation_model(foundation_model, split, transform)
            y, c = self.get_labels_and_concepts(split)

            if not include_provided_concepts:
                c = torch.empty(size=(c.shape[0], 0), dtype=torch.float32)
            if additional_concepts is not None:
                c = torch.concatenate((c, torch.from_numpy(additional_concepts.astype(np.float32))), axis=1)
            if use_concepts:
                dataset = TensorDataset(x, y, c)
            else:
                dataset = TensorDataset(x, y)

        return DataLoader(
            dataset,
            batch_size=256,
            num_workers=7)

class DataGetterWrapper:
    def __init__(self, getter, length):
        self.getter = getter
        self.length = length

    def __getitem__(self, index):
        return self.getter(index)

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
