import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from cemcd.data import transforms
from pathlib import Path
from tqdm import trange
import clip
import numpy as np

class CEMDataset(Dataset):
    def __init__(self, data_getter, transform=None, additional_concepts=None):
        self.data_getter = data_getter
        self.transform = transform
        self.additional_concepts = additional_concepts

    def __len__(self):
        return self.data_getter.length

    def __getitem__(self, idx):
        image, class_label, attr_label = self.data_getter(idx)

        if self.transform:
            image = self.transform(image)

        if self.additional_concepts is not None:
            attr_label = torch.concat((attr_label, torch.from_numpy(self.additional_concepts[idx].astype(np.float32))))

        return image, class_label, attr_label

class Datasets:
    def __init__(
        self,
        train_getter,
        val_getter,
        test_getter,
        foundation_model=None,
        train_img_transform=None,
        val_test_img_transform=None,
        cache_dir=None,
        model_dir="/checkpoints",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.foundation_model = foundation_model
        self.train_getter = train_getter
        self.val_getter = val_getter
        self.test_getter = test_getter
        self.train_img_transform = train_img_transform
        self.val_test_img_transform = val_test_img_transform

        if self.foundation_model is not None:
            if cache_dir is not None and (Path(cache_dir) / f"{self.foundation_model}.npz").exists():
                cache_file = Path(cache_dir) / f"{self.foundation_model}.npz"
                print(f"Loading representations from {cache_file}.")
                with np.load(cache_file) as data:
                    self.train_x = data["train_x"]
                    self.train_y = data["train_y"]
                    self.train_c = data["train_c"]
                    self.val_x = data["val_x"]
                    self.val_y = data["val_y"]
                    self.val_c = data["val_c"]
                    self.test_x = data["test_x"]
                    self.test_y = data["test_y"]
                    self.test_c = data["test_c"]
            else:
                self.train_x, self.train_y, self.train_c = self.run_foundation_model(train_img_transform, model_dir, train_getter, device)
                self.val_x, self.val_y, self.val_c = self.run_foundation_model(val_test_img_transform, model_dir, val_getter, device)
                self.test_x, self.test_y, self.test_c = self.run_foundation_model(val_test_img_transform, model_dir, test_getter, device)
                if cache_dir is not None:
                    np.savez(
                        file=Path(cache_dir) / f"{self.foundation_model}.npz",
                        train_x=self.train_x,
                        train_y=self.train_y,
                        train_c=self.train_c,
                        val_x=self.val_x,
                        val_y=self.val_y,
                        val_c=self.val_c,
                        test_x=self.test_x,
                        test_y=self.test_y,
                        test_c=self.test_c
                    )

        self.n_concepts = None
        self.n_tasks = None
        self.concept_bank = None
        self.concept_test_ground_truth = None
        self.concept_names = None

        if foundation_model == "dinov2":
            self.latent_representation_size = 1536
        elif foundation_model == "clip":
            self.latent_representation_size = 768
        else:
            self.latent_representation_size = None

    def run_foundation_model(self, img_transform, model_dir, data_getter, device):
        if self.foundation_model == "dinov2":
            torch.hub.set_dir(Path(model_dir) / "dinov2")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
            model.eval()
            transform = transforms.get_default_transforms()
        elif self.foundation_model == "clip":
            ckpt_dir = Path(model_dir) / "clip"
            model, transform = clip.load("ViT-L/14", device=device, download_root=ckpt_dir)
            model.eval()
            model = model.encode_image
            transform.transforms[2] = transforms._convert_image_to_rgb
            transform.transforms[3] = transforms._safe_to_tensor
        else:
            raise ValueError(f"Unrecognised foundation model: {model}.")

        if img_transform is not None:
            transform = img_transform

        xs = []
        ys = []
        cs = []
        with torch.no_grad():
            for i in trange(data_getter.length):
                img, y, c = data_getter(i)
                img = transform(img)

                img = img[None, ...].to(device)
                x = model(img).detach().cpu().squeeze().float()

                xs.append(x)
                ys.append(y)
                cs.append(c)
        return torch.stack(xs), torch.tensor(ys), torch.stack(cs)
    
    def train_dl(self, additional_concepts=None):
        if self.foundation_model is not None:
            c = self.train_c
            if additional_concepts is not None:
                c = torch.concatenate((c, torch.from_numpy(additional_concepts.astype(np.float32))), axis=1)
            dataset = TensorDataset(self.train_x, self.train_y, c)
        else:
            dataset = CEMDataset(
                data_getter=self.train_getter,
                transform=self.train_img_transform,
                additional_concepts=additional_concepts
            )

        return DataLoader(
            dataset,
            batch_size=128,
            num_workers=7
        )

    def val_dl(self, additional_concepts=None):
        if self.foundation_model is not None:
            c = self.val_c
            if additional_concepts is not None:
                c = torch.concatenate((c, torch.from_numpy(additional_concepts.astype(np.float32))), axis=1)
            dataset = TensorDataset(self.val_x, self.val_y, c)
        else:
            dataset = CEMDataset(
                data_getter=self.val_getter,
                transform=self.val_test_img_transform,
                additional_concepts=additional_concepts
            )

        return DataLoader(
            dataset,
            batch_size=128,
            num_workers=7
        )
    
    def test_dl(self, additional_concepts=None):
        if self.foundation_model is not None:
            c = self.test_c
            if additional_concepts is not None:
                c = torch.concatenate((c, torch.from_numpy(additional_concepts.astype(np.float32))), axis=1)
            dataset = TensorDataset(self.test_x, self.test_y, c)
        else:
            dataset = CEMDataset(
                data_getter=self.test_getter,
                transform=self.val_test_img_transform,
                additional_concepts=additional_concepts
            )
        
        return DataLoader(
                dataset,
                batch_size=128,
                num_workers=7
        )
