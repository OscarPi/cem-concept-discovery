import torch
from torch.utils.data import DataLoader, TensorDataset
from cemcd.data import transforms
from pathlib import Path
from tqdm import tqdm
import clip

class Datasets:
    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        foundation_model=None,
        train_img_transform=None,
        val_test_img_transform=None,
        model_dir="/checkpoints",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.train_x, self.train_y, self.train_c = self.run_foundation_model(foundation_model, train_img_transform, model_dir, train_data, device)
        self.val_x, self.val_y, self.val_c = self.run_foundation_model(foundation_model, val_test_img_transform, model_dir, val_data, device)
        self.test_x, self.test_y, self.test_c = self.run_foundation_model(foundation_model, val_test_img_transform, model_dir, test_data, device)

        self.n_concepts = None
        self.n_tasks = None
        self.concept_bank = None
        self.concept_test_ground_truth = None
        self.concept_names = None

        self.foundation_model = foundation_model
        if foundation_model == "dinov2":
            self.latent_representation_size = 1536
        elif foundation_model == "clip":
            self.latent_representation_size = 768
        else:
            self.latent_representation_size = None

    def run_foundation_model(self, foundation_model, img_transform, model_dir, data, device):
        if foundation_model is None:
            transform = None
            model = None
        elif foundation_model == "dinov2":
            torch.hub.set_dir(Path(model_dir) / "dinov2")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
            model.eval()
            transform = transforms.get_default_transforms()
        elif foundation_model == "clip":
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
            for img, y, c in tqdm(data):
                img = transform(img)
                if model is not None:
                    img = img[None, ...].to(device)
                    x = model(img).detach().cpu().squeeze().float()
                else:
                    x = img
                xs.append(x)
                ys.append(y)
                cs.append(c)
        return torch.stack(xs), torch.tensor(ys), torch.stack(cs)
    
    def train_dl(self, additional_concepts=None):
        c = self.train_c
        if additional_concepts is not None:
            c = torch.concatenate((c, additional_concepts))
        return DataLoader(
            TensorDataset(self.train_x, self.train_y, c),
            batch_size=128,
            num_workers=7
        )

    def val_dl(self, additional_concepts=None):
        c = self.val_c
        if additional_concepts is not None:
            c = torch.concatenate((c, additional_concepts))
        return DataLoader(
            TensorDataset(self.val_x, self.val_y, c),
            batch_size=128,
            num_workers=7
        )
    
    def test_dl(self, additional_concepts=None):
        c = self.test_c
        if additional_concepts is not None:
            c = torch.concatenate((c, additional_concepts))
        return DataLoader(
            TensorDataset(self.test_x, self.test_y, c),
            batch_size=128,
            num_workers=7
        )
