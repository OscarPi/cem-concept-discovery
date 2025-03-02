from pathlib import Path
from tqdm import trange
import sklearn.model_selection
import numpy as np
import torch
import torchvision
import clip
from cemcd.data.base import Datasets
from cemcd.data import transforms

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
def load_mnist(dataset_dir):
    global x_train, y_train, x_val, y_val, x_test, y_test
    if len(x_train) > 0:
        return

    train_val_ds = torchvision.datasets.MNIST(dataset_dir, train=True, download=True)
    x_train_val = []
    y_train_val = []
    for x, y in train_val_ds:
        x_train_val.append(x)
        y_train_val.append(y)
    x_train_val = np.stack(x_train_val, axis=0)
    y_train_val = np.stack(y_train_val, axis=0)

    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=42
    )

    test_ds = torchvision.datasets.MNIST(dataset_dir, train=False, download=True)
    for x, y in test_ds:
        x_test.append(x)
        y_test.append(y)
    x_test = np.stack(x_test, axis=0)
    y_test = np.stack(y_test, axis=0)

def create_addition_set(x, y, n_digits, selected_digits, dataset_size):
    x = x[np.isin(y, selected_digits)]
    y = y[np.isin(y, selected_digits)]
    samples = []
    labels = []
    for _ in range(dataset_size):
        digits = []
        digit_labels = []
        for _ in range(n_digits):
            idx = np.random.choice(x.shape[0])
            digits.append(np.expand_dims(x[idx], axis=2))
            digit_labels.append(y[idx])
        samples.append(np.concatenate(digits, axis=2))
        labels.append(np.array(digit_labels))
    return np.array(samples), np.array(labels)

class MNISTDatasets(Datasets):
    def __init__(
            self,
            n_digits,
            max_digit,
            foundation_model=None,
            dataset_dir="/datasets",
            model_dir="/checkpoints",
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        selected_digits = tuple(range(max_digit + 1))
        load_mnist(dataset_dir)

        np.random.seed(42)
        self.train_samples, self.train_labels = create_addition_set(x_train, y_train, n_digits, selected_digits, 10000)
        self.val_samples, self.val_labels = create_addition_set(x_val, y_val, n_digits, selected_digits, int(10000 * 0.2))
        self.test_samples, self.test_labels = create_addition_set(x_test, y_test, n_digits, selected_digits, 10000)

        def data_getter(samples, labels):
            def getter(idx):
                return (
                    torchvision.transforms.ToTensor()(samples[idx]),
                    np.sum(labels[idx]),
                    torch.tensor(labels[idx] > (max_digit / 2), dtype=torch.float32))
            getter.length = len(samples)
            return getter

        representation_cache_dir = Path(dataset_dir) / "MNIST" / f"{n_digits}-{max_digit}"
        representation_cache_dir.mkdir(exist_ok=True)
        super().__init__(
            train_getter=data_getter(self.train_samples, self.train_labels),
            val_getter=data_getter(self.val_samples, self.val_labels),
            test_getter=data_getter(self.test_samples, self.test_labels),
            foundation_model=foundation_model,
            train_img_transform=None,
            val_test_img_transform=None,
            dataset_dir=representation_cache_dir,
            model_dir=model_dir,
            device=device
        )

        train_concepts = []
        test_concepts = []
        concept_names = []
        for i in range(n_digits):
            for j in selected_digits:
                train_concepts.append(self.train_labels[:, i] == j)
                test_concepts.append(self.test_labels[:, i] == j)
                concept_names.append(f"Digit {i} is {j}")
        train_concepts = np.stack(train_concepts, axis=1)
        test_concepts = np.stack(test_concepts, axis=1)

        self.concept_bank = np.concatenate((train_concepts, np.logical_not(train_concepts)), axis=1)
        self.concept_test_ground_truth = np.concatenate((test_concepts, np.logical_not(test_concepts)), axis=1)
        self.concept_names = concept_names + list(map(lambda s: "NOT " + s, concept_names))

        self.n_concepts = n_digits
        self.n_tasks = max_digit * n_digits + 1

        if foundation_model == "dinov2":
            self.latent_representation_size = n_digits * 1536
        elif foundation_model == "clip":
            self.latent_representation_size = n_digits * 768
        else:
            self.latent_representation_size = None

    def run_foundation_model(self, img_transform, model_dir, data_getter, device):
        if self.foundation_model == "dinov2":
            torch.hub.set_dir(Path(model_dir) / "dinov2")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
            model.eval()
            transform = transforms.default_transforms
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

        pre_transform = torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        xs = []
        ys = []
        cs = []
        with torch.no_grad():
            for i in trange(data_getter.length):
                imgs, y, c = data_getter(i)

                x_full = []
                for j in range(imgs.shape[0]):
                    img = imgs[j]
                    img = torch.repeat_interleave(img[torch.newaxis, ...], 3, dim=0)
                    img = transform(pre_transform(img))
                    img = img[torch.newaxis, ...].to(device)
                    x = model(img).detach().cpu().squeeze().float()
                    x_full.append(x)

                xs.append(torch.concatenate(x_full))
                ys.append(y)
                cs.append(c)
        return torch.stack(xs), torch.tensor(ys), torch.stack(cs)
