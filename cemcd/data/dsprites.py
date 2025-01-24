from pathlib import Path
import numpy as np
import torch
import torchvision
from cemcd.data.base import Datasets

class DspritesDatasets(Datasets):
    def __init__(
            self,
            foundation_model=None,
            dataset_dir="/datasets",
            cache_dir=None,
            model_dir="/checkpoints",
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        dataset_zip = np.load(Path(dataset_dir) / "dSprites" / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

        latents = dataset_zip["latents_classes"]
        no_rotation = latents[:, 3].squeeze() == 0

        self.imgs = dataset_zip["imgs"][no_rotation]

        l = len(self.imgs)
        generator = np.random.default_rng(42)
        self.permutation = generator.permutation(l)

        self.val_start = int(0.6*l)
        self.test_start = int(0.75*l)
        self.length = l

        latents = latents[no_rotation][self.permutation]
        loc = latents[:, (False, False, False, False, True, True)] > 15
        quadrant = loc[:, 0] * 2 + loc[:, 1]
        shape = latents[:, (False, True, False, False, False, False)].squeeze()
        scale = latents[:, (False, False, True, False, False, False)].squeeze()

        self.quadrant_train = torch.tensor(quadrant[:self.val_start])
        self.shape_train = torch.tensor(shape[:self.val_start])
        self.scale_train = torch.tensor(scale[:self.val_start])

        self.quadrant_val = torch.tensor(quadrant[self.val_start:self.test_start])
        self.shape_val = torch.tensor(shape[self.val_start:self.test_start])
        self.scale_val = torch.tensor(scale[self.val_start:self.test_start])

        self.quadrant_test = torch.tensor(quadrant[self.test_start:])
        self.shape_test = torch.tensor(shape[self.test_start:])
        self.scale_test = torch.tensor(scale[self.test_start:])

        def data_getter(split):
            if split == "train":
                concepts = []
                concepts.append(self.quadrant_train > 1)
                concepts.append(self.shape_train == 0)
                concepts.append(self.scale_train > 2)
                c = torch.stack(concepts, dim=1).float()
                y = self.quadrant_train + self.shape_train + self.scale_train
                imgs_start = 0
                imgs_end = self.val_start
            elif split == "val":
                concepts = []
                concepts.append(self.quadrant_val > 1)
                concepts.append(self.shape_val == 0)
                concepts.append(self.scale_val > 2)
                c = torch.stack(concepts, dim=1).float()
                y = self.quadrant_val + self.shape_val + self.scale_val
                imgs_start = self.val_start
                imgs_end = self.test_start
            elif split == "test":
                concepts = []
                concepts.append(self.quadrant_test > 1)
                concepts.append(self.shape_test == 0)
                concepts.append(self.scale_test > 2)
                c = torch.stack(concepts, dim=1).float()
                y = self.quadrant_test + self.shape_test + self.scale_test
                imgs_start = self.test_start
                imgs_end = self.length
            else:
                raise ValueError(f"Invalid split: {split}")
            transform = torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

            def getter(idx):
                img_idx = self.permutation[idx + imgs_start]
                img = self.imgs[img_idx]
                img = np.repeat(img[np.newaxis, ...], 3, axis=0)
                img = transform(img)

                return img, y[idx], c[idx]

            getter.length = imgs_end - imgs_start
            return getter

        super().__init__(
            train_getter=data_getter("train"),
            val_getter=data_getter("val"),
            test_getter=data_getter("test"),
            foundation_model=foundation_model,
            train_img_transform=None,
            val_test_img_transform=None,
            cache_dir=cache_dir,
            model_dir=model_dir,
            device=device
        )

        train_concepts = np.stack((
            self.scale_train == 0,
            self.scale_train == 1,
            self.scale_train == 2,
            self.scale_train == 3,
            self.scale_train == 4,
            self.scale_train == 5,
            self.shape_train == 0,
            self.shape_train == 1,
            self.shape_train == 2,
            self.quadrant_train == 0,
            self.quadrant_train == 1,
            self.quadrant_train == 2,
            self.quadrant_train == 3
        ), axis=1)
        self.concept_bank = np.concatenate((train_concepts, np.logical_not(train_concepts)), axis=1)

        test_concepts = np.stack((
            self.scale_test == 0,
            self.scale_test == 1,
            self.scale_test == 2,
            self.scale_test == 3,
            self.scale_test == 4,
            self.scale_test == 5,
            self.shape_test == 0,
            self.shape_test == 1,
            self.shape_test == 2,
            self.quadrant_test == 0,
            self.quadrant_test == 1,
            self.quadrant_test == 2,
            self.quadrant_test == 3
        ), axis=1)
        self.concept_test_ground_truth = np.concatenate((test_concepts, np.logical_not(test_concepts)), axis=1)

        names = [
            "Scale 0",
            "Scale 1",
            "Scale 2",
            "Scale 3",
            "Scale 4",
            "Scale 5",
            "Shape 0",
            "Shape 1",
            "Shape 2",
            "Quadrant 0",
            "Quadrant 1",
            "Quadrant 2",
            "Quadrant 3"
        ]
        self.concept_names = names + list(map(lambda s: "NOT " + s, names))

        self.n_concepts = 3
        self.n_tasks = 11
