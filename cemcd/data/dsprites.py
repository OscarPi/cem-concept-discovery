from pathlib import Path
import numpy as np
import torch
import torchvision
from cemcd.data.base import Datasets

class DSpritesDatasets(Datasets):
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

        imgs = dataset_zip["imgs"][no_rotation]

        length = len(imgs)
        generator = np.random.default_rng(42)
        permutation = generator.permutation(length)
        imgs = imgs[permutation]

        val_start = int(0.6*length)
        test_start = int(0.75*length)

        latents = latents[no_rotation][permutation]
        loc = latents[:, (False, False, False, False, True, True)] > 15
        quadrant = loc[:, 0] * 2 + loc[:, 1]
        shape = latents[:, (False, True, False, False, False, False)].squeeze()
        scale = latents[:, (False, False, True, False, False, False)].squeeze()


        imgs_train = imgs[:val_start]
        quadrant_train = torch.tensor(quadrant[:val_start])
        shape_train = torch.tensor(shape[:val_start])
        scale_train = torch.tensor(scale[:val_start])
        c_train = torch.stack((
            quadrant_train > 1,
            shape_train == 0,
            scale_train > 2
        ), dim=1).float()
        y_train = quadrant_train + shape_train + scale_train

        imgs_val = imgs[val_start:test_start]
        quadrant_val = torch.tensor(quadrant[val_start:test_start])
        shape_val = torch.tensor(shape[val_start:test_start])
        scale_val = torch.tensor(scale[val_start:test_start])
        c_val = torch.stack((
            quadrant_val > 1,
            shape_val == 0,
            scale_val > 2
        ), dim=1).float()
        y_val = quadrant_val + shape_val + scale_val

        imgs_test = imgs[test_start:]
        quadrant_test = torch.tensor(quadrant[test_start:])
        shape_test = torch.tensor(shape[test_start:])
        scale_test = torch.tensor(scale[test_start:])
        c_test = torch.stack((
            quadrant_test > 1,
            shape_test == 0,
            scale_test > 2
        ), dim=1).float()
        y_test = quadrant_test + shape_test + scale_test

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)])

        def data_getter(imgs, y, c):
            def getter(idx):
                img = imgs[idx]
                img = np.repeat(img[..., np.newaxis], 3, axis=2)
                img = transform(img)

                return img, y[idx], c[idx]

            getter.length = len(imgs)
            return getter

        super().__init__(
            train_getter=data_getter(imgs_train, y_train, c_train),
            val_getter=data_getter(imgs_val, y_val, c_val),
            test_getter=data_getter(imgs_test, y_test, c_test),
            foundation_model=foundation_model,
            train_img_transform=None,
            val_test_img_transform=None,
            cache_dir=cache_dir,
            model_dir=model_dir,
            device=device
        )

        train_concepts = np.stack((
            scale_train == 0,
            scale_train == 1,
            scale_train == 2,
            scale_train == 3,
            scale_train == 4,
            scale_train == 5,
            shape_train == 0,
            shape_train == 1,
            shape_train == 2,
            quadrant_train == 0,
            quadrant_train == 1,
            quadrant_train == 2,
            quadrant_train == 3
        ), axis=1)
        self.concept_bank = np.concatenate((train_concepts, np.logical_not(train_concepts)), axis=1)

        test_concepts = np.stack((
            scale_test == 0,
            scale_test == 1,
            scale_test == 2,
            scale_test == 3,
            scale_test == 4,
            scale_test == 5,
            shape_test == 0,
            shape_test == 1,
            shape_test == 2,
            quadrant_test == 0,
            quadrant_test == 1,
            quadrant_test == 2,
            quadrant_test == 3
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
