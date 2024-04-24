import numpy as np
import torch

DATASET_PATH = "datasets/dSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

class DSpritesDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, imgs_start, imgs_end, permutation, concepts, labels):
        self.imgs = imgs
        self.imgs_start = imgs_start
        self.imgs_end = imgs_end
        self.permutation = permutation
        self.concepts = concepts
        self.labels = labels

    def __len__(self):
        return self.imgs_end - self.imgs_start

    def __getitem__(self, idx):
        img_idx = self.permutation[idx + self.imgs_start]
        x = torch.tensor(self.imgs[img_idx][None, :]) / 255.0

        return x, self.labels[idx], self.concepts[idx]

class DSpritesDatasets:
    def __init__(self, dataset_path=DATASET_PATH):
        dataset_zip = np.load(dataset_path)

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

    def train_dl(self, additional_concepts=None):
        concepts = []
        concepts.append(self.quadrant_train > 1)
        concepts.append(self.shape_train == 0)
        concepts.append(self.scale_train > 2)
        
        c = torch.stack(concepts, dim=1).float()
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)

        y = self.quadrant_train + self.shape_train + self.scale_train

        dataset = DSpritesDataset(self.imgs, 0, self.val_start, self.permutation, c, y)

        return torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=7)


    def val_dl(self, additional_concepts=None):
        concepts = []
        concepts.append(self.quadrant_val > 1)
        concepts.append(self.shape_val == 0)
        concepts.append(self.scale_val > 2)
        
        c = torch.stack(concepts, dim=1).float()
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)

        y = self.quadrant_val + self.shape_val + self.scale_val

        dataset = DSpritesDataset(self.imgs, self.val_start, self.test_start, self.permutation, c, y)

        return torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=7)

    def test_dl(self, additional_concepts=None):
        concepts = []
        concepts.append(self.quadrant_test > 1)
        concepts.append(self.shape_test == 0)
        concepts.append(self.scale_test > 2)
        
        c = torch.stack(concepts, dim=1).float()
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)

        y = self.quadrant_test + self.shape_test + self.scale_test

        dataset = DSpritesDataset(self.imgs, self.test_start, self.length, self.permutation, c, y)
    
        return torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=7)
