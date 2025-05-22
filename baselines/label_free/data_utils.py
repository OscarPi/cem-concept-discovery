import os
import torch
from tqdm import tqdm
from cemcd.data.transforms import dino_transforms
from cemcd.data import cub, awa, shapes

def get_data(dataset_name, dataset_dir, preprocess=dino_transforms):
    if dataset_name == "cub_train":
        datasets = cub.CUBDatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.train_dl(no_concepts=True)
    elif dataset_name == "cub_val":
        datasets = cub.CUBDatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.val_dl(no_concepts=True)
    elif dataset_name == "cub_test":
        datasets = cub.CUBDatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.test_dl(no_concepts=True)
    elif dataset_name == "awa_train":
        datasets = awa.AwADatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.train_dl(no_concepts=True)
    elif dataset_name == "awa_val":
        datasets = awa.AwADatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.val_dl(no_concepts=True)
    elif dataset_name == "awa_test":
        datasets = awa.AwADatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.test_dl(no_concepts=True)
    elif dataset_name == "shapes_train":
        datasets = shapes.ShapesDatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.train_dl(no_concepts=True)
    elif dataset_name == "shapes_val":
        datasets = shapes.ShapesDatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.val_dl(no_concepts=True)
    elif dataset_name == "shapes_test":
        datasets = shapes.ShapesDatasets(dataset_dir=dataset_dir)
        datasets.train_img_transform = preprocess
        datasets.val_test_img_transform = preprocess
        return datasets.test_dl(no_concepts=True)
               
    raise ValueError(f"Unknown dataset: {dataset_name}")

def get_targets_only(dataset_name, dataset_dir):
    dl = get_data(dataset_name, dataset_dir)
    targets = torch.tensor([])
    for x, y in tqdm(dl):
        targets = torch.concat((targets, y.to(targets.device)))
    return targets.long()

def get_target_model(device, model_dir):
    torch.hub.set_dir(os.path.join(model_dir, "dinov2"))
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
    model.eval()
    preprocess = dino_transforms
    target_model = lambda x: model(x).float()

    return target_model, preprocess
