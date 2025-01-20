# Adapted from https://github.com/mlbio-epfl/turtle/blob/main/dataset_preparation/data_utils.py

import torchvision.transforms
import torch

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def _convert_image_to_rgb(image):
    if torch.is_tensor(image):
        return image
    else:
        return image.convert("RGB")

def _safe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torchvision.transforms.ToTensor()(x)

default_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(224),
    _convert_image_to_rgb,
    _safe_to_tensor,
    torchvision.transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

resnet_train = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
    torchvision.transforms.RandomResizedCrop(299),
    torchvision.transforms.RandomHorizontalFlip(),
    _safe_to_tensor,
    torchvision.transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
])

resnet_val_test = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(299),
    _safe_to_tensor,
    torchvision.transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
])
