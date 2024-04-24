import torch

def get_pre_concept_model(width, height, channels, output_dim=128):
    intermediate_maps = 16
    return torch.nn.Sequential(*[
        torch.nn.Conv2d(
            in_channels=channels,
            out_channels=intermediate_maps,
            kernel_size=(3,3),
            padding='same',
        ),
        torch.nn.BatchNorm2d(num_features=intermediate_maps),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(
            in_channels=intermediate_maps,
            out_channels=intermediate_maps,
            kernel_size=(3,3),
            padding='same',
        ),
        torch.nn.BatchNorm2d(num_features=intermediate_maps),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(
            in_channels=intermediate_maps,
            out_channels=intermediate_maps,
            kernel_size=(3,3),
            padding='same',
        ),
        torch.nn.BatchNorm2d(num_features=intermediate_maps),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(
            in_channels=intermediate_maps,
            out_channels=intermediate_maps,
            kernel_size=(3,3),
            padding='same',
        ),
        torch.nn.BatchNorm2d(num_features=intermediate_maps),
        torch.nn.LeakyReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(
            width*height*intermediate_maps,
            output_dim,
        ),
    ])
