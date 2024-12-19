import torch
import numpy as np

class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, output_padding=0):
        super().__init__()

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            output_padding=output_padding
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.upsample = None
        if stride != 1:
            self.upsample = torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                output_padding=output_padding
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

def get_resnet_decoder(input_size, output_size):
    output_size = np.array(output_size)
    s1 = np.floor((output_size - 1) / 2 + 1)
    s2 = np.floor((s1 - 1) / 2 + 1)
    s3 = np.floor((s2 - 1) / 2 + 1)
    s4 = np.floor((s3 - 1) / 2 + 1)
    s5 = np.floor((s4 - 1) / 2 + 1)

    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.Unflatten(1, (512, 1, 1)),
        torch.nn.Upsample(size=tuple(map(int, s5))),
        BasicBlock(512, 512),
        BasicBlock(512, 512),
        BasicBlock(512, 256, stride=2, output_padding=tuple(map(int, s4 - 2*(s5 - 1) - 1))),
        BasicBlock(256, 256),
        BasicBlock(256, 256),
        BasicBlock(256, 256),
        BasicBlock(256, 256),
        BasicBlock(256, 256),
        BasicBlock(256, 128, stride=2, output_padding=tuple(map(int, s3 - 2*(s4 - 1) - 1))),
        BasicBlock(128, 128),
        BasicBlock(128, 128),
        BasicBlock(128, 128),
        BasicBlock(128, 64, stride=2, output_padding=tuple(map(int, s2 - 2*(s3 - 1) - 1))),
        BasicBlock(64, 64),
        BasicBlock(64, 64),
        BasicBlock(64, 64),
        torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            output_padding=tuple(map(int, s1 - 2*(s2 - 1) - 1))
        ),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            output_padding=tuple(map(int, output_size - 2*(s1 - 1) - 1))
        )
    )
