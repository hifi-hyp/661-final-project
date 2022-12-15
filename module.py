# import necessary dependencies
import argparse
import os, sys
import time
import datetime
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.modules import BatchNorm2d


# define the resnet mode;
class BuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.shortcut is None else self.shortcut(x)
        return F.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, num_blocks, out_channels):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self.make_layer(16, 16, num_blocks[0], 1)
        self.layer2 = self.make_layer(16, 32, num_blocks[1], 2)
        self.layer3 = self.make_layer(32, 64, num_blocks[2], 2)
        self.fc = nn.Linear(64, out_channels)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BuildingBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BuildingBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = torch.flatten(out, 1)
        return self.fc(out), out


# net = ResNet([3, 3, 3], 10)
# data = data = torch.randn(5, 3, 32, 32)
# # Forward pass "data" through "net" to get output "out"
# out = net(data)
# print(out.shape)



