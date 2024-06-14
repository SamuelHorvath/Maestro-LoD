'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

from ..layers.conv import MaestroConv2d
from ..layers.linear import MaestroLinear


def create_linear_layer(decomposition=False, *args, **kwargs):
    if decomposition:
        return MaestroLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


def create_conv2d_layer(decomposition=False, *args, **kwargs):
    if decomposition:
        return MaestroConv2d(*args, **kwargs)
    else:
        return nn.Conv2d(*args, **kwargs)


class MaestroLeNet(nn.Module):
    def __init__(self, decomposition=False):
        super().__init__()
        self.conv1 = create_conv2d_layer(decomposition, 1, 6, 5)
        self.conv2 = create_conv2d_layer(decomposition, 6, 16, 5)
        self.fc1 = create_linear_layer(decomposition, 256, 120)
        self.fc2 = create_linear_layer(decomposition, 120, 84)
        self.fc3 = create_linear_layer(decomposition, 84, 10)

    def forward(self, input, sampler=None):
        if sampler is None:
            out = F.relu(self.conv1(input))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
        else:
            out = F.relu(self.conv1(input, sampler()))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out, sampler()))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out, sampler()))
            out = F.relu(self.fc2(out, sampler()))
            out = self.fc3(out, sampler())
        return out
