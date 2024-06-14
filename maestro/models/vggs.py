'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from ..layers.conv import MaestroConv2d
from ..layers.linear import MaestroLinear
from .utils import MaestroSequential


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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


class VGG(nn.Module):
    def __init__(self, vgg_name, norm_layer=None,
                 decomposition=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.features = self._make_layers(
            cfg[vgg_name], norm_layer, decomposition)
        self.classifier = create_linear_layer(
            decomposition, 512, 10)

    def forward(self, input, sampler=None):
        out = self.features(input, sampler)
        out = out.view(out.size(0), -1)
        if sampler is None:
            out = self.classifier(out)
        else:
            out = self.classifier(out, sampler())
        return out

    def _make_layers(self, cfg, norm_layer, decomposition):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [create_conv2d_layer(
                    decomposition, in_channels, x, kernel_size=3, padding=1),
                           norm_layer(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return MaestroSequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
