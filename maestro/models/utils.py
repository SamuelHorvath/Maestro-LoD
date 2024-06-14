from torch import nn

from ..layers.linear import MaestroLinear
from ..layers.conv import MaestroConv2d


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


class MaestroSequential(nn.Sequential):

    def __init__(self, *layers):
        super(MaestroSequential, self).__init__(*layers)

    def forward(self, input, sampler=None):
        if sampler is None:
            for module in self:
                input = module(input)
        else:
            for module in self:
                if isinstance(module, MaestroLinear) or \
                        isinstance(module, MaestroConv2d):
                    input = module(input, sampler())
                else:
                    contains_maestro_block = False
                    for submodule in module.modules():
                        if isinstance(submodule, MaestroLinear) or \
                                isinstance(submodule, MaestroConv2d):
                            contains_maestro_block = True
                            break
                    if contains_maestro_block:
                        input = module(input, sampler)
                    else:
                        input = module(input)
        return input
