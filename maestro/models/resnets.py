from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, \
    ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, \
    ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, \
    ResNeXt101_64X4D_Weights, Wide_ResNet50_2_Weights, \
    Wide_ResNet101_2_Weights

from ..layers.conv import MaestroConv2d
from .utils import create_linear_layer, create_conv2d_layer, \
    MaestroSequential
from ..samplers import BaseSampler

__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes: int, out_planes: int,
            stride: int = 1, groups: int = 1,
            dilation: int = 1, decomposition: bool = False
            ) -> Union[nn.Conv2d, MaestroConv2d]:
    """3x3 convolution with padding"""
    return create_conv2d_layer(
        decomposition,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int,
            stride: int = 1, decomposition: bool = False
            ) -> Union[nn.Conv2d, MaestroConv2d]:
    """1x1 convolution"""
    return create_conv2d_layer(
        decomposition,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        decomposition: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample
        # the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,
                             decomposition=decomposition)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             decomposition=decomposition)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor,
                sampler: BaseSampler = None) -> Tensor:

        if sampler is None:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
        else:
            identity = x

            if hasattr(self.conv1, 'is_od'):
                out = self.conv1(x, p=sampler())
            else:
                out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            if hasattr(self.conv2, 'is_od'):
                out = self.conv2(out, p=sampler())
            else:
                out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(
                    x, sampler=sampler)

            out += identity
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        decomposition: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample
        # the input when stride != 1
        self.conv1 = conv1x1(inplanes, width,
                             decomposition=decomposition)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation,
                             decomposition=decomposition)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion,
                             decomposition=decomposition)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor,
                sampler: BaseSampler = None) -> Tensor:
        if sampler is None:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
        else:
            identity = x

            if hasattr(self.conv1, 'is_od'):
                out = self.conv1(x, p=sampler())
            else:
                out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            if hasattr(self.conv2, 'is_od'):
                out = self.conv2(out, p=sampler())
            else:
                out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            if hasattr(self.conv3, 'is_od'):
                out = self.conv3(out, p=sampler())
            else:
                out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(
                    x, sampler=sampler)

            out += identity
            out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        decomposition: bool = False,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = create_conv2d_layer(
            decomposition, 3, self.inplanes, kernel_size=7,
            stride=2, padding=3, bias=False,)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], decomposition=decomposition)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0],
            decomposition=decomposition)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1],
            decomposition=decomposition)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2],
            decomposition=decomposition)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = create_linear_layer(
            decomposition,
            512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, MaestroConv2d):
                raise ValueError("not implemented")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        decomposition: bool = False,
    ) -> MaestroSequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MaestroSequential(
                conv1x1(self.inplanes, planes * block.expansion, stride,
                        decomposition=decomposition),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer,
                decomposition=decomposition
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    decomposition=decomposition,
                )
            )

        return MaestroSequential(*layers)

    def _forward_impl(self, x: Tensor,
                      sampler: BaseSampler = None) -> Tensor:
        # See note [TorchScript super()]
        if sampler is None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        else:
            if hasattr(self.conv1, 'is_od'):
                x = self.conv1(x, p=sampler())
            else:
                x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x, sampler=sampler)
            x = self.layer2(x, sampler=sampler)
            x = self.layer3(x, sampler=sampler)
            x = self.layer4(x, sampler=sampler)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if hasattr(self.fc, 'is_od'):
                x = self.fc(x, p=sampler())
            else:
                x = self.fc(x)
        return x

    def forward(self, x: Tensor, sampler: BaseSampler = None) -> Tensor:
        return self._forward_impl(x, sampler=sampler)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(
            kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def resnet18(*, weights: Optional[ResNet18_Weights] = None,
             progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def resnet34(*, weights: Optional[ResNet34_Weights] = None,
             progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet50(*, weights: Optional[ResNet50_Weights] = None,
             progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet101(*, weights: Optional[ResNet101_Weights] = None,
              progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet101_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnet152(*, weights: Optional[ResNet152_Weights] = None,
              progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet152_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


def resnext50_32x4d(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None,
    progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnext101_32x8d(
    *, weights: Optional[ResNeXt101_32X8D_Weights] = None,
    progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = ResNeXt101_32X8D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnext101_64x4d(
    *, weights: Optional[ResNeXt101_64X4D_Weights] = None,
    progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = ResNeXt101_64X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None,
    progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def wide_resnet101_2(
    *, weights: Optional[Wide_ResNet101_2_Weights] = None,
    progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = Wide_ResNet101_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
