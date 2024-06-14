import torch
from torch import nn, norm
from torch.nn.parameter import Parameter
import numpy as np

from .utils import check_layer


__all__ = ["MaestroConv2d", "decompose_conv2d"]


class MaestroConv2d(nn.Module):
    def __init__(self, in_features, out_features,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(MaestroConv2d, self).__init__()

        self.in_channels = in_features
        self.out_channels = out_features
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if groups != 1:
            raise NotImplementedError

        kernel_total_size = np.prod(kernel_size) if isinstance(
            kernel_size, tuple) else kernel_size**2
        self.inner_dim = min(out_features, in_features*kernel_total_size)

        self.conv_u = nn.Conv2d(in_features, self.inner_dim, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, bias=False)
        self.conv_v = nn.Conv2d(self.inner_dim, out_features, 1,
                                stride=1, padding=0, bias=bias)
        self.low_rank = (self.importance() >= 0).cpu()  # all True
        # for sampler
        self.is_od = True

    @property
    def width(self):
        return self.inner_dim

    def forward(self, input, p=None):
        p = check_layer(self, p)
        if not p:
            inner_dim = self.inner_dim
        else:
            assert 0 < p <= 1
            inner_dim = int(np.ceil(self.inner_dim * p))

        # only sample low_rank coefficients
        weight_u = self.conv_u.weight[self.low_rank, :, :, :][
            :inner_dim, :, :, :]
        weight_v = self.conv_v.weight[:, self.low_rank, :, :][
            :, :inner_dim, :, :]
        # do not go through the inner layers if inner_dim is 0
        if inner_dim == 0:
            if self.conv_v.bias is None:
                bias = torch.zeros(self.out_channels).to(input.device)
            else:
                bias = self.conv_v.bias
            out = bias.unsqueeze(0)
            for _ in range(len(input.shape) - 2):
                out = out.unsqueeze(0)
            return out
        else:
            intermediate = nn.functional.conv2d(
                input, weight_u, None, self.stride, self.padding,
                self.dilation, groups=1)
            y = nn.functional.conv2d(
                intermediate, weight_v, self.conv_v.bias, 1, 0,
                dilation=1, groups=1
                )
            return y

    def _compute_lasso(self, keepdim=False, hierarchical=False):
        conv_u_resh = self.conv_u.weight.reshape(
            self.conv_u.weight.shape[0], -1)
        conv_v_resh = self.conv_v.weight.swapaxes(0, 1).reshape(
            self.conv_v.weight.shape[1], -1)
        if not hierarchical:
            in_lasso = norm(conv_u_resh, dim=1)
            out_lasso = norm(conv_v_resh, dim=1)
        else:
            in_lasso = conv_u_resh.pow(2).flip(
                dims=[0]).cumsum(dim=0).flip(dims=[0]).sum(dim=1).sqrt()
            out_lasso = conv_v_resh.pow(2).flip(
                dims=[0]).cumsum(dim=0).flip(dims=[0]).sum(dim=1).sqrt()
        if keepdim:
            in_lasso = in_lasso.reshape(-1, 1, 1, 1)
            out_lasso = out_lasso.reshape(1, -1, 1, 1)
        return in_lasso, out_lasso

    def group_lasso(self, p=None, hierarchical=False):
        in_lasso, out_lasso = self._compute_lasso(
            hierarchical=hierarchical)
        res = in_lasso + out_lasso
        if p is None:
            return res
        else:
            inner_dim = int(np.ceil(self.inner_dim * p))
            return res[:inner_dim]

    @torch.no_grad()
    def importance(self, hierarchical=False):
        in_lasso, out_lasso = self._compute_lasso(
            hierarchical=hierarchical)
        return in_lasso * out_lasso

    @torch.no_grad()
    def assign_low_rank(self, treshold, prune=False,
                        importance=None, hierarchical=False):
        if importance is None:
            importance = self.importance(hierarchical=hierarchical)
        low_rank = importance >= treshold
        low_rank_total = low_rank.sum().item()
        # necessary for multi-gpu
        self.low_rank = low_rank.cpu()
        self.inner_dim = low_rank_total
        if prune:
            # do not prune during training as you would need to
            # register these weights for the optimizer
            self.conv_u.weight = \
                Parameter(self.conv_u.weight[low_rank, :, :, :].data)
            self.conv_v.weight = \
                Parameter(self.conv_v.weight[:, low_rank, :, :].data)
            self.low_rank = (self.importance(
                hierarchical=hierarchical) >= 0).cpu()  # all True


def decompose_conv2d(layer: nn.Conv2d, device: str = None):
    if device is None:
        device = layer.weight.device
    maestro_layer = MaestroConv2d(
        in_features=layer.in_channels,
        out_features=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=True if layer.bias is not None else False,
        ).to(device)
    maestro_layer.conv_v.bias = layer.bias
    U, S, Vh = torch.linalg.svd(
        layer.weight.reshape(layer.out_channels, -1), full_matrices=False)
    sqrt_S = S.reshape(1, -1)**(1/2)
    U = U * sqrt_S
    V = Vh.T * sqrt_S
    maestro_layer.conv_v.weight.data = U.reshape(
        layer.out_channels, -1, 1, 1)
    maestro_layer.conv_u.weight.data = V.T.reshape(
        -1, layer.in_channels, *layer.kernel_size)
    return maestro_layer


def do_not_decompose_conv2d(layer: MaestroConv2d):
    n_params_decomposed = \
        layer.conv_u.weight.numel() + layer.conv_v.weight.numel()
    conv2d = nn.Conv2d(layer.in_channels, layer.out_channels,
                       layer.conv_u.kernel_size,
                       stride=layer.stride, padding=layer.padding,
                       dilation=layer.dilation, bias=False)
    n_params_full = conv2d.weight.numel()
    return n_params_full < n_params_decomposed


def maestro_to_full_conv2d(layer: MaestroConv2d, device: str = None):
    if device is None:
        device = layer.conv_v.weight.device
    full_layer = nn.Conv2d(
        layer.in_channels, layer.out_channels,
        layer.conv_u.kernel_size,
        stride=layer.stride, padding=layer.padding,
        dilation=layer.dilation, bias=layer.bias).to(device)
    # remove last two dimensions of conv_u.weight
    conv_v_weight = layer.conv_v.weight.data.clone().reshape(
        *layer.conv_v.weight.shape[:2])
    conv_u_weight = layer.conv_u.weight.data.clone().reshape(
        layer.conv_u.weight.shape[0], -1)
    new_weight = conv_v_weight @ conv_u_weight
    full_layer.weight.data = new_weight.reshape(
        *full_layer.weight.shape)
    if full_layer.bias is not None:
        full_layer.bias.data = layer.conv_v.bias.data.clone()
    return full_layer
