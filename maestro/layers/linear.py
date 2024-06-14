import numpy as np

import torch
from torch import nn, Tensor, norm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.module import Module

from .utils import check_layer


__all__ = ["MaestroLinear", "decompose_linear"]


class MaestroLinear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 ) -> None:
        super(MaestroLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.inner_dim = min(self.in_features, self.out_features)
        # decompose layer with 2 NN layers
        # new decomposed layer as UV^T
        self.in_network = nn.Linear(
            self.in_features, self.inner_dim, bias=False)
        self.out_network = nn.Linear(
            self.inner_dim, self.out_features, bias=bias)
        self.reset_parameters()
        self.low_rank = (self.importance() >= 0).cpu()  # all True
        # for sampler
        self.is_od = True

    @property
    def width(self):
        return self.inner_dim

    def reset_parameters(self) -> None:
        self.in_network.reset_parameters()
        self.out_network.reset_parameters()

    def forward(self, input: Tensor, p=None) -> Tensor:
        p = check_layer(self, p)

        if not p:
            inner_dim = self.inner_dim
        else:
            assert 0 < p <= 1
            inner_dim = int(np.ceil(self.inner_dim * p))

        # only sample low_rank coefficients
        weight_in = self.in_network.weight[self.low_rank, :][:inner_dim, :]
        weight_out = self.out_network.weight[:, self.low_rank][
            :, :inner_dim]
        # do not go through the inner layers if inner_dim is 0
        if inner_dim == 0:
            if self.out_network.bias is None:
                bias = torch.zeros(self.out_features).to(input.device)
            else:
                bias = self.out_network.bias
            out = bias.unsqueeze(0)
            for _ in range(len(input.shape) - 2):
                out = out.unsqueeze(0)
            out = out.repeat(*input.shape[:-1], 1)
        else:
            intermediate = F.linear(input, weight_in, bias=None)
            out = F.linear(intermediate, weight_out, self.out_network.bias)
        return out

    def _compute_lasso(self, keepdim=False, hierarchical=False):
        if not hierarchical:
            in_lasso = norm(self.in_network.weight, dim=1, keepdim=keepdim)
            out_lasso = norm(self.out_network.weight, dim=0, keepdim=keepdim)
        else:
            # do not initialize weights with zeros as cumsum will return
            # nan for the gradient
            in_lasso = self.in_network.weight.pow(2).flip(
                dims=[0]).cumsum(dim=0).flip(dims=[0]).sum(
                dim=1, keepdim=keepdim).sqrt()
            out_lasso = self.out_network.weight.pow(2).flip(
                dims=[1]).cumsum(dim=1).flip(dims=[1]).sum(
                dim=0, keepdim=keepdim).sqrt()
        return in_lasso, out_lasso

    def group_lasso(self, p=None, hierarchical=False):
        in_lasso, out_lasso = self._compute_lasso(
            hierarchical=hierarchical)
        lasso = in_lasso + out_lasso
        if p is None:
            return lasso
        else:
            out_keep = int(np.ceil(self.inner_dim * p))
            return lasso[:out_keep]

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
        # for multi-gpu
        self.low_rank = low_rank.cpu()
        self.inner_dim = low_rank_total
        if prune:
            # do not prune during training as you would need to
            # register these weights for the optimizer
            self.in_network.weight = \
                Parameter(self.in_network.weight[low_rank, :].data)
            self.out_network.weight = \
                Parameter(self.out_network.weight[:, low_rank].data)
            self.low_rank = (self.importance(
                hierarchical=hierarchical) >= 0).cpu()  # all True

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.out_network is not None
        )


def decompose_linear(layer: nn.Linear, device: str = None):
    if device is None:
        device = layer.weight.device
    maestro_layer = MaestroLinear(
        layer.in_features, layer.out_features,
        True if layer.bias is not None else False
        ).to(device)
    maestro_layer.out_network.bias = layer.bias
    U, S, Vh = torch.linalg.svd(layer.weight.data, full_matrices=False)
    sqrt_S = S.reshape(1, -1)**(1/2)
    U = U * sqrt_S
    V = Vh.T * sqrt_S
    maestro_layer.out_network.weight.data = U
    maestro_layer.in_network.weight.data = V.T
    return maestro_layer


def do_not_decompose_linear(layer: MaestroLinear):
    n_params_decomposed = \
        (layer.in_features + layer.out_features) * layer.inner_dim
    n_params_full = layer.in_features * layer.out_features
    return n_params_full < n_params_decomposed


def maestro_to_full_linear(layer: MaestroLinear, device: str = None):
    if device is None:
        device = layer.in_network.weight.device
    full_layer = nn.Linear(
        layer.in_features, layer.out_features,
        True if layer.out_network.bias is not None else False
        ).to(device)
    full_layer.weight.data = layer.out_network.weight.data.clone() \
        @ layer.in_network.weight.data.clone()
    if full_layer.bias is not None:
        full_layer.bias.data = layer.out_network.bias.data.clone()
    return full_layer
