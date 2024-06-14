from torch import nn

from .linear import decompose_linear, MaestroLinear, \
    do_not_decompose_linear, maestro_to_full_linear
from .conv import decompose_conv2d, MaestroConv2d, \
    do_not_decompose_conv2d, maestro_to_full_conv2d


def get_submodel_and_name(model, name):
    levels = name.split('.')
    submodel = model
    for level in levels[:-1]:
        try:
            submodel = getattr(submodel, level)
        except AttributeError:
            continue
    return submodel, levels[-1]


def decompose_model(model: nn.Module, ignore_k_first_layers=0,
                    ignore_last_layer=False):
    ignored = 0

    iterate_over = [(name, m) for name, m in model.named_modules()]
    if ignore_last_layer:
        # weight and bias of last layer
        iterate_over = iterate_over[:-2]

    for name, m in iterate_over:
        # if len(list(m.children())) > 0:
        #     # compound module, go inside it
        #     decompose_model(m, device)
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if ignored < ignore_k_first_layers:
                ignored += 1
                continue

        if isinstance(m, nn.Linear):
            submodel, level_name = get_submodel_and_name(model, name)
            setattr(submodel, level_name, decompose_linear(m))
        elif isinstance(m, nn.Conv2d):
            submodel, level_name = get_submodel_and_name(model, name)
            setattr(submodel, level_name, decompose_conv2d(m))


def model_to_full(model: nn.Module):
    for name, m in model.named_modules():
        # if len(list(m.children())) > 0:
        #     # compound module, go inside it
        #     model_to_full(m)
        pass
        if isinstance(m, MaestroLinear):
            if do_not_decompose_linear(m):
                submodel, level_name = get_submodel_and_name(model, name)
                setattr(submodel, level_name,
                        maestro_to_full_linear(m,))
        elif isinstance(m, MaestroConv2d):
            if do_not_decompose_conv2d(m):
                submodel, level_name = get_submodel_and_name(model, name)
                setattr(submodel, level_name,
                        maestro_to_full_conv2d(m))
