from .independent import IndependentSampler
from .single_layer import SingleLayerSampler
from .pufferfish_sampler import PufferfishSampler


def get_sampler(sampler_name, model, with_layer=False):
    if sampler_name is None:
        return None
    elif sampler_name == 'across_layers':
        return IndependentSampler(model, with_layer=with_layer)
    elif sampler_name == 'per_layer':
        return SingleLayerSampler(model, with_layer=with_layer)
    elif sampler_name == 'pufferfish':
        return PufferfishSampler(model, with_layer=with_layer)
    else:
        raise ValueError(f'Unknown sampler {sampler_name}')
