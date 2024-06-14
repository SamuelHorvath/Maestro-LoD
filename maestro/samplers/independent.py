import numpy as np

from .base_sampler import BaseSampler


class IndependentSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super(IndependentSampler, self).__init__(*args, **kwargs)

    def width_sampler(self):
        while True:
            for i in range(self.num_od_layers):
                yield np.random.randint(
                    1, self.widths[i] + 1) / self.widths[i]
