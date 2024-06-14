import numpy as np

from .base_sampler import BaseSampler


class SingleLayerSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super(SingleLayerSampler, self).__init__(*args, **kwargs)

    def create_samples(self):
        sample_s = []
        for i in range(self.num_od_layers):
            for j in range(1, self.widths[i] + 1):
                sample = self.num_od_layers * [None]
                sample[i] = j / self.widths[i]
                sample_s.append(sample)
        self.samples = np.array(sample_s)

    def prepare_sampler(self):
        self._prepare_sampler()
        self.create_samples()
        self.width_samples = self.width_sampler()
        self.layer_samples = self.layer_sampler()

    def width_sampler(self):
        while True:
            np.random.shuffle(self.samples)
            for sample in self.samples.flatten():
                yield sample
