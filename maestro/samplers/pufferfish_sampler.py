from .base_sampler import BaseSampler


class PufferfishSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super(PufferfishSampler, self).__init__(*args, **kwargs)

    def width_sampler(self):
        while True:
            yield 0.25
