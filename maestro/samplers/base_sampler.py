class BaseSampler:

    def __init__(self, model, with_layer=False):
        self.model = model
        self.with_layer = with_layer
        self.prepare_sampler()

    def _prepare_sampler(self):
        self.num_od_layers = 0
        self.widths = []
        self.od_layers = []
        for m in self.model.modules():
            if hasattr(m, 'is_od') and m.is_od:
                self.num_od_layers += 1
                self.widths.append(m.width)
                self.od_layers.append(m)
        if self.num_od_layers == 0:
            raise ValueError('No layers to sample')

    def prepare_sampler(self):
        self._prepare_sampler()
        self.width_samples = self.width_sampler()
        self.layer_samples = self.layer_sampler()

    def width_sampler(self):
        while True:
            yield None

    def layer_sampler(self):
        while True:
            for m in self.od_layers:
                yield m

    def __call__(self):
        if self.with_layer:
            return next(self.width_samples), next(self.layer_samples)
        else:
            return next(self.width_samples)
