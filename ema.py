import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    # use the formula in https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    def forward(self, name, x):
        assert name in self.shadow
        # new_avg = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] -= (1 - self.mu) * (self.shadow[name] - x)
        return new_avg
