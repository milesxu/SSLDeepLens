import torch.nn as nn
import copy


class EMA:
    def __init__(self, mu, net: nn.Module, has_cuda=True):
        super(EMA, self).__init__()
        self.mu = mu
        self.src = net
        self.cpy = copy.deepcopy(net)
        if has_cuda:
            self.cpy = self.cpy.cuda()

    # use the formula in https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    def update(self):
        for s_param, c_param in \
                zip(self.src.parameters(), self.cpy.parameters()):
            if s_param.requires_grad:
                c_param.data -= (1 - self.mu) * (c_param.data - s_param.data)

    def __call__(self, x):
        self.cpy.eval()
        return self.cpy(x)
