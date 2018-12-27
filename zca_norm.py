import torch


class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x: torch.Tensor):
        s = x.size()
        dim1 = torch.tensor(s[1:])
        x = x.clone().reshape((s[0], torch.prod(dim1)))
        m = torch.mean(x, dim=0)
        x -= m
        sigma = torch.mm(x.t(), x) / x.size(0)
        U, S, V = torch.svd(sigma)
        tmp = torch.mm(U, torch.diag(1 / torch.sqrt(S + self.regularization)))
        tmp2 = torch.mm(U, torch.diag(torch.sqrt(S + self.regularization)))
        self.ZCA_mat = torch.mm(tmp, U.t())
        self.inv_ZCA_mat = torch.mm(tmp2, U.t())
        self.mean = m

    def apply(self, x: torch.Tensor):
        s = x.size()
        dim1 = torch.tensor(s[1:])
        return torch.mm(x.reshape((s[0], torch.prod(dim1))) - self.mean,
                        self.ZCA_mat).reshape(s)

    def invert(self, x: torch.Tensor):
        s = x.size()
        dim1 = torch.tensor(s[1:])
        return (torch.mm(x.reshape((s[0], torch.prod(dim1))), self.inv_ZCA_mat)
                + self.mean).reshape(s)
