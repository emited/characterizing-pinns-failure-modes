import torch
from torch import nn

from pbc_examples.modules.max_spectral_norm import max_spectral_norm
from pbc_examples.net_pbc import get_activation



# class ActivationsAndLinear(nn.Module):
#     def __init__(self, in_dim, out_features, activations, mults=None):
#         super().__init__()
#         # self.lin = nn.Linear(len(activations) * in_dim, out_features)
#         self.lins = nn.ModuleList([nn.Linear(in_dim, out_features)] * len(activations))
#         self.lins[1].weight.data.zero_()
#         self.acts = [get_activation(act)() for act in activations]
#         self.mults = mults
#         if mults is None:
#             self.mults = (1,) * len(self.acts)
#
#     def forward(self, x):
#         # return self.lin(torch.cat([mult * act(x) for mult, act in zip(self.mults, self.acts)], -1))
#         return torch.mean(torch.stack([lin(mult * act(x)) for mult, act, lin in zip(self.mults, self.acts, self.lins)], -1), -1)


class ExpSineAndLinearFeatures(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.lin = nn.Linear(3, 1, bias=False)
        # self.exp_lin = max_spectral_norm(nn.Linear(in_features, out_features, bias=False), max_spectral_norm=1.5)
        self.exp_mult  = nn.Parameter(torch.zeros(1, out_features))
        # self.lin_mult = nn.Parameter(torch.zeros(1, out_features))
        self.sine_mult = nn.Parameter(torch.ones(1, out_features))
        # self.lin = nn.Linear(4 * in_features, out_features, bias=False)
        self.sin = get_activation('sin')()
        self.cos = get_activation('cos')()
        self.exp = get_activation('exp')()

    def forward(self, x_feat):
        # x_feat: (B, X, out_features)
        exp = self.exp(self.exp_mult * x_feat)
        # lin = self.lin_mult * x_feat
        presine = self.sine_mult * x_feat
        c = 1
        sin = self.sin(c * presine)
        cos = self.cos(c * presine)
        # return torch.cat([0 * exp, 0 * lin, sin, cos], -1)
        return self.lin(torch.stack([exp, sin, cos], -1)).squeeze(-1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, mult):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.mult = mult
        self.sin = get_activation('sin')()
        self.cos = get_activation('cos')()

    def forward(self, x_feat):
        # x_feat: (B, X, out_features)
        presine = self.mult * self.lin(x_feat)
        sin = self.sin(presine)
        cos = self.cos(presine)
        return torch.cat([sin, cos], -1)



class ExpSineAndLinear(nn.Module):
    def __init__(self, in_features, out_features, mults=None):
        super().__init__()
        # self.lin = nn.Linear(len(activations) * in_features, out_features)
        self.lins = nn.ModuleList([nn.Linear(in_features, out_features),
                                   max_spectral_norm(nn.Linear(in_features, out_features), max_spectral_norm=0.95)])
        # with torch.no_grad():
        #     self.lins[1].weight.data.zero_()
        self.acts = [get_activation(act)() for act in ['smallsin', 'exp']]
        self.mults = mults
        if mults is None:
            self.mults = (1,) * len(self.acts)

    def forward(self, x):
        # return self.lin(torch.cat([mult * act(x) for mult, act in zip(self.mults, self.acts)], -1))
        return torch.mean(torch.stack([lin(mult * act(x))
                                       for mult, act, lin in zip(self.mults, self.acts, self.lins)],
                                      -1), -1)