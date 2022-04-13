import torch
from torch import nn

from pbc_examples.modules.features import ExpSineAndLinear
from pbc_examples.net_pbc import get_activation


class BlockMod(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, activation_name, mults=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.activation_name = activation_name
        self.lin = nn.Linear(self.hidden_dim, self.hidden_dim)

        if isinstance(self.activation_name, (list, tuple)):
            self.acts_and_lin2 = ExpSineAndLinear(self.hidden_dim, self.hidden_dim, mults)
        else:
            self.activation = get_activation(self.activation_name)()
            self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.lin_emb = nn.Linear(self.emb_dim, 3 * self.hidden_dim)
        # self.lin_emb.weight.data.zero_()
        # self.ling = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.prelin = nn.Linear(self.hidden_dim, self.hidden_dim)
        # if last_weight_is_zero_init:
        #     with torch.no_grad():
        #         self.lin.weight.data.zero_()


    def forward(self, h, x):
        params = self.lin_emb(x)
        scale, shift, mod = params[..., :self.hidden_dim],\
                            params[..., self.hidden_dim: 2 * self.hidden_dim],\
                            params[..., 2 * self.hidden_dim:]
        # self.lin.weight = self.lin.weight / self.lin.weight.sum(1, keepdim=True)
        # g = torch.sigmoid(self.ling(h))
        # preact = (1-g) * self.lin(h * (1 + scale) + shift) + g * h
        preact = self.lin(h * (1 + scale) + shift)
        if isinstance(self.activation_name, (tuple, list)):
            linact = self.acts_and_lin2(preact)
        else:
            act = self.activation(preact)
            linact = self.lin2(act)
        assert mod.shape == linact.shape
        # postact = torch.sigmoid(mod) * linact
        # postact = mod * mod * linact
        postact = torch.sigmoid(mod) * linact
        # postact = (1-g) * postact + g * h

        # postact = mod * linact
        return postact
