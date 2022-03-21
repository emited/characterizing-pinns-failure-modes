import math
import torch
from torch import nn, Tensor
from torch.nn import init

from pbc_examples.net_pbc import SymmetricInitDNN, get_activation


class LowRankLinear():
    def __init__(self):
        pass
    def forward(self, x):
        pass

class BlockMod(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.activation = activation


        self.lin_emb = nn.Linear(self.emb_dim, 3 * self.hidden_dim)
        self.lin = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.prelin = nn.Linear(self.hidden_dim, self.hidden_dim)
        # if last_weight_is_zero_init:
        #     with torch.no_grad():
        #         self.lin.weight.data.zero_()

        self.activation = get_activation(activation)()

    def forward(self, h, x):
        params = self.lin_emb(x)
        scale, shift, mod = params[..., :self.hidden_dim],\
                            params[..., self.hidden_dim: 2 * self.hidden_dim],\
                            params[..., 2 * self.hidden_dim:]
        # self.lin.weight = self.lin.weight / self.lin.weight.sum(1, keepdim=True)
        preact = self.lin(h * torch.sigmoid(scale) + shift)
        act = self.activation(preact)
        linact = self.lin(act)
        assert mod.shape == linact.shape
        postact = torch.sigmoid(mod) * linact
        return postact


class SeparationParamMod(torch.nn.Module):
    """Multiplying activations by space-time dependant scalars"""
    def __init__(self, param_dim):
        super(SeparationParamMod, self).__init__()
        self.latent_dim = 2

        num_blocks = 6
        num_xt_blocks = 6
        hidden_dim = 64

        self.d = SymmetricInitDNN([hidden_dim, 1], "identity")
        self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.hx0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.ht0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.blocks = nn.ModuleList(
            [BlockMod(2 * self.latent_dim, hidden_dim, 'sin',)
             for _ in range(num_blocks)])
        self.e2lsx = nn.ModuleList(
            [BlockMod(param_dim + 1, hidden_dim, 'sin',)
             for _ in range(num_xt_blocks)])
        # self.llx = nn.Linear(hidden_dim, self.latent_dim)
        self.llx = SymmetricInitDNN([hidden_dim, self.latent_dim], "identity")

        self.e2lst = nn.ModuleList(
            [BlockMod(param_dim + 1, hidden_dim, 'sin')
             for _ in range(num_xt_blocks)])
        # self.llt = nn.Linear(hidden_dim, self.latent_dim)
        # self.llt.weight.data.zero_()
        self.llt = SymmetricInitDNN([hidden_dim, self.latent_dim], "identity")
        self.zt = None

    def forward(self, x, t, param, ):
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)

        # (B, emb_dim) -> (B, X, emb_dim)
        ex_broadcasted = param.unsqueeze(1).expand(-1, x.shape[1], -1)
        # (1, hidden_dim) -> (1, X, hidden_dim)
        hhx = self.hx0.unsqueeze(1).expand(-1, x.shape[1], -1)
        for b in self.e2lsx:
            hhx = b(hhx, torch.cat([ex_broadcasted, x], -1))
        px = self.llx(hhx)

        # (B, emb_dim) -> (B, T, emb_dim)
        et_broadcasted = param.unsqueeze(1).expand(-1, t.shape[1], -1)
        hht = self.ht0.unsqueeze(1).expand(-1, t.shape[1], -1)
        for b in self.e2lst:
            hht = b(hht, torch.cat([et_broadcasted, t], -1))
        zt = self.llt(hht)

        zt_broadcasted = zt.unsqueeze(2).expand(-1, -1, px.shape[1], -1)
        px_broadcasted = px.unsqueeze(1).expand(-1, zt.shape[1], -1, -1)

        h0_repeated = self.h0.unsqueeze(1).unsqueeze(2).expand(*zt_broadcasted.shape[:-1], self.h0.shape[1])
        h = h0_repeated
        for b in self.blocks:
            ztpx = torch.cat([zt_broadcasted, px_broadcasted], -1)
            h = b(h, ztpx)
        u_pred = self.d(h)

        return {'u_pred': u_pred,
                'zt': zt, 'px': px,
                }