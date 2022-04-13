import math
import torch
from torch import nn
from torch.nn import init

from pbc_examples.modules.separation_embedding import Block
from pbc_examples.net_pbc import SymmetricInitDNN


class ParamModule:
    pass


class SeparationParam(torch.nn.Module, ParamModule):
    """Add central embedding to the main network"""
    def __init__(self, param_dim):
        super(SeparationParam, self).__init__()
        self.latent_dim = 2
        num_blocks = 4
        num_xt_blocks = 2
        hidden_dim = 256
        emb_dim = self.latent_dim

        self.d = SymmetricInitDNN([hidden_dim, 1], "identity")
        self.h0 = torch.nn.Parameter(torch.zeros(1, emb_dim))
        self.blocks = nn.ModuleList(
            [Block(i == 0, 2 * self.latent_dim, hidden_dim, 'sin', 1, first_emb_dim=emb_dim) for i in range(num_blocks)])
        self.e2lsx = nn.ModuleList([Block(i == 0, param_dim + 1, hidden_dim, 'sin', 1,
                                          first_emb_dim=param_dim) for i in range(num_xt_blocks)])
        self.llx = nn.Linear(hidden_dim, self.latent_dim)
        # self.llx.weight.data.zero_()
        # self.llx.bias.data.zero_()
        with torch.no_grad():
            xbound = 1 / self.latent_dim
            xbound = xbound / math.sqrt(100)
            xbias = torch.empty((self.latent_dim,))
            init.uniform_(xbias, -xbound, xbound)
            self.llx.bias.data = xbias
            # self.llx.bias.data.zero_()
            init.normal_(self.llx.weight, 0, xbound * 0.1)
            # init.normal_(self.llx.weight, 0, xbound * 1)

        self.e2lst = nn.ModuleList([Block(i == 0, param_dim + 1, hidden_dim, 'sin', 1,
                                          first_emb_dim=param_dim) for i in range(num_xt_blocks)])
        self.llt = nn.Linear(hidden_dim, self.latent_dim)
        # self.llt.weight.data.zero_()
        with torch.no_grad():
            tbound = 1 / self.latent_dim
            tbound = tbound / math.sqrt(100)
            tbias = torch.empty((self.latent_dim,))
            init.uniform_(tbias, -tbound, tbound)
            self.llt.bias.data = tbias
            # self.llt.bias.data.zero_()
            init.normal_(self.llt.weight, 0, xbound * 0.1)

        self.zt = None

    def forward(self, x, t, param, ):
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)
        ex_broadcasted = param.unsqueeze(1).expand(-1, x.shape[1], -1)
        hhx = ex_broadcasted
        for b in self.e2lsx:
            hhx = b(hhx, torch.cat([ex_broadcasted, x], -1))
        px = self.llx(hhx)

        et_broadcasted = param.unsqueeze(1).expand(-1, t.shape[1], -1)
        hht = et_broadcasted
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
                # 'ex': ex, 'et': et,
                }