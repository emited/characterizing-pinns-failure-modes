import torch
from torch import nn

from pbc_examples.modules.blockmod import BlockMod
from pbc_examples.modules.separation_param import ParamModule
from pbc_examples.net_pbc import SymmetricInitDNN



class SeparationParamMod(torch.nn.Module, ParamModule):
    """Multiplying activations by space-time dependant scalars"""
    def __init__(self, param_dim=None, x_param_dim=None, t_param_dim=None, separate_params=False):
        super(SeparationParamMod, self).__init__()
        self.separate_params = separate_params
        self.param_dim = param_dim
        self.x_param_dim = x_param_dim
        self.t_param_dim = t_param_dim
        self.latent_dim = 128
        emb_dim = 128
        num_blocks = 4
        num_xt_blocks = 4
        hidden_dim = 128

        if not self.separate_params:
            self.p2e = SymmetricInitDNN([param_dim, hidden_dim, emb_dim, ], 'sin',)
            self.e2ex = nn.Linear(emb_dim, emb_dim)
            self.e2et = nn.Linear(emb_dim, emb_dim)
        else:
            self.xp2ex = SymmetricInitDNN([x_param_dim, hidden_dim, emb_dim, ], 'sin',)
            self.tp2et = SymmetricInitDNN([t_param_dim, hidden_dim, emb_dim, ], 'sin',)

        self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.hx0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.ht0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.blocks = nn.ModuleList(
            [BlockMod(2 * self.latent_dim, hidden_dim, 'sin',)
            # [BlockMod(2 * self.latent_dim, hidden_dim, ['sin', 'exp'], [1, 1])
             for _ in range(num_blocks)])
        self.e2lsx = nn.ModuleList(
            # [BlockMod(hidden_dim + 1, hidden_dim, ['smallsin', 'exp'],)
            [BlockMod(emb_dim + 1, hidden_dim, 'sin',)
             for _ in range(num_xt_blocks)])
        # self.llx = nn.Linear(hidden_dim, self.latent_dim)
        self.llx = SymmetricInitDNN([hidden_dim, self.latent_dim], "identity")

        self.e2lst = nn.ModuleList(
            [BlockMod(emb_dim + 1, hidden_dim, 'sin',)
            # [BlockMod(hidden_dim + 1, hidden_dim, ['smallsin', 'exp'])
             for _ in range(num_xt_blocks)])
        # self.llt = nn.Linear(hidden_dim, self.latent_dim)
        # self.llt.weight.data.zero_()
        self.llt = SymmetricInitDNN([hidden_dim, self.latent_dim], "identity")
        self.d = SymmetricInitDNN([hidden_dim, 1], "identity")

        # self.x_param_group = [self.hx0]
        # self.x_param_group.extend(self.e2lsx.parameters())
        # self.x_param_group.extend(self.llx.parameters())
        # self.t_param_group = [self.ht0]
        # self.t_param_group.extend(self.e2lst.parameters())
        # self.t_param_group.extend(self.llt.parameters())
        # self.xt_param_group = [self.h0]
        # self.xt_param_group.extend(self.blocks.parameters())
        # self.xt_param_group.extend(self.d.parameters())

    def forward(self, x, t, param, ):
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)

        # (B, emb_dim) -> (B, X, emb_dim)
        # if self.separate_params:
        #     assert param.shape[1] == 2
        #     xparam = param[:, 0]
        #     tparam = param[:, 1]
        # else:
        #     xparam = tparam = param

        if self.separate_params:
            xparam, tparam = param['x_params'], param['t_params']
        else:
            xparam = tparam = param


        # hparam = hparam * hparam
        if not self.separate_params:
            e = self.p2e(param)
            ex = self.e2ex(e)
        else:
            ex = self.xp2ex(xparam)

        # ex = ex * ex

        ex_broadcasted = ex.unsqueeze(1).expand(-1, x.shape[1], -1)
        # (1, hidden_dim) -> (1, X, hidden_dim)
        hhx = self.hx0.unsqueeze(1).expand(-1, x.shape[1], -1)
        for b in self.e2lsx:
            hhx = b(hhx, torch.cat([ex_broadcasted, x], -1))
        px = self.llx(hhx)
        # px = px * px

        # (B, emb_dim) -> (B, T, emb_dim)
        if not self.separate_params:
            e = self.p2e(param)
            et = self.e2et(e)
        else:
            et = self.tp2et(tparam)
        # et = et * et

        # htparam = self.hp2ht(tparam)
        et_broadcasted = et.unsqueeze(1).expand(-1, t.shape[1], -1)
        hht = self.ht0.unsqueeze(1).expand(-1, t.shape[1], -1)
        for b in self.e2lst:
            hht = b(hht, torch.cat([et_broadcasted, t], -1))
        zt = self.llt(hht)
        # zt = zt * zt

        zt_broadcasted = zt.unsqueeze(2).expand(-1, -1, px.shape[1], -1)
        px_broadcasted = px.unsqueeze(1).expand(-1, zt.shape[1], -1, -1)

        h0_repeated = self.h0.unsqueeze(1).unsqueeze(2)\
            .expand(*zt_broadcasted.shape[:-1], self.h0.shape[1])
        h = h0_repeated
        for b in self.blocks:
            ztpx = torch.cat([zt_broadcasted, px_broadcasted], -1)
            h = b(h, ztpx)
        u_pred = self.d(h)

        return {'u_pred': u_pred,
                'zt': zt, 'px': px,
                }