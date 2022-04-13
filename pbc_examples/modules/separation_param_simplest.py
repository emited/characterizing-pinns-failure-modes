from collections import OrderedDict

import torch
from torch import nn

from pbc_examples.modules.coords_to_latents import PX
from pbc_examples.modules.features import ExpSineAndLinearFeatures
from pbc_examples.modules.separation_param import ParamModule
from pbc_examples.modules.separation_param_modulation_big import BlockMod
from pbc_examples.net_pbc import SymmetricInitDNN, DNN


class PX(nn.Module):
    def __init__(self, emb_dim, out_features, learn_mult=False):
        super(PX, self).__init__()
        self.learn_mult = learn_mult
        self.feat = ExpSineAndLinearFeatures(out_features)
        self.lin_feat = nn.Linear(emb_dim, out_features)
        self.lin_feat_bias = nn.Linear(emb_dim, out_features)
        if self.learn_mult:
            self.lin_u0 = nn.Linear(emb_dim, out_features)

    def forward(self, x, emb):
        # x : (B, X, 1)
        # emb : (B, emb_dim)

        # x_transformed : (B, X, out_features)
        x_transformed = x * self.lin_feat(emb).unsqueeze(-2)\
                        + self.lin_feat_bias(emb).unsqueeze(-2)
        if self.learn_mult:
            m = self.lin_u0(emb).unsqueeze(-2)
        else:
            m = 1
        return self.feat(x_transformed) * m




class SeparationParamSimplest(torch.nn.Module, ParamModule):
    """Multiplying activations by space-time dependant scalars"""
    def __init__(self, param_dim=None, x_param_dim=None, t_param_dim=None, separate_params=False):
        super(SeparationParamSimplest, self).__init__()
        self.separate_params = separate_params
        assert self.separate_params

        dim = 32
        self.param_dim = param_dim
        self.x_param_dim = x_param_dim
        self.t_param_dim = t_param_dim
        self.latent_dim = dim
        emb_dim = dim
        hidden_dim = dim
        num_blocks = 6

        # self.xp2ex = DNN([x_param_dim, hidden_dim, self.latent_dim, ], 'lrelu',)
        self.xp2ex = DNN([x_param_dim,  hidden_dim, emb_dim, ], 'sin',)
        # self.xp2exd = DNN([x_param_dim, hidden_dim, self.latent_dim, ], 'sin',)
        # self.tp2et = DNN([t_param_dim, hidden_dim,  emb_dim, ], 'lrelu')
        self.tp2et = DNN([t_param_dim,  hidden_dim, emb_dim, ], 'sin')
        # self.p2e = SymmetricInitDNN([t_param_dim + x_param_dim, hidden_dim, hidden_dim, hidden_dim, emb_dim, ], 'sin')
        # self.p2e = DNN([t_param_dim + x_param_dim,  hidden_dim, emb_dim, ], 'sin')
        # self.d = DNN([hidden_dim, hidden_dim, hidden_dim], "sin")
        self.phi = DNN([hidden_dim, hidden_dim, hidden_dim,  hidden_dim], "sin")
        # self.phi = DNN([2 * self.latent_dim + emb_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim], "sin")
        # self.f = DNN([2 + emb_dim, hidden_dim,], "sin")

        # self.hx = DNN([1, hidden_dim, hidden_dim, hidden_dim, hidden_dim], "sin")
        # self.zt = DNN([1, hidden_dim, hidden_dim, hidden_dim, hidden_dim], "sin")


        self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        # self.hx0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        # self.ht0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.fblocks = nn.ModuleList(
            [BlockMod(2 + emb_dim, hidden_dim, 'sin',)
            # [BlockMod(2 * self.latent_dim, hidden_dim, ['sin', 'exp'], [1, 1])
             for _ in range(num_blocks)])

        self.gh0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.gblocks = nn.ModuleList(
            [BlockMod(hidden_dim + emb_dim, hidden_dim,  'sin',)
            # [BlockMod(2 * self.latent_dim, hidden_dim, ['sin', 'exp'], [1, 1])
             for _ in range(num_blocks)])

        self.d = SymmetricInitDNN([hidden_dim, 1], "identity")

    def forward(self, x, t, param, ):
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)

        # (B, emb_dim) -> (B, X, emb_dim)
        xparam, tparam = param['x_params'], param['t_params']
        # e = self.p2e(torch.cat([xparam, tparam], -1))
        ex = self.xp2ex(xparam)
        # exb = self.xp2exd(xparam)
        et = self.tp2et(tparam)

        tr = t.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        xr = x.unsqueeze(1).expand(-1, t.shape[1], -1, -1)
        # h = self.d(torch.cat([xr, tr], -1))
        # hx = self.hx(xr)
        # zt = self.zt(tr)
        etu = et.unsqueeze(1).unsqueeze(2).expand(-1, t.shape[1], x.shape[1], -1)
        # eu = e.unsqueeze(1).unsqueeze(2).expand(-1, t.shape[1], x.shape[1], -1)

        # zt_broadcasted = zt.unsqueeze(2).expand(-1, -1, px.shape[1], -1)
        # px_broadcasted = px.unsqueeze(1).expand(-1, zt.shape[1], -1, -1)

        h0_repeated = self.h0.unsqueeze(1).unsqueeze(2).expand(*tr.shape[:-1], self.h0.shape[1])
        h = h0_repeated
        for b in self.fblocks:
            ztpx = torch.cat([xr, tr, etu], -1)
            h = b(h, ztpx)

        # f = self.f(torch.cat([xr, tr, etu], -1))
        # f = self.f(torch.cat([hx, zt, etu], -1))

        # exu = ex.unsqueeze(1).unsqueeze(1)
        # exbu = exb.unsqueeze(1).unsqueeze(1)
        # etu = et.unsqueeze(1).unsqueeze(1)
        # u_pred = torch.mean(torch.sigmoid(exu)  * torch.sigmoid(etu) * h, -1, keepdim=True)
        # u_pred = torch.mean( torch.sigmoid(exd.unsqueeze(1).unsqueeze(1))
        #                      * torch.sin(hx*ex.unsqueeze(1).unsqueeze(1) + et.unsqueeze(1).unsqueeze(1)*zt), -1, keepdim=True)
        # u_pred = torch.mean(exu * self.phi(h) + exbu, -1, keepdim=True)

        gh0_repeated = self.gh0.unsqueeze(1).unsqueeze(2).expand(*tr.shape[:-1], self.h0.shape[1])
        gh = gh0_repeated
        ex_broadcasted = ex.unsqueeze(1).unsqueeze(2).expand(-1, t.shape[1], x.shape[1], -1)
        # e_broadcasted = e.unsqueeze(1).unsqueeze(2).expand(-1, t.shape[1], x.shape[1], -1)
        for b in self.gblocks:
            gh = b(gh, torch.cat([h, ex_broadcasted], -1))

        u_pred = self.d(gh)

        return {'u_pred': u_pred,
                }



class SeparationParamSimplestOld(torch.nn.Module, ParamModule):
    """Multiplying activations by space-time dependant scalars"""
    def __init__(self, param_dim=None, x_param_dim=None, t_param_dim=None, separate_params=False):
        super(SeparationParamSimplestOld, self).__init__()
        self.separate_params = separate_params
        assert self.separate_params
        self.param_dim = param_dim
        self.x_param_dim = x_param_dim
        self.t_param_dim = t_param_dim
        self.latent_dim = 128
        emb_dim = 128
        hidden_dim = 128

        self.xp2ex = DNN([x_param_dim, hidden_dim, self.latent_dim, ], 'sin',)
        self.xp2exd = DNN([x_param_dim, hidden_dim, self.latent_dim, ], 'sin',)
        self.tp2et = DNN([t_param_dim, hidden_dim,  emb_dim, ], 'sin')
        # self.p2e = SymmetricInitDNN([t_param_dim + x_param_dim, hidden_dim, hidden_dim, hidden_dim, emb_dim, ], 'sin')
        # self.d = DNN([2, hidden_dim, hidden_dim, hidden_dim, hidden_dim], "sin")
        self.phi = DNN([hidden_dim,  hidden_dim], "sin")
        # self.phi = DNN([2 * self.latent_dim + emb_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim], "sin")
        self.f = DNN([2 + emb_dim, hidden_dim,], "sin")

        self.hx = DNN([1, hidden_dim, hidden_dim, hidden_dim, hidden_dim], "sin")
        self.zt = DNN([1, hidden_dim, hidden_dim, hidden_dim, hidden_dim], "sin")

    def forward(self, x, t, param, ):
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)

        # (B, emb_dim) -> (B, X, emb_dim)
        xparam, tparam = param['x_params'], param['t_params']
        # e = self.p2e(torch.cat([xparam, tparam], -1))
        ex = self.xp2ex(xparam)
        exb = self.xp2exd(xparam)
        et = self.tp2et(tparam)

        tr = t.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        xr = x.unsqueeze(1).expand(-1, t.shape[1], -1, -1)
        # h = self.d(torch.cat([xr, tr], -1))
        # hx = self.hx(xr)
        # zt = self.zt(tr)
        etu = et.unsqueeze(1).unsqueeze(2).expand(-1, t.shape[1], x.shape[1], -1)
        f = self.f(torch.cat([xr, tr, etu], -1))
        # f = self.f(torch.cat([hx, zt, etu], -1))

        exu = ex.unsqueeze(1).unsqueeze(1)
        exbu = exb.unsqueeze(1).unsqueeze(1)
        # etu = et.unsqueeze(1).unsqueeze(1)
        # u_pred = torch.mean(torch.sigmoid(exu)  * torch.sigmoid(etu) * h, -1, keepdim=True)
        # u_pred = torch.mean( torch.sigmoid(exd.unsqueeze(1).unsqueeze(1))
        #                      * torch.sin(hx*ex.unsqueeze(1).unsqueeze(1) + et.unsqueeze(1).unsqueeze(1)*zt), -1, keepdim=True)
        u_pred = torch.mean(exu * self.phi(f) + exbu, -1, keepdim=True)

        return {'u_pred': u_pred,
                }
