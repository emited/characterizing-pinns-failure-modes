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


class SeparationParamSimpleLatentUn(torch.nn.Module, ParamModule):
    """Multiplying activations by space-time dependant scalars"""
    def __init__(self, param_dim=None, x_param_dim=None, t_param_dim=None, separate_params=False):
        super(SeparationParamSimpleLatentUn, self).__init__()
        self.separate_params = separate_params
        assert self.separate_params
        self.param_dim = param_dim
        self.x_param_dim = x_param_dim
        self.t_param_dim = t_param_dim
        self.latent_dim = 128
        emb_dim = 128
        num_blocks = 1
        num_xt_blocks = 4
        hidden_dim = 128

        # self.xp2ex = SymmetricInitDNN([x_param_dim, hidden_dim, hidden_dim, self.latent_dim, ], 'sin',)
        self.xp2ex = DNN([x_param_dim, hidden_dim, self.latent_dim, ], 'sin',)
        self.xp2ex_bias = DNN([x_param_dim,hidden_dim,  self.latent_dim, ], 'sin',)
        self.xp2ex_omega = DNN([x_param_dim,hidden_dim,  self.latent_dim, ], 'sin',)
        self.tp2et = SymmetricInitDNN([t_param_dim, emb_dim, ], 'sin')
        # self.tp2et = SymmetricInitDNN([t_param_dim, hidden_dim, hidden_dim, emb_dim, ], 'sin',)



        # self.lint = nn.Linear(emb_dim, self.latent_dim, bias=False)
        # self.linx = nn.Linear(emb_dim, self.latent_dim, bias=True)
        # self.px = DNN([emb_dim + 1, hidden_dim, hidden_dim, self.latent_dim], 'sin')
        # self.px = DNN([emb_dim + 1, hidden_dim, hidden_dim, self.latent_dim], 'sin')
        self.zt = SymmetricInitDNN([emb_dim + 1,  hidden_dim, hidden_dim, self.latent_dim], 'sin')
        self.ztp = SymmetricInitDNN([emb_dim + 1,  hidden_dim, hidden_dim, self.latent_dim], 'sin')
        self.px = SymmetricInitDNN([emb_dim + 1,  hidden_dim, hidden_dim, self.latent_dim], 'sin')
        self.pxp = SymmetricInitDNN([emb_dim + 1,  hidden_dim, hidden_dim, self.latent_dim], 'sin')
        # self.zt = ExpSineAndLinearFeatures(emb_dim, self.latent_dim)

        self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.blocks = nn.ModuleList(
            [BlockMod(2 * self.latent_dim + emb_dim, hidden_dim, 'sin',)
            # [BlockMod(self.latent_dim, hidden_dim, 'sin',)
            # [BlockMod(1, hidden_dim, ['sin', 'exp'], [1, 1])
             for _ in range(num_blocks)])
        self.d = SymmetricInitDNN([hidden_dim, 1], "identity")

    def forward(self, x, t, param, ):
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)

        # (B, emb_dim) -> (B, X, emb_dim)
        xparam, tparam = param['x_params'], param['t_params']

        ex = self.xp2ex(xparam)
        ex_bias = self.xp2ex_bias(xparam)
        ex_omega = self.xp2ex_omega(xparam)
        et = self.tp2et(tparam)

        # (B, emb_dim) -> (B, X, emb_dim)
        ex_broadcasted = ex.unsqueeze(1).unsqueeze(2).expand(-1, t.shape[1], x.shape[1], -1)
        # (B, emb_dim) -> (B, T, emb_dim)
        et_broadcasted = et.unsqueeze(1).expand(-1, t.shape[1], -1)
        etx_broadcasted = et.unsqueeze(1).expand(-1, x.shape[1], -1)

        px = self.px(torch.cat([x, etx_broadcasted], -1))
        # pxp = self.pxp(torch.cat([x, etx_broadcasted], -1))
        zt = self.zt(torch.cat([t, et_broadcasted], -1))
        # ztp = self.ztp(torch.cat([t, et_broadcasted], -1))
        # px = px * px - pxp * pxp
        # zt = zt * zt - ztp * ztp


        # zt: (B, T, dh, latent_dim) -> (B, T, X, dh, latent_dim)
        zt_broadcasted = zt.unsqueeze(2).expand(-1, -1, px.shape[1], -1)
        # px: (B, X, latent_dim) -> (B, T, X, latent_dim)
        px_broadcasted = px.unsqueeze(1).expand(-1, zt.shape[1], -1, -1)
        # ex_broadcasted = px.unsqueeze(1).expand(-1, zt.shape[1], -1, -1)
        h0_repeated = self.h0.unsqueeze(1).unsqueeze(2)\
            .expand(*zt_broadcasted.shape[:-1], self.h0.shape[1])
        h = h0_repeated
        # ztpx = torch.mean(zt_broadcasted * px_broadcasted, -1)
        # for b in self.blocks:
        #     ztpx = zt_broadcasted * px_broadcasted
            # ztpx = torch.cat([zt_broadcasted, px_broadcasted, ex_broadcasted], -1)
            # h = b(h, ztpx)

        # h = torch.sin((torch.sigmoid(ex_broadcasted) - 0.5) * (zt_broadcasted + px_broadcasted))
        # u_pred = torch.mean(torch.sin((torch.sigmoid(ex_broadcasted) - 0.5) * (zt_broadcasted + px_broadcasted)), -1, keepdims=True)
        # u_pred = torch.sum(ex.unsqueeze(1).unsqueeze(1) *
        #                    torch.sin(ex_omega.unsqueeze(1).unsqueeze(1) * ((zt_broadcasted + px_broadcasted)
        #                              + ex_bias.unsqueeze(1).unsqueeze(1))), -1, keepdims=True)
        u_pred = torch.sum(0.1 * ex.unsqueeze(1).unsqueeze(1) *
                           torch.sin(zt_broadcasted + px_broadcasted), -1, keepdims=True)
        # u_pred = torch.mean(t orch.cos(10 * (zt_broadcasted + px_broadcasted)), -1, keepdim=True)

        # u_pred = self.d(h)

        return {'u_pred': u_pred,
                'zt': zt.squeeze(2), 'px': px.squeeze(2),
                }