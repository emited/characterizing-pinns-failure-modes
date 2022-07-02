from collections import OrderedDict

import torch
from torch import nn

from pbc_examples.modules.coords_to_latents import PX
from pbc_examples.modules.features import ExpSineAndLinearFeatures
from pbc_examples.modules.separation_param import ParamModule
from pbc_examples.modules.separation_param_modulation_big import BlockMod
from pbc_examples.net_pbc import SymmetricInitDNN



class ParamLinear(nn.Linear):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        if bias is not None:
            output += bias.unsqueeze(-2)
        return output


class FactorizedMultiplicativeModulation(nn.Module):
    def __init__(self, in_features, out_features, in_mod_features, rank, bias=True,):
        super(FactorizedMultiplicativeModulation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = ParamLinear(in_features, out_features)
        # self.A = nn.Linear(in_mod_features, out_features * rank)

        self.mod_params = nn.Linear(in_mod_features, in_features * rank + out_features * (rank + bias))
        self.base_weight = nn.Parameter(torch.empty((out_features, in_features),))
        if bias:
            self.base_bias = nn.Parameter(torch.empty(out_features,))

        self.in_features = in_features
        self.rank = rank
        self.is_bias = bias

    def forward(self, input, mod):
        mparams = self.mod_params(mod)

        mparam_cursor = 0
        if self.is_bias:
            # bias_mod = torch.sigmoid(mparams[..., [0]])
            bias_mod = mparams[..., :self.out_features]
            mparam_cursor += self.out_features
        else:
            bias_mod = None
        A_size = self.out_features * self.rank
        A = mparams[..., mparam_cursor: mparam_cursor + A_size]\
            .reshape((-1, self.out_features, self.rank))
        mparam_cursor += A_size
        Bt = mparams[..., mparam_cursor:]\
            .reshape((-1, self.rank, self.in_features))
        weight_mod = torch.sigmoid(torch.bmm(A, Bt))

        params = {}
        params['weight'] = weight_mod * self.base_weight.unsqueeze(0)
        # params['bias'] = bias_mod  * self.base_bias
        params['bias'] = bias_mod # as in FFMs, no base bias
        output = self.lin(input, params)
        return output



class SeparationParamSimpleLatent(torch.nn.Module, ParamModule):
    """Multiplying activations by space-time dependant scalars"""
    def __init__(self, param_dim=None, x_param_dim=None, t_param_dim=None, separate_params=False):
        super(SeparationParamSimpleLatent, self).__init__()
        self.separate_params = separate_params
        self.param_dim = param_dim
        self.x_param_dim = x_param_dim
        self.t_param_dim = t_param_dim
        self.latent_dim = 32
        emb_dim = 128
        num_blocks = 1
        num_xt_blocks = 4
        self.dh = 32
        hidden_dim = 128

        if not self.separate_params:
            self.p2e = SymmetricInitDNN([param_dim, hidden_dim, emb_dim, ], 'sin',)
            self.e2ex = nn.Linear(emb_dim, emb_dim)
            self.e2et = nn.Linear(emb_dim, emb_dim)
        else:
            self.xp2ex = SymmetricInitDNN([x_param_dim, hidden_dim, emb_dim, ], 'sin',)
            self.tp2et = SymmetricInitDNN([t_param_dim, hidden_dim, emb_dim, ], 'sin',)

        # self.lint = nn.Linear(emb_dim, self.latent_dim, bias=False)
        # self.linx = nn.Linear(emb_dim, self.latent_dim, bias=True)
        self.px = PX(emb_dim, self.latent_dim * self.dh, learn_mult=True)
        self.zt = PX(emb_dim, self.latent_dim * self.dh, learn_mult=True)
        # self.zt = ExpSineAndLinearFeatures(emb_dim, self.latent_dim)

        self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
        self.blocks = nn.ModuleList(
            [BlockMod(self.dh, hidden_dim, 'sin',)
            # [BlockMod(self.latent_dim, hidden_dim, 'sin',)
            # [BlockMod(1, hidden_dim, ['sin', 'exp'], [1, 1])
             for _ in range(num_blocks)])
        self.d = SymmetricInitDNN([hidden_dim, 1], "identity")

    def forward(self, x, t, param, ):
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)

        # (B, emb_dim) -> (B, X, emb_dim)
        if self.separate_params:
            xparam, tparam = param['x_params'], param['t_params']
        else:
            xparam = tparam = param

        # hparam = hparam * hparam
        if not self.separate_params:
            e = self.p2e(param)
            ex = self.e2ex(e)
            et = self.e2et(e)
        else:
            ex = self.xp2ex(xparam)
            et = self.tp2et(tparam)

        B, X = x.shape[:2]
        T = t.shape[1]

        px = self.px(x, ex)
        px = px.reshape(B, X, self.dh, self.latent_dim)

        zt = self.zt(t, et)
        zt = zt.reshape(B, T, self.dh, self.latent_dim)

        # zt: (B, T, dh, latent_dim) -> (B, T, X, dh, latent_dim)
        zt_broadcasted = zt.unsqueeze(2).expand(-1, -1, px.shape[1], -1, -1)
        # px: (B, X, latent_dim) -> (B, T, X, latent_dim)
        px_broadcasted = px.unsqueeze(1).expand(-1, zt.shape[1], -1, -1, -1)

        h0_repeated = self.h0.unsqueeze(1).unsqueeze(2) \
            .expand(*zt_broadcasted.shape[:-2], self.h0.shape[1])
        h = h0_repeated
        ztpx = torch.mean(zt_broadcasted * px_broadcasted, -1)
        for b in self.blocks:
            # ztpx = zt_broadcasted * px_broadcasted
            # ztpx = torch.cat([zt_broadcasted, px_broadcasted], -1)
            h = b(h, ztpx)

        u_pred = self.d(h)

        return {'u_pred': u_pred,
                'zt': zt.squeeze(2), 'px': px.squeeze(2),
                }

