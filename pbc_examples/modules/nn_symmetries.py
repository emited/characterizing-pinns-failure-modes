import math

import torch
from torch import nn

from pbc_examples.modules.modulated_linear import ModulatedLinear
from pbc_examples.modules.separation_embedding import Block
from pbc_examples.modules.separation_param import ParamModule
from pbc_examples.modules.separation_param_simple_latents import FactorizedMultiplicativeModulation as FMM
from pbc_examples.net_pbc import SymmetricInitDNN, DNN


class SymmetryNet(torch.nn.Module, ParamModule):
    """Add central embedding to the main network"""
    def __init__(self, coord_dim, param_dim):
        super(SymmetryNet, self).__init__()
        self.latent_dim = 128
        num_blocks = 4
        hidden_dim = 128
        # is_first, emb_dim, hidden_dim, activation, num_args = 3, last_weight_is_zero_init = False, first_emb_dim = -1,):
        self.blocks = nn.ModuleList(
            [Block(i == 0, self.latent_dim, hidden_dim, 'sin', 1, first_emb_dim=coord_dim)
             for i in range(num_blocks)])
        # self.style_net = DNN([param_dim, hidden_dim, hidden_dim], 'sin')
        self.style_net = DNN([param_dim, hidden_dim], 'relu')
        self.affines = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(num_blocks)])
        self.d = DNN([hidden_dim, 1], "identity")

    def forward(self, coords, param, ):
        w = self.style_net(param)
        h = coords
        for i, b in enumerate(self.blocks):
            s = self.affines[i](w)
            h = b(h, s)
        u_pred = 10 * self.d(h)

        return u_pred


class FMMSymmetryNet(nn.Module):
    """Add central embedding to the main network"""

    def __init__(self, coord_dim, param_dim, out_dim=1, max_rank=10):
        super(FMMSymmetryNet, self).__init__()
        self.latent_dim = 128
        self.max_rank = max_rank
        num_blocks = 4
        hidden_dim = 128

        mlins = []
        for i in range(num_blocks):
            if i == 0:
                in_dim = coord_dim
            else:
                in_dim = hidden_dim
            if i == num_blocks - 1:
                out_dim = out_dim
            else:
                out_dim = hidden_dim
            # in_features, out_features, in_mod_features, rank, bias = True
            mlin = FMM(in_dim, out_dim, self.latent_dim,
                                   rank=min(min(in_dim, out_dim), max_rank))
            mlins.append(mlin)
        self.mlins = nn.ModuleList(mlins)
        self.style_net = DNN([param_dim, hidden_dim, hidden_dim, hidden_dim], 'relu')

    def forward(self, coords, param, ):
        w = self.style_net(param)
        h = coords
        for i in range(len(self.mlins) - 1):
            h = self.mlins[i](h, w)
            h = torch.relu(h)
        h = self.mlins[-1](h)
        return h


class ModulatedSymmetryNet(nn.Module):
    """Add central embedding to the main network"""

    def __init__(self, coord_dim, z_dim, hidden_dim=128, out_dim=1, num_blocks=4, ):
        super(ModulatedSymmetryNet, self).__init__()
        self.z_dim = z_dim
        self.coord_dim = coord_dim

        mlins = []
        lins = []
        affines = []
        affines_bias = []
        for i in range(num_blocks):
            if i == 0:
                in_dim = coord_dim
            else:
                in_dim = hidden_dim
            if i == num_blocks - 1:
                outt_dim = out_dim
            else:
                outt_dim = hidden_dim
            # in_features, out_features, in_mod_features, rank, bias = True
            mlin = ModulatedLinear(in_dim, outt_dim, bias=True)
            # lin = nn.Linear(in_dim, outt_dim, bias=True)
            affine = nn.Linear(hidden_dim, in_dim, bias=True)
            affine_bias = nn.Linear(hidden_dim, outt_dim, bias=True)
            mlins.append(mlin)
            affines.append(affine)
            affines_bias.append(affine_bias)
            # lins.append(lin)
        # self.lins = nn.ModuleList(lins)
        self.mlins = nn.ModuleList(mlins)
        self.affines = nn.ModuleList(affines)
        self.affines_bias = nn.ModuleList(affines_bias)
        self.style_net = DNN([z_dim, hidden_dim, hidden_dim, hidden_dim], 'sin')
        # self.style_net = DNN([z_dim, hidden_dim], 'sin')

    def forward(self, coords, z, ):
        """
        coords: (B, N, coord_dim)
        z: (B, N, z_dim)
        """
        w = self.style_net(z)
        h = coords
        for i in range(len(self.mlins) - 1):
            # if i > 0:
            #     h = self.lins[i](h)
            # else:
            a = self.affines[i](w)
            a_bias = self.affines_bias[i](w)
            h = self.mlins[i](h, a, a_bias)
            h = torch.sin(h)
        a = self.affines[-1](w)
        a_bias = self.affines_bias[-1](w)
        h = self.mlins[-1](h, a, a_bias)
        return h, w


