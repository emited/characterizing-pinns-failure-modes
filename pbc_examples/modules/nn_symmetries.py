import torch
from torch import nn

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

class ModulatedSymmetryNet(torch.nn.Module, ParamModule):
    """Add central embedding to the main network"""

    def __init__(self, coord_dim, param_dim, out_dim=1, max_rank=10):
        super(ModulatedSymmetryNet, self).__init__()
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


def param_linear(input, weight, bias=None):
    '''
    The difference with the other ParamLinear linear in the code is the fact that
    there are no weights that are saved in memory that are not used anyway
    '''
    # bias = params.get('bias', None)
    # weight = params['weight']
    output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
    if bias is not None:
        output += bias.unsqueeze(-2)
    return output


class ModulatedLinear(nn.Linear):
    """Fully connected version of stylegan2's modulated conv"""
    def __init__(self, in_features, out_features, bias=True):
        super(ModulatedLinear, self).__init__(in_features, out_features, bias=bias)
        self.eps = 1e-8

    def forward(self, input, style):
        """
        style: (B, I), weight: (O, I), input: (B, I), bias: (B, O)
        output: (B, O)
        """
        assert self.weight.shape[1] == style.shape[1]
        # mod_weight: (B, O, I)
        mod_weight = self.weight.unsqueeze(0) * style.unsqueeze(1)
        norm_weight = mod_weight.pow(2).sum(-1, keepdim=True)
        demod_weight = mod_weight / torch.sqrt(norm_weight + self.eps)
        output = param_linear(input, demod_weight, self.bias)
        return output