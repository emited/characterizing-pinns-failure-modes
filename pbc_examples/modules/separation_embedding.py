import math
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import init

from pbc_examples.net_pbc import SymmetricInitDNN, get_activation


class Block(torch.nn.Module):
    def __init__(self, is_first, emb_dim, hidden_dim, activation, num_args=3, last_weight_is_zero_init=False, first_emb_dim=-1,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.activation = activation
        self.is_first = is_first

        self.lin_emb = nn.Linear(self.emb_dim, 2 * self.hidden_dim)
        self.lin = nn.Linear(self.hidden_dim, self.hidden_dim)
        if not self.is_first:
            self.prelin = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.prelin = nn.Linear(first_emb_dim, self.hidden_dim)
        # if last_weight_is_zero_init:
        #     with torch.no_grad():
        #         self.lin.weight.data.zero_()

        self.activation = get_activation(activation)()

    def forward(self, h, x):
        h = self.prelin(h)
        params = self.lin_emb(x)
        scale, shift = params[..., :self.hidden_dim], params[..., self.hidden_dim:]
        # self.lin.weight = self.lin.weight / self.lin.weight.sum(1, keepdim=True)
        preact = self.lin(h * scale + shift)
        # preact = h * scale + shift
        act = self.activation(preact)
        return act


class SeparationEmbedding(torch.nn.Module):
    """Add central embedding to the main network"""
    def __init__(self, num_samples):
        super(SeparationEmbedding, self).__init__()
        self.latent_dim = 2
        num_blocks = 6
        num_xt_blocks = 6
        hidden_dim = 256

        # bias_before = True
        # last_weight_is_zero_init = True
        # self.d = lambda x: torch.sum(x, -1, keepdim=True)
        # self.d = nn.Linear(hidden_dim, 1)
        # if last_weight_is_zero_init:
        #     with torch.no_grad():
        #         self.d.weight.data.zero_()
        self.d = SymmetricInitDNN([hidden_dim, 1], "identity")
        # self.d = SymmetricInitDNN([hidden_dim, hidden_dim, 1], "lrelu")
        # self.z = DNN([1, hidden_dim, hidden_dim, hidden_dim, self.latent_dim], 'sin',
        #              bias_before=bias_before, last_weight_is_zero_init=True)
        # self.p = DNN([1, hidden_dim, hidden_dim,  hidden_dim, self.latent_dim], 'sin',
        #              bias_before=bias_before, last_weight_is_zero_init=True)
        # self.z = DNN([1, hidden_dim,  self.latent_dim], 'lrelu',
        #              bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
        # self.p = DNN([1, hidden_dim, self.latent_dim], 'lrelu',
        #              bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
        # emb_dim = self.latent_dim
        emb_dim = self.latent_dim
        # self.h0 = torch.nn.Parameter(torch.randn(1, hidden_dim))
        # self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))

        self.h0 = torch.nn.Parameter(torch.zeros(1, emb_dim))

        # with torch.no_grad():
        #     self.e.weight.data.zero_()
        # self.e2l = DNN([emb_dim+1, hidden_dim, hidden_dim, hidden_dim,  self.latent_dim], 'sin',
        #              bias_before=bias_before, last_weight_is_zero_init=True)
        arg_dim = 2 * self.latent_dim
        self.blocks = nn.ModuleList(
            [Block(i == 0, arg_dim, hidden_dim, 'sin', 1, first_emb_dim=emb_dim) for i in range(num_blocks)])

        # self.hh0x = torch.nn.Parameter(torch.randn(1, hidden_dim))
        self.ex = torch.nn.Embedding(num_samples, emb_dim)
        self.ex.weight.data.zero_()

        # with torch.no_grad():
        #     xbound = 1 / emb_dim
        #     # xbound = bound / np.sqrt(T)
        #     xbias = torch.empty((emb_dim,))
        #     init.uniform_(xbias, -xbound, xbound)
        #     self.ex.weight.data = xbias.unsqueeze(0).repeat(self.ex.weight.data.shape[0], 1)

        self.e2lsx = nn.ModuleList([Block(i == 0, emb_dim + 1, hidden_dim, 'sin', 1,
                                          first_emb_dim=emb_dim) for i in range(num_xt_blocks)])
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

        # self.hh0t = torch.nn.Parameter(torch.randn(1, hidden_dim))
        self.et = torch.nn.Embedding(num_samples, emb_dim)
        self.et.weight.data.zero_()
        # with torch.no_grad():
        #     tbound = 1 / emb_dim
        #     # tbound = bound / np.sqrt(L)
        #     tbias = torch.empty((emb_dim,))
        #     init.uniform_(tbias, -tbound, tbound)
        #     self.et.weight.data = tbias.unsqueeze(0).repeat(self.et.weight.data.shape[0], 1)

        self.e2lst = nn.ModuleList([Block(i == 0, emb_dim + 1, hidden_dim, 'sin', 1,
                                          first_emb_dim=emb_dim) for i in range(num_xt_blocks)])
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
            # init.normal_(self.llt.weight, 0, xbound * 1)
        # d = 3
        # self.extra_lins_zt = nn.Sequential(*[nn.Linear(self.latent_dim, self.latent_dim) for i in range(d)])
        # self.extra_lins_px = nn.Sequential(*[nn.Linear(self.latent_dim, self.latent_dim) for i in range(d)])

        self.zt = None

    def forward(self, x, t, sample_index, ex=None, et=None):
        if ex is None:
            ex = self.ex(sample_index)
        if et is None:
            et = self.et(sample_index)
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)
        ex_broadcasted = ex.unsqueeze(1).expand(-1, x.shape[1], -1)
        hhx = ex_broadcasted
        for b in self.e2lsx:
            hhx = b(hhx, torch.cat([ex_broadcasted, x], -1))
        px = self.llx(hhx)
        # px = self.extra_lins_px(px)

        et_broadcasted = et.unsqueeze(1).expand(-1, t.shape[1], -1)
        hht = et_broadcasted
        for b in self.e2lst:
            hht = b(hht, torch.cat([et_broadcasted, t], -1))
        zt = self.llt(hht)
        # zt = self.extra_lins_px(zt)

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
                'ex': ex, 'et': et,
                }