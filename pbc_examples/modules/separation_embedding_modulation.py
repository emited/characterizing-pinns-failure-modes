import math

import torch
from torch import nn

from pbc_examples.modules.separation_embedding import EmbeddingModule
from pbc_examples.modules.separation_param_modulation_big import BlockMod
from pbc_examples.net_pbc import SymmetricInitDNN


class SeparationEmbeddingMod(torch.nn.Module, EmbeddingModule):
    """Add central embedding to the main network"""
    def __init__(self, num_samples):
        super(SeparationEmbeddingMod, self).__init__()
        self.latent_dim = 128
        num_blocks = 4
        num_xt_blocks = 4
        hidden_dim = 128
        hidden_dim_sep = 128
        emb_dim = 128
        # emb_dim = self.latent_dim

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
        # self.h0 = torch.nn.Parameter(torch.randn(1, hidden_dim))
        # self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))

        self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))

        # with torch.no_grad():
        #     self.e.weight.data.zero_()
        # self.e2l = DNN([emb_dim+1, hidden_dim, hidden_dim, hidden_dim,  self.latent_dim], 'sin',
        #              bias_before=bias_before, last_weight_is_zero_init=True)
        arg_dim = 2 * self.latent_dim
        self.blocks = nn.ModuleList(
            [BlockMod(arg_dim, hidden_dim, 'sin',) for i in range(num_blocks)])

        self.hh0x = torch.nn.Parameter(torch.randn(1, hidden_dim_sep))
        self.ex = torch.nn.Embedding(num_samples, emb_dim)
        # self.ex.weight.data.zero_()
        with torch.no_grad():
            self.ex.weight.data = torch.randn(self.ex.weight.shape) / math.sqrt(100)
        # with torch.no_grad():
        #     xbound = 1 / emb_dim
        #     # xbound = bound / np.sqrt(T)
        #     xbias = torch.empty((emb_dim,))
        #     init.uniform_(xbias, -xbound, xbound)
        #     self.ex.weight.data = xbias.unsqueeze(0).repeat(self.ex.weight.data.shape[0], 1)

        self.e2lsx = nn.ModuleList([BlockMod(emb_dim + 1, hidden_dim_sep, 'sin',
                                          ) for i in range(num_xt_blocks)])
        # self.llx = nn.Linear(hidden_dim_sep, self.latent_dim)
        self.llx = SymmetricInitDNN([hidden_dim_sep, self.latent_dim], "identity")

        # self.llx.weight.data.zero_()
        # self.llx.bias.data.zero_()
        # with torch.no_grad():
        #     xbound = 1 / self.latent_dim
        #     xbound = xbound / math.sqrt(100)
        #     xbias = torch.empty((self.latent_dim,))
        #     init.uniform_(xbias, -xbound, xbound)
        #     self.llx.bias.data = xbias
        #     # self.llx.bias.data.zero_()
        #     init.normal_(self.llx.weight, 0, xbound * 0.1)
        #     # init.normal_(self.llx.weight, 0, xbound * 1)

        self.hh0t = torch.nn.Parameter(torch.randn(1, hidden_dim_sep))
        self.et = torch.nn.Embedding(num_samples, emb_dim)
        # self.et.weight.data.zero_()
        with torch.no_grad():
            self.et.weight.data = torch.randn(self.et.weight.shape) / math.sqrt(100)
        # with torch.no_grad():
        #     tbound = 1 / emb_dim
        #     # tbound = bound / np.sqrt(L)
        #     tbias = torch.empty((emb_dim,))
        #     init.uniform_(tbias, -tbound, tbound)
        #     self.et.weight.data = tbias.unsqueeze(0).repeat(self.et.weight.data.shape[0], 1)

        # self.e2lst = nn.ModuleList([BlockMod(emb_dim + 1, hidden_dim_sep, ['sin', 'exp'],) for i in range(num_xt_blocks)])
        self.e2lst = nn.ModuleList([BlockMod(emb_dim + 1, hidden_dim_sep, 'sin',) for i in range(num_xt_blocks)])
        # self.llt = nn.Linear(hidden_dim_sep, self.latent_dim)
        self.llt = SymmetricInitDNN([hidden_dim_sep, self.latent_dim], "identity")
        # self.llt.weight.data.zero_()
        # with torch.no_grad():
        #     tbound = 1 / self.latent_dim
        #     tbound = tbound / math.sqrt(100)
        #     tbias = torch.empty((self.latent_dim,))
        #     init.uniform_(tbias, -tbound, tbound)
        #     self.llt.bias.data = tbias
        #     # self.llt.bias.data.zero_()
        #     init.normal_(self.llt.weight, 0, xbound * 0.1)
        #     # init.normal_(self.llt.weight, 0, xbound * 1)
        # d = 3
        # self.extra_lins_zt = nn.Sequential(*[nn.Linear(self.latent_dim, self.latent_dim) for i in range(d)])
        # self.extra_lins_px = nn.Sequential(*[nn.Linear(self.latent_dim, self.latent_dim) for i in range(d)])
        self.emb_param_group = list(self.ex.parameters())
        self.emb_param_group.extend(list(self.et.parameters()))
        # self.x_param_group = []
        # self.x_param_group.extend(self.e2lsx.parameters())
        # self.x_param_group.extend(self.llx.parameters())
        # self.t_param_group = [self.ht0]
        # self.t_param_group.extend(self.e2lst.parameters())
        # self.t_param_group.extend(self.llt.parameters())
        # self.xt_param_group = [self.h0]
        # self.xt_param_group.extend(self.blocks.parameters())
        # self.xt_param_group.extend(self.d.parameters())

    def get_ex(self, sample_index):
        return self.ex(sample_index)

    def get_et(self, sample_index):
        return self.et(sample_index)

    def forward(self, x, t, sample_index=None, ex=None, et=None):
        if ex is None:
            ex = self.ex(sample_index)
        if et is None:
            et = self.et(sample_index)
        # ex = ex ** 2
        # et = et ** 2
        # ex = ex * (torch.sigmoid(ex) - 0.5)
        # et = et * (torch.sigmoid(et) - 0.5)
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)
        ex_broadcasted = ex.unsqueeze(1).expand(-1, x.shape[1], -1)
        hhx = self.hh0x.unsqueeze(1).expand(-1, x.shape[1], -1)
        for b in self.e2lsx:
            hhx = b(hhx, torch.cat([ex_broadcasted, x], -1))
        # hhx = hhx * hhx
        px = self.llx(hhx)
        # px = px * px
        # px = self.extra_lins_px(px)

        et_broadcasted = et.unsqueeze(1).expand(-1, t.shape[1], -1)
        hht = self.hh0t.unsqueeze(1).expand(-1, t.shape[1], -1)
        for b in self.e2lst:
            hht = b(hht, torch.cat([et_broadcasted, t], -1))

        zt = self.llt(hht)
        # zt = zt * zt
        # zt = self.extra_lins_px(zt)

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
                'ex': ex, 'et': et,
                }