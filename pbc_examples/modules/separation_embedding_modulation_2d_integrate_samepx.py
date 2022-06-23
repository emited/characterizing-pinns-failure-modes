import math

import torch
from torch import nn

from pbc_examples.modules.separation_embedding import EmbeddingModule
from pbc_examples.modules.separation_param_modulation_big import BlockMod
from pbc_examples.net_pbc import SymmetricInitDNN, EulerNet, DNN


class SeparationEmbeddingMod2dIntegrateSamePX(torch.nn.Module, EmbeddingModule):
    """Add central embedding to the main network"""
    def __init__(self, num_samples):
        super(SeparationEmbeddingMod2dIntegrateSamePX, self).__init__()
        # self.latent_dim = 128
        # num_blocks = 4
        # num_xt_blocks = 4
        # hidden_dim = 128
        # hidden_dim_sep = 128
        # emb_dim = 128
        # emb_dim = self.latent_dim
        self.latent_dim = 2
        num_blocks = 4
        num_xt_blocks = 4
        hidden_dim = 128
        hidden_dim_sep = self.latent_dim
        emb_dim = 6
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
        arg_dim = 3 * self.latent_dim + emb_dim
        self.blocks = nn.ModuleList(
            [BlockMod(arg_dim, hidden_dim, 'sin',) for i in range(num_blocks)])

        self.hh0x = torch.nn.Parameter(torch.randn(1, hidden_dim_sep))
        self.ex = torch.nn.Embedding(2, emb_dim)
        self.ey = torch.nn.Embedding(2, emb_dim)
        # self.ex.weight.data.zero_()
        # with torch.no_grad():
        #     self.ex.weight.data = torch.randn(self.ex.weight.shape) / math.sqrt(emb_dim)
        #     self.ey.weight.data = torch.randn(self.ey.weight.shape) / math.sqrt(emb_dim)
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

        # self.emb_nn_t = DNN([emb_dim, hidden_dim, hidden_dim, self.latent_dim], "sin")
        # self.emb_nn_t0 = DNN([emb_dim, hidden_dim, hidden_dim, self.latent_dim], "sin")
        # self.emb_nn_x = DNN([emb_dim, hidden_dim, hidden_dim, self.latent_dim], "sin")
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

        # self.hh0t = torch.nn.Parameter(torch.randn(1, hidden_dim_sep))
        self.et = torch.nn.Embedding(num_samples, emb_dim)
        self.et0 = torch.nn.Embedding(num_samples, hidden_dim_sep)
        self.e0 = torch.nn.Embedding(num_samples, hidden_dim_sep)
        # self.et.weight.data.zero_()
        # with torch.no_grad():
        #     self.et.weight.data = torch.randn(self.et.weight.shape) / math.sqrt(emb_dim)
        #     self.et0.weight.data = torch.randn(self.et0.weight.shape) / math.sqrt(emb_dim)
        # with torch.no_grad():
        #     tbound = 1 / emb_dim
        #     # tbound = bound / np.sqrt(L)
        #     tbias = torch.empty((emb_dim,))
        #     init.uniform_(tbias, -tbound, tbound)
        #     self.et.weight.data = tbias.unsqueeze(0).repeat(self.et.weight.data.shape[0], 1)

        # self.e2lst = nn.ModuleList([BlockMod(emb_dim + 1, hidden_dim_sep, ['sin', 'exp'],) for i in range(num_xt_blocks)])
        # self.e2lst = nn.ModuleList([BlockMod(emb_dim + 1, hidden_dim_sep, 'sin',) for i in range(num_xt_blocks)])
        # self.llt = nn.Linear(hidden_dim_sep, self.latent_dim)
        # self.llt = SymmetricInitDNN([hidden_dim_sep, self.latent_dim], "identity")
        self.zt_func = EulerNet(self.latent_dim, hidden_dim, use_aux=False, steps=1)
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


    def get_ex(self, sample_index):
        return self.ex(sample_index)

    def get_ey(self, sample_index):
        return self.ey(sample_index)

    def get_et(self, sample_index):
        return self.et(sample_index)

    def forward(self, x, y, t, sample_index=None, ex=None, ey=None, et=None):
        if ex is None:
            # ex = self.emb_nn_x(self.ex(sample_index))
            ex = self.ex(sample_index * 0 + 1)
            # ey = self.emb_nn_x(self.ey(sample_index))
            ey = self.ey(sample_index * 0)
            e0 = self.e0(sample_index)
        if et is None:
            # et = self.emb_nn_t(self.et(sample_index))
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

        ey_broadcasted = ey.unsqueeze(1).expand(-1, y.shape[1], -1)
        hhy = self.hh0x.unsqueeze(1).expand(-1, x.shape[1], -1)
        for b in self.e2lsx:
            hhy = b(hhy, torch.cat([ey_broadcasted, y], -1))
        # hhx = hhx * hhx
        py = self.llx(hhy)


        # et_broadcasted = et.unsqueeze(1).expand(-1, t.shape[1], -1)
        # hht = self.hh0t.unsqueeze(1).expand(-1, t.shape[1], -1)
        # for b in self.e2lst:
        #     hht = b(hht, torch.cat([et_broadcasted, t], -1))
        # zt = self.llt(hht)
        # et0 = self.emb_nn_t0(self.et0(sample_index))
        et0 = self.et0(sample_index)
        zt = self.zt_func(t, x0=et0)
        # zt = zt * zt
        # zt = self.extra_lins_px(zt)
        # zt: (B, T, latent_dim) -> (B, T, Y, X, latent_dim)
        zt_broadcasted = zt.unsqueeze(2).unsqueeze(2).expand(-1, -1, py.shape[1], px.shape[1], -1)
        px_broadcasted = px.unsqueeze(1).unsqueeze(1).expand(-1, zt.shape[1], py.shape[1], -1, -1)
        py_broadcasted = py.unsqueeze(1).unsqueeze(3).expand(-1, zt.shape[1], -1, px.shape[1], -1)
        e0_broadcasted = e0.unsqueeze(1).unsqueeze(2).unsqueeze(3)\
            .expand(-1, zt.shape[1], py.shape[1], px.shape[1], e0.shape[1])
        h0_repeated = self.h0.unsqueeze(1).unsqueeze(2)\
            .expand(*zt_broadcasted.shape[:-1], self.h0.shape[1])
        print(self.h0.shape, h0_repeated.shape, e0.shape, zt_broadcasted.shape)
        h = h0_repeated
        for b in self.blocks:
            ztpx = torch.cat([zt_broadcasted, px_broadcasted, py_broadcasted, e0_broadcasted], -1)
            h = b(h, ztpx)
        u_pred = self.d(h)

        return {'u_pred': u_pred,
                'zt': zt, 'px': px, 'py': py,
                'ex': ex, 'et': et,
                }