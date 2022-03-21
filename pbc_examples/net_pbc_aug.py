import torch.nn
from torch import nn
from torch.nn import init

from choose_optimizer import *

# CUDA support
from pbc_examples.net_pbc import PhysicsInformedNN_pbc, SeparationDNN, DNN, SymmetricInitDNN, get_activation

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('using cuda')
else:
    device = torch.device('cpu')

def hash(l):
    hashtable, cnter = {}, 0
    for li in l:
        if li not in hashtable:
            hashtable[li] = cnter
            cnter += 1
    return hashtable


class Block(torch.nn.Module):
    def __init__(self, is_first, embedding_dim, hidden_dim, activation, num_args=3, last_weight_is_zero_init=False, first_embedding_dim=-1,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.is_first = is_first

        self.lin_emb = torch.nn.Linear(self.embedding_dim, 2 * self.hidden_dim)
        self.lin = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        if not self.is_first:
            self.prelin = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.prelin = torch.nn.Linear(first_embedding_dim, self.hidden_dim)
        # if last_weight_is_zero_init:
        #     with torch.no_grad():
        #         self.lin.weight.data.zero_()

        self.activation = get_activation(activation)()

    def forward(self, h, *args):
        h = self.prelin(h)
        params = self.lin_emb(torch.cat(args, 1))
        scale, shift = params[..., :self.hidden_dim], params[..., self.hidden_dim:]
        # self.lin.weight = self.lin.weight / self.lin.weight.sum(1, keepdim=True)
        preact = self.lin(h * scale + shift)
        # preact = h * scale + shift
        act = self.activation(preact)
        return act

#
# class SeparationDNNA(torch.nn.Module):
#     def __init__(self, data, use_batch_norm=False, use_instance_norm=False,
#                  entangling_method='product'):
#         super(SeparationDNNA, self).__init__()
#         self.entangling_method = entangling_method
#         self.latent_dim = 2
#         num_blocks = 1
#         hidden_dim = 128
#
#         bias_before = True
#         last_weight_is_zero_init = True
#         self.d = lambda x: torch.sum(x, -1, keepdim=True)
#         # self.d = nn.Linear(hidden_dim, 1)
#         # if last_weight_is_zero_init:
#         #     with torch.no_grad():
#         #         self.d.weight.data.zero_()
#         self.data = data
#         # self.d = SymmetricInitDNN([hidden_dim, 1], "identity")
#         # self.d = SymmetricInitDNN([hidden_dim, hidden_dim, 1], "lrelu")
#         self.z = DNN([1, hidden_dim, hidden_dim,  hidden_dim, self.latent_dim], 'sin',
#                      bias_before=bias_before, last_weight_is_zero_init=True)
#         # self.p = DNN([1, hidden_dim, hidden_dim,  hidden_dim, self.latent_dim], 'sin',
#         #              bias_before=bias_before, last_weight_is_zero_init=True)
#         # self.z = DNN([1, hidden_dim,  self.latent_dim], 'lrelu',
#         #              bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         # self.p = DNN([1, hidden_dim, self.latent_dim], 'lrelu',
#         #              bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         # embedding_dim = self.latent_dim
#         embedding_dim = 2
#         self.e = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
#         self.h0 = torch.nn.Parameter(torch.randn(1, hidden_dim))
#         self.hh0 = torch.nn.Parameter(torch.randn(1, hidden_dim))
#         # with torch.no_grad():
#         #     self.e.weight.data.zero_()
#         # self.e2l = DNN([embedding_dim+1, hidden_dim, hidden_dim, hidden_dim,  self.latent_dim], 'sin',
#         #              bias_before=bias_before, last_weight_is_zero_init=True)
#         self.blocks = nn.ModuleList([Block(i==0, 2 * self.latent_dim, hidden_dim, 'sin', 1) for i in range(num_blocks)])
#         self.e2ls = nn.ModuleList([Block(i==0, embedding_dim + 1, hidden_dim, 'sin', 1) for i in range(1)])
#         self.ll = nn.Linear(hidden_dim, self.latent_dim)
#         self.ll.weight.data.zero_()
#         self.zt = None
#
#     def forward(self, x, variables):
#         x, t = x[:, [0]], x[:, [1]]
#         # zt, px = self.z(t), self.p(x)
#         zt = self.z(t)
#         # print(torch.unique(variables))
#
#         if torch.sum(variables==-1) > 0:
#             mean_emb = self.e.weight.sum(0)
#             index = variables==-1
#             variables[variables==-1] = 0 # set these to any value
#             ev = self.e(variables)
#             ev[index] = mean_emb
#         else:
#             ev = self.e(variables)
#
#         # print(torch.unique(ev[:, 0]))
#         # ev = self.e2l(ev)
#         hh = self.hh0
#         for b in self.e2ls:
#             hh = b(hh, torch.cat([ev, x], 1))
#         hh = self.ll(hh)
#
#         # ev = self.e2l(torch.cat([ev, x], 1))
#         h = self.h0
#         for b in self.blocks:
#             # h = b(h, zt * ev)
#             h = b(h, zt, hh)
#         ut = self.d(h)
#
#         self.zt, self.px = zt, hh
#         self.prod = zt * hh
#         self.t, self.x = t, x
#
#         return ut

#
# class SeparationDNNAA(torch.nn.Module):
#     def __init__(self, data, use_batch_norm=False, use_instance_norm=False,
#                  entangling_method='product'):
#         super(SeparationDNNAA, self).__init__()
#         self.entangling_method = entangling_method
#         self.latent_dim = 20
#         num_blocks = 4
#         hidden_dim = 256
#
#         # bias_before = True
#         # last_weight_is_zero_init = True
#         # self.d = lambda x: torch.sum(x, -1, keepdim=True)
#         # self.d = nn.Linear(hidden_dim, 1)
#         # if last_weight_is_zero_init:
#         #     with torch.no_grad():
#         #         self.d.weight.data.zero_()
#         self.data = data
#         self.d = SymmetricInitDNN([hidden_dim, 1], "identity")
#         # self.d = SymmetricInitDNN([hidden_dim, hidden_dim, 1], "lrelu")
#         # self.z = DNN([1, hidden_dim, hidden_dim, hidden_dim, self.latent_dim], 'sin',
#         #              bias_before=bias_before, last_weight_is_zero_init=True)
#         # self.p = DNN([1, hidden_dim, hidden_dim,  hidden_dim, self.latent_dim], 'sin',
#         #              bias_before=bias_before, last_weight_is_zero_init=True)
#         # self.z = DNN([1, hidden_dim,  self.latent_dim], 'lrelu',
#         #              bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         # self.p = DNN([1, hidden_dim, self.latent_dim], 'lrelu',
#         #              bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         # embedding_dim = self.latent_dim
#         embedding_dim = self.latent_dim
#         # self.h0 = torch.nn.Parameter(torch.randn(1, hidden_dim))
#         self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))
#
#         # with torch.no_grad():
#         #     self.e.weight.data.zero_()
#         # self.e2l = DNN([embedding_dim+1, hidden_dim, hidden_dim, hidden_dim,  self.latent_dim], 'sin',
#         #              bias_before=bias_before, last_weight_is_zero_init=True)
#         self.blocks = nn.ModuleList(
#             [Block(i == 0, 2 * self.latent_dim, hidden_dim, 'sin', 1) for i in range(num_blocks)])
#
#         self.hh0x = torch.nn.Parameter(torch.randn(1, hidden_dim))
#         self.ex = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
#         # with torch.no_grad():
#         #     xbound = 1 / embedding_dim
#         #     # xbound = bound / np.sqrt(T)
#         #     xbias = torch.empty((embedding_dim,))
#         #     init.uniform_(xbias, -xbound, xbound)
#         #     self.ex.weight.data = xbias.unsqueeze(0).repeat(self.ex.weight.data.shape[0], 1)
#
#         self.e2lsx = nn.ModuleList([Block(i == 0, embedding_dim + 1, hidden_dim, 'sin', 1) for i in range(5)])
#         self.llx = nn.Linear(hidden_dim, self.latent_dim)
#         self.llx.weight.data.zero_()
#         with torch.no_grad():
#             xbound = 1 / self.latent_dim
#             xbound = xbound / np.sqrt(100)
#             xbias = torch.empty((self.latent_dim,))
#             init.uniform_(xbias, -xbound, xbound)
#             self.llx.bias.data = xbias
#             init.normal_(self.llx.weight, 0, xbound * 0.1)
#             # init.normal_(self.llx.weight, 0, xbound * 1)
#
#         self.hh0t = torch.nn.Parameter(torch.randn(1, hidden_dim))
#         self.et = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
#         # with torch.no_grad():
#         #     tbound = 1 / embedding_dim
#         #     # tbound = bound / np.sqrt(L)
#         #     tbias = torch.empty((embedding_dim,))
#         #     init.uniform_(tbias, -tbound, tbound)
#         #     self.et.weight.data = tbias.unsqueeze(0).repeat(self.et.weight.data.shape[0], 1)
#
#         self.e2lst = nn.ModuleList([Block(i == 0, embedding_dim + 1, hidden_dim, 'sin', 1) for i in range(5)])
#         self.llt = nn.Linear(hidden_dim, self.latent_dim)
#         self.llt.weight.data.zero_()
#         with torch.no_grad():
#             tbound = 1 / self.latent_dim
#             tbound = tbound / np.sqrt(100)
#             tbias = torch.empty((self.latent_dim,))
#             init.uniform_(tbias, -tbound, tbound)
#             self.llt.bias.data = tbias
#             init.normal_(self.llt.weight, 0, xbound * 0.1)
#             # init.normal_(self.llt.weight, 0, xbound * 1)
#         d = 2
#         self.extra_lins_zt = nn.Sequential(*[nn.Linear(self.latent_dim, self.latent_dim) for i in range(d)])
#         self.extra_lins_px = nn.Sequential(*[nn.Linear(self.latent_dim, self.latent_dim) for i in range(d)])
#
#         self.zt = None
#
#     def forward(self, x, variables, evx=None, evt=None):
#         x, t = x[:, [0]], x[:, [1]]
#         if evx is None:
#             if any(variables == -1):
#                 mean_emb = self.ex.weight.sum(0)
#                 index = variables == -1
#                 variables[variables == -1] = 0  # set these to any value
#                 evx = self.ex(variables)
#                 evx[index] = mean_emb
#             else:
#                 evx = self.ex(variables)
#         if evt is None:
#             evt = self.et(variables)
#
#         hhx = self.hh0x
#         for b in self.e2lsx:
#             hhx = b(hhx, torch.cat([evx, x], 1))
#         px = self.llx(hhx)
#         px = self.extra_lins_px(px)
#
#         # hhx_zero = self.hh0x
#         # for b in self.e2lsx:
#         #     hhx_zero = b(hhx_zero, torch.cat([evx, x[[0]].repeat(len(evx), 1)], 1))
#         # pxz_zero = self.llx(hhx)
#         # px = px - pxz_zero
#
#         hht = self.hh0t
#         for b in self.e2lst:
#             hht = b(hht, torch.cat([evt, t], 1))
#         zt = self.llt(hht)
#         zt = self.extra_lins_px(zt)
#
#         # hht_zero = self.hh0t
#         # for b in self.e2lst:
#         #     hht_zero = b(hht_zero, torch.cat([evt, t[[0]].repeat(len(evt), 1)], 1))
#         # zt_zero = self.llt(hht_zero)
#         # zt = zt - zt_zero
#
#         h = self.h0
#         for b in self.blocks:
#             h = b(h, zt, px)
#         ut = self.d(h)
#
#         self.zt, self.px = zt, px
#         self.prod = zt * px
#         self.t, self.x = t, x
#         self.evx = evx
#         self.evt = evt
#
#         return ut


class SeparationDNNAAEmbedding(torch.nn.Module):
    """Add central embedding to the main network"""
    def __init__(self, data, use_batch_norm=False, use_instance_norm=False,
                 entangling_method='product'):
        super(SeparationDNNAAEmbedding, self).__init__()
        self.entangling_method = entangling_method
        self.latent_dim = 20
        num_blocks = 4
        hidden_dim = 256

        # bias_before = True
        # last_weight_is_zero_init = True
        # self.d = lambda x: torch.sum(x, -1, keepdim=True)
        # self.d = nn.Linear(hidden_dim, 1)
        # if last_weight_is_zero_init:
        #     with torch.no_grad():
        #         self.d.weight.data.zero_()
        self.data = data
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
        # embedding_dim = self.latent_dim
        embedding_dim = self.latent_dim
        self.use_central_emb = False
        # self.h0 = torch.nn.Parameter(torch.randn(1, hidden_dim))
        # self.h0 = torch.nn.Parameter(torch.zeros(1, hidden_dim))

        if self.use_central_emb:
            self.h0 = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
        else:
            self.h0 = torch.nn.Parameter(torch.zeros(1, embedding_dim))

        # with torch.no_grad():
        #     self.e.weight.data.zero_()
        # self.e2l = DNN([embedding_dim+1, hidden_dim, hidden_dim, hidden_dim,  self.latent_dim], 'sin',
        #              bias_before=bias_before, last_weight_is_zero_init=True)
        arg_dim = 2 * self.latent_dim + (embedding_dim if self.use_central_emb else 0)
        self.blocks = nn.ModuleList(
            [Block(i == 0, arg_dim, hidden_dim, 'sin', 1, first_embedding_dim=embedding_dim) for i in range(num_blocks)])

        # self.hh0x = torch.nn.Parameter(torch.randn(1, hidden_dim))
        self.ex = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
        self.ex.weight.data.zero_()

        # with torch.no_grad():
        #     xbound = 1 / embedding_dim
        #     # xbound = bound / np.sqrt(T)
        #     xbias = torch.empty((embedding_dim,))
        #     init.uniform_(xbias, -xbound, xbound)
        #     self.ex.weight.data = xbias.unsqueeze(0).repeat(self.ex.weight.data.shape[0], 1)

        self.e2lsx = nn.ModuleList([Block(i == 0, embedding_dim + 1, hidden_dim, 'sin', 1, first_embedding_dim=embedding_dim) for i in range(5)])
        self.llx = nn.Linear(hidden_dim, self.latent_dim)
        # self.llx.weight.data.zero_()
        # self.llx.bias.data.zero_()
        with torch.no_grad():
            xbound = 1 / self.latent_dim
            xbound = xbound / np.sqrt(100)
            xbias = torch.empty((self.latent_dim,))
            init.uniform_(xbias, -xbound, xbound)
            self.llx.bias.data = xbias
            # self.llx.bias.data.zero_()
            init.normal_(self.llx.weight, 0, xbound * 0.1)
            # init.normal_(self.llx.weight, 0, xbound * 1)

        # self.hh0t = torch.nn.Parameter(torch.randn(1, hidden_dim))
        self.et = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
        self.et.weight.data.zero_()
        # with torch.no_grad():
        #     tbound = 1 / embedding_dim
        #     # tbound = bound / np.sqrt(L)
        #     tbias = torch.empty((embedding_dim,))
        #     init.uniform_(tbias, -tbound, tbound)
        #     self.et.weight.data = tbias.unsqueeze(0).repeat(self.et.weight.data.shape[0], 1)

        self.e2lst = nn.ModuleList([Block(i == 0, embedding_dim + 1, hidden_dim, 'sin', 1, first_embedding_dim=embedding_dim) for i in range(5)])
        self.llt = nn.Linear(hidden_dim, self.latent_dim)
        # self.llt.weight.data.zero_()
        with torch.no_grad():
            tbound = 1 / self.latent_dim
            tbound = tbound / np.sqrt(100)
            tbias = torch.empty((self.latent_dim,))
            init.uniform_(tbias, -tbound, tbound)
            self.llt.bias.data = tbias
            # self.llt.bias.data.zero_()
            init.normal_(self.llt.weight, 0, xbound * 0.1)
            # init.normal_(self.llt.weight, 0, xbound * 1)
        d = 3
        self.extra_lins_zt = nn.Sequential(*[nn.Linear(self.latent_dim, self.latent_dim) for i in range(d)])
        self.extra_lins_px = nn.Sequential(*[nn.Linear(self.latent_dim, self.latent_dim) for i in range(d)])

        self.zt = None

    def forward(self, x, variables, evx=None, evt=None):
        x, t = x[:, [0]], x[:, [1]]
        if evx is None:
            if any(variables == -1):
                mean_emb = self.ex.weight.sum(0)
                index = variables == -1
                variables[variables == -1] = 0  # set these to any value
                evx = self.ex(variables)
                evx[index] = mean_emb
            else:
                evx = self.ex(variables)
        if evt is None:
            evt = self.et(variables)

        hhx = evx
        for b in self.e2lsx:
            hhx = b(hhx, torch.cat([evx, x], 1))
        px = self.llx(hhx)
        px = self.extra_lins_px(px)

        # hhx_zero = self.hh0x
        # for b in self.e2lsx:
        #     hhx_zero = b(hhx_zero, torch.cat([evx, x[[0]].repeat(len(evx), 1)], 1))
        # pxz_zero = self.llx(hhx)
        # px = px - pxz_zero

        hht = evt
        for b in self.e2lst:
            hht = b(hht, torch.cat([evt, t], 1))
        zt = self.llt(hht)
        zt = self.extra_lins_px(zt)

        # hht_zero = self.hh0t
        # for b in self.e2lst:
        #     hht_zero = b(hht_zero, torch.cat([evt, t[[0]].repeat(len(evt), 1)], 1))
        # zt_zero = self.llt(hht_zero)
        # zt = zt - zt_zero
        if self.use_central_emb:
            h0_repeated = self.h0(variables)
        else:
            h0_repeated = self.h0.repeat((zt.shape[0], 1))
        h = h0_repeated
        # h = self.h0
        for b in self.blocks:
            args = (zt, px)
            if self.use_central_emb:
                args = (*args, h0_repeated)
            h = b(h, *args)
        ut = self.d(h)

        self.zt, self.px = zt, px
        self.prod = zt * px
        self.t, self.x = t, x
        self.evx = evx
        self.evt = evt

        return ut


# class SeparationDNNAugS(torch.nn.Module):
#     def __init__(self, data, use_batch_norm=False, use_instance_norm=False,
#                  entangling_method='product'):
#         super(SeparationDNNAugS, self).__init__()
#         self.entangling_method = entangling_method
#         self.latent_dim = 2
#         self.depth = 4
#         hidden_dim = 256
#         bias_before = False
#         last_weight_is_zero_init = True
#
#         if entangling_method == 'concat':
#             latent_dim_factor = 2
#         else:
#             latent_dim_factor = 1
#         self.d = lambda x: torch.sum(x, -1, keepdim=True)
#         # self.d = DNN([self.latent_dim] + [layers[-1]], 'identity', use_batch_norm, use_instance_norm,
#         #     last_weight_is_zero_init=True)
#         # self.d = SymmetricInitDNN([self.latent_dim * latent_dim_factor, hidden_dim, 1],
#         #                           "lrelu", use_batch_norm, use_instance_norm)
#         bias_before = False
#         last_weight_is_zero_init = True
#         self.z = DNN([1, hidden_dim, hidden_dim, self.latent_dim], 'sin',
#                      bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         # self.z = EulerNet(self.latent_dim)
#         self.p = DNN([1, hidden_dim, hidden_dim, self.latent_dim], 'sin',
#                      # bias_before=bias_before, last_weight_is_zero_init=True)
#                      bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         embedding_dim = self.latent_dim
#         self.e2l = DNN([1 + self.latent_dim, hidden_dim, hidden_dim, self.latent_dim], 'relu',
#                      bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         self.e = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
#         with torch.no_grad():
#             self.e.weight.data.zero_()
#         # self.zphase = nn.Parameter(torch.linspace(0, 1, self.latent_dim).unsqueeze(0))
#         self.zt, self.px, self.ev = (None,) * 3
#
#     def forward(self, x, variables):
#         x, t = x[:, [0]], x[:, [1]]
#         ev = self.e(variables)
#         zt, px = self.z(t), self.p(x)
#         zt = zt - self.z(t[[0]] * 0)  # setting z(t=0) to 1
#         l = self.e2l(torch.concat([ev, x], 1))
#         self.zt = zt
#         self.px = px
#         if self.entangling_method == 'product':
#             entangling = zt * px * l
#             self.prod = entangling
#         elif self.entangling_method == 'concat':
#             entangling = torch.concat([zt, px], dim=-1)
#             self.prod = zt
#         else:
#             raise NotImplementedError(self.entangling_method)
#         self.t = t
#         self.x = x
#         ut = self.d(entangling)
#         return ut
#
# #
# class SeparationDNNAugM(torch.nn.Module):
#     def __init__(self, data, use_batch_norm=False, use_instance_norm=False,
#                  entangling_method='product'):
#         super(SeparationDNNAugM, self).__init__()
#         self.entangling_method = entangling_method
#         self.latent_dim = 2
#         self.depth = 4
#         hidden_dim = 256
#         bias_before = False
#         last_weight_is_zero_init = True
#
#         if entangling_method == 'concat':
#             latent_dim_factor = 2
#         else:
#             latent_dim_factor = 1
#         # self.d = lambda x: torch.sum(x, -1, keepdim=True)
#         # self.d = DNN([self.latent_dim] + [layers[-1]], 'identity', use_batch_norm, use_instance_norm,
#         #     last_weight_is_zero_init=True)
#         # self.d = SymmetricInitDNN([self.latent_dim * latent_dim_factor, hidden_dim, 1],
#         #                           "lrelu", use_batch_norm, use_instance_norm)
#
#         self.zs = torch.nn.ModuleList(
#             [DNN([1, hidden_dim, hidden_dim, self.latent_dim], 'smallsin',
#                 bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#             for _ in range(self.depth)]
#         )
#         self.ps = torch.nn.ModuleList(
#             [DNN([1, hidden_dim, hidden_dim, self.latent_dim], 'smallsin',
#                 bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#             for _ in range(self.depth)]
#         )
#         embedding_dim = self.latent_dim
#         self.e2ls = torch.nn.ModuleList(
#             [DNN([embedding_dim, hidden_dim, hidden_dim, self.latent_dim], 'smallsin',
#                      bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#             for _ in range(self.depth)]
#         )
#         self.lins = torch.nn.ModuleList(
#             [
#                 # DNN([self.latent_dim, hidden_dim, hidden_dim, self.latent_dim], 'lrelu',
#                 #  bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#                 torch.nn.Linear(self.latent_dim, self.latent_dim)
#              for _ in range(self.depth)]
#         )
#         with torch.no_grad():
#             for lin in self.lins:
#                 torch.nn.init.constant_(lin.bias.data, 0.9)
#                 torch.nn.init.orthogonal_(lin.weight, gain=0.1)
#         self.e = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
#         self.last_lin = torch.nn.Linear(self.latent_dim, 1)
#         with torch.no_grad():
#             self.e.weight.data.zero_()
#         self.zt, self.px, self.ev = (None,) * 3
#
#     def forward(self, x, variables):
#         x, t = x[:, [0]], x[:, [1]]
#         ev = self.e(variables)
#         out = 1
#         for d, (z, p, e2l, lin) in enumerate(zip(self.zs, self.ps, self.e2ls, self.lins)):
#             zt, px, l = z(t), p(x), e2l(ev)
#             out  = lin(zt * px * l) * out
#             self.zt, self.px, self.ev, self.prod = zt, px, l, out
#         self.t = t
#         self.x = x
#         return self.last_lin(out)

#
# class SeparationDNNAug(torch.nn.Module):
#     def __init__(self, data, use_batch_norm=False, use_instance_norm=False,
#                  entangling_method='product'):
#         super(SeparationDNNAug, self).__init__()
#         self.entangling_method = entangling_method
#         self.latent_dim = 12
#         hidden_dim = 256
#
#         if entangling_method == 'concat':
#             latent_dim_factor = 2
#         else:
#             latent_dim_factor = 1
#         # self.d = lambda x: torch.sum(x, -1, keepdim=True)
#         # self.d = DNN([self.latent_dim] + [layers[-1]], 'identity', use_batch_norm, use_instance_norm,
#         #     last_weight_is_zero_init=True)
#         self.d = SymmetricInitDNN([self.latent_dim * latent_dim_factor, hidden_dim, 1],
#                                   "lrelu", use_batch_norm, use_instance_norm)
#         # self.d = DNN([self.latent_dim * latent_dim_factor, hidden_dim, 1],
#         #                           "lrelu", use_batch_norm, use_instance_norm, last_weight_is_zero_init=True)
#         # self.d = DNN([self.latent_dim * latent_dim_factor, hidden_dim, 1],
#         #                           "identity", use_batch_norm, use_instance_norm, last_weight_is_zero_init=True)
#         # self.d = Resnet(self.latent_dim * latent_dim_factor, 1, [hidden_dim, hidden_dim],
#         #                 5, "relu")
#         # self.d = SymmetricInitDNN([self.latent_dim * latent_dim_factor, hidden_dim, 1], 'identity', use_batch_norm, use_instance_norm)
#         # self.d = SymmetricInitDNN([self.latent_dim] + [layers[-1]], 'identity', use_batch_norm, use_instance_norm)
#         # self.d = DNN([self.latent_dim, hidden_dim, 1], "relu", use_batch_norm, use_instance_norm)
#         # self.d = DNN([self.latent_dim, 512, 1], 'identity', use_batch_norm, use_instance_norm)
#         # self.z = Resnet(1, self.latent_dim, [hidden_dim, hidden_dim], 5, activation)
#         # self.p = Resnet(1, self.latent_dim, [hidden_dim, hidden_dim], 5, activation)
#         bias_before = True
#         last_weight_is_zero_init = True
#         self.z = DNN([1, hidden_dim, hidden_dim, self.latent_dim], 'sin',
#                      bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         # self.z = EulerNet(self.latent_dim)
#         self.p = DNN([1, hidden_dim, hidden_dim, self.latent_dim], 'sin',
#                      # bias_before=bias_before, last_weight_is_zero_init=True)
#                      bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         embedding_dim = self.latent_dim
#         self.e2l = DNN([embedding_dim, hidden_dim, self.latent_dim], 'sin',
#                      bias_before=bias_before, last_weight_is_zero_init=last_weight_is_zero_init)
#         self.e = torch.nn.Embedding(len(data.variables_hash), embedding_dim)
#         with torch.no_grad():
#             self.e.weight.data.zero_()
#         # self.zphase = nn.Parameter(torch.linspace(0, 1, self.latent_dim).unsqueeze(0))
#         self.zt, self.px, self.ev = (None,) * 3
#
#     def forward(self, x, variables):
#         x, t = x[:, [0]], x[:, [1]]
#         zt, px = self.z(t), self.p(x)
#         px = px - self.p(x[[0]]*0) + 1
#         zt = zt - self.z(t[[0]]*0) + 1# setting z(t=0) to 1
#         self.zt = zt
#         self.px = px
#         if self.entangling_method == 'product':
#             entangling = zt * px
#             if variables is not None:
#                 self.evl = self.e2l(self.e(variables))
#                 entangling *= self.evl
#             self.prod = entangling
#
#         elif self.entangling_method == 'concat':
#             entangling = torch.concat([zt, px], dim=-1)
#             self.prod = zt
#         else:
#             raise NotImplementedError(self.entangling_method)
#         self.t = t
#         self.x = x
#         ut = self.d(entangling)
#         return ut
#
#



class Data:
    def __init__(self, system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho, u_star, X_star, u0_str):
        self.system = system
        self.u0_str = u0_str
        self.x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)

        self.u_star = torch.tensor(u_star).float().to(device)  # for regression to the solution directly
        self.X_star = torch.tensor(X_star).float().to(device)

        self.u = torch.tensor(u_train, requires_grad=True).float().to(device)
        self.layers = layers
        self.nu = nu
        self.beta = beta
        self.rho = rho

        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)

class DataList:
    def __init__(self, variables_attr, data_list):
        self.variables_attr = variables_attr
        self._list = data_list
        for attr in ['system', 'layers', 'nu', 'beta', 'rho', 'u0_str']:
            setattr(self, attr, [getattr(data, attr)  for data in data_list])

        for attr in ['x_u', 't_u', 'x_f', 't_f', 'x_bc_lb', 't_bc_lb',
                     'x_bc_ub', 't_bc_ub', 'u_star', 'X_star', 'u', 'G']:
            setattr(self, attr, torch.concat([getattr(data, attr)  for data in data_list], 0))

        self.variables = list(zip(*[getattr(self, attr) for attr in variables_attr]))
        self.variables_hash = hash(self.variables)
        self.variable_tensor = self.to_tensor(self.variables, len(self._list[0].X_star))

    def to_tensor(self, variables, size):
        variable_list = []
        for v in variables:
            if v == 'interp':
                variable_list += [-1] * size
            else:
                variable_list += [self.variables_hash[v]] * size
        return torch.tensor(variable_list).to(self.X_star.device)

    def __getitem__(self, item):
        return self._list[item]

    def __iter__(self):
        return iter(self._list)


class PhysicsInformedNN_pbc_aug:
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, data, optimizer_name, lr,
                 net, L=1, activation='tanh', loss_style='mean', train_method='pinns'):
        self.data = data
        self.train_method = train_method
        self.net = net
        if self.net == 'DNN':
            # self.dnn = LatentDNN(layers, activation).to(device)
            # self.dnn = SeparationDNNAug(data).to(device)
            # self.dnn = SeparationDNNAugM(data).to(device)
            # self.dnn = SeparationDNNA(data).to(device)
            # self.dnn = SeparationDNNAA(data).to(device)
            self.dnn = SeparationDNNAAEmbedding(data).to(device)
        else:  # "pretrained" can be included in model path
            # the dnn is within the PINNs class
            self.dnn = torch.load(net).dnn
        self.L = L
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)
        self.loss_style = loss_style
        self.iter = 0


    def net_u(self, x, t, variable_tensor):
        """The standard DNN that takes (x,t) --> u."""
        u = self.dnn(torch.cat([x, t], dim=1), variable_tensor)
        return u

    # def net_f(self, x, t):
    #     """ Autograd for calculating the residual for different systems."""
    #     u = self.net_u(x, t)
    #
    #     u_t = torch.autograd.grad(
    #         u, t,
    #         grad_outputs=torch.ones_like(u),
    #         retain_graph=True,
    #         create_graph=True
    #     )[0]
    #     u_x = torch.autograd.grad(
    #         u, x,
    #         grad_outputs=torch.ones_like(u),
    #         retain_graph=True,
    #         create_graph=True
    #     )[0]
    #
    #     u_xx = torch.autograd.grad(
    #         u_x, x,
    #         grad_outputs=torch.ones_like(u_x),
    #         retain_graph=True,
    #         create_graph=True
    #     )[0]
    #
    #     if 'convection' in self.system or 'diffusion' in self.system:
    #         if any(self.G != 0):
    #             assert 'Neeeded to put back self.G'
    #         # f = u_t - self.nu*u_xx + self.beta*u_x - self.G
    #         f = u_t - self.nu*u_xx + self.beta*u_x
    #     elif 'rd' in self.system:
    #         f = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
    #     elif 'reaction' in self.system:
    #         f = u_t - self.rho*u + self.rho*u**2
    #     return f
    #
    # def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
    #     """For taking BC derivatives."""
    #
    #     u_lb_x = torch.autograd.grad(
    #         u_lb, x_bc_lb,
    #         grad_outputs=torch.ones_like(u_lb),
    #         retain_graph=True,
    #         create_graph=True
    #     )[0]
    #
    #     u_ub_x = torch.autograd.grad(
    #         u_ub, x_bc_ub,
    #         grad_outputs=torch.ones_like(u_ub),
    #         retain_graph=True,
    #         create_graph=True
    #     )[0]
    #
    #     return u_lb_x, u_ub_x

    def loss_regression(self, verbose=True):
        u_pred = self.net_u(self.data.X_star[:, [0]],
                            self.data.X_star[:, [1]],
                            self.data.variable_tensor)
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        if self.loss_style == 'mean':
            loss_u = torch.mean((u_pred - self.data.u_star) ** 2)
        elif self.loss_style == 'sum':
            loss_u = torch.sum((u_pred - self.data.u_star) ** 2)
        loss = loss_u

        if loss.requires_grad:
            loss.backward()

        # grad_norm = 0
        # for p in self.dnn.parameters():
        #     param_norm = p.grad.detach().data.norm(2)
        #     grad_norm += param_norm.item() ** 2
        # grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 500 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss: %.5e' % (self.iter, 0, loss.item(), )
                )
            self.iter += 1

        return loss

    #
    # def loss_pinn(self, verbose=True):
    #     """ Loss function. """
    #     if torch.is_grad_enabled():
    #         self.optimizer.zero_grad()
    #     u_pred = self.net_u(self.x_u, self.t_u)
    #     u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb)
    #     u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub)
    #     if self.nu != 0:
    #         u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
    #     f_pred = self.net_f(self.x_f, self.t_f)
    #
    #     if self.loss_style == 'mean':
    #         loss_u = torch.mean((self.u - u_pred) ** 2)
    #         loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
    #         if self.nu != 0:
    #             loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
    #         loss_f = torch.mean(f_pred ** 2)
    #     elif self.loss_style == 'sum':
    #         loss_u = torch.sum((self.u - u_pred) ** 2)
    #         loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
    #         if self.nu != 0:
    #             loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
    #         loss_f = torch.sum(f_pred ** 2)
    #
    #     loss = loss_u + loss_b + self.L*loss_f
    #
    #     if loss.requires_grad:
    #         loss.backward()
    #
    #     grad_norm = 0
    #     for p in self.dnn.parameters():
    #         param_norm = p.grad.detach().data.norm(2)
    #         grad_norm += param_norm.item() ** 2
    #     grad_norm = grad_norm ** 0.5
    #
    #     if verbose:
    #         if self.iter % 100 == 0:
    #             print(
    #                 'epoch %d, gradient: %.5e, loss: %.5e, loss_u: %.5e, loss_b: %.5e, loss_f: %.5e' % (self.iter, grad_norm, loss.item(), loss_u.item(), loss_b.item(), loss_f.item())
    #             )
    #         self.iter += 1
    #
    #     return loss

    def train(self):
        self.dnn.train()
        if self.train_method == 'pinns':
            self.optimizer.step(self.loss_pinn)
        elif self.train_method == 'regression':
            self.optimizer.step(self.loss_regression)
        else:
            assert False


    def predict(self, X, variable):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        variable_tensor = self.data.to_tensor(variable, len(X))

        self.dnn.eval()
        u = self.net_u(x, t, variable_tensor)
        u = u.detach().cpu().numpy()

        return u
    #
    # def predict_f(self, X):
    #     x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
    #     t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
    #
    #     self.dnn.eval()
    #     f = self.net_f(x, t)
    #     f = f.detach().cpu().numpy() ** 2
    #
    #     return f