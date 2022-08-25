from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pbc_examples.data.params import to_numpy, create_params
from pbc_examples.data.simple_pde import SimplePDEDataset
from pbc_examples.data.plot import plot_solution_1d, plot_latents
from pbc_examples.fourier_continuous_test import v_translation, equiv_crit
from pbc_examples.modules.separation_embedding import SeparationEmbedding, EmbeddingModule
# from pbc_examples.modules.separation_embedding_modulation import SeparationEmbeddingMod
# from pbc_examples.modules.separation_param import SeparationParam, ParamModule
# from pbc_examples.modules.separation_param_modulation import SeparationParamM
# from pbc_examples.modules.separation_param_modulation_big import SeparationParamMod
# from pbc_examples.modules.separation_param_simple_latents import SeparationParamSimpleLatent, ModulatedLinear
# from pbc_examples.modules.separation_param_simple_latents_un import SeparationParamSimpleLatentUn
from pbc_examples.modules.separation_param_simplest import *



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

    def forward(self, coords, param):
        # ex : (B, emb_dim)
        # x: (B, xgrid, 1)
        x, t = coords[..., [0]], coords[..., [1]]

        # (B, emb_dim) -> (B, X, emb_dim)
        xparam, tparam = param['x_params'], param['t_params']
        # e = self.p2e(torch.cat([xparam, tparam], -1))
        ex = self.xp2ex(xparam.reshape((*xparam.shape[:-2], xparam.shape[-2] * xparam.shape[-1])))
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

        gh0_repeated = self.gh0.unsqueeze(1).unsqueeze(2).expand(*tr.shape[:-1], self.h0.shape[1])
        gh = gh0_repeated
        ex_broadcasted = ex.unsqueeze(1).unsqueeze(2).expand(-1, t.shape[1], x.shape[1], -1)
        # e_broadcasted = e.unsqueeze(1).unsqueeze(2).expand(-1, t.shape[1], x.shape[1], -1)
        for b in self.gblocks:
            gh = b(gh, torch.cat([h, ex_broadcasted], -1))

        u_pred = self.d(gh)

        return {'u_pred': u_pred,
                }




class SeparationExperiment(pl.LightningModule):
    def __init__(self, num_samples, param_dim=None, x_param_dim=None, t_param_dim=None, prefix=None, separate_params=True):
        super().__init__()
        self.prefix = prefix
        self.separate_params = separate_params
        # self.model = SeparationEmbedding(num_samples)
        # self.model = SeparationEmbeddingMod(num_samples)
        # self.model = SeparationParam(param_dim)
        # self.model = SeparationParamM(param_dim)
        # self.model = SeparationParamMod(x_param_dim=x_param_dim,
        #                                 t_param_dim=t_param_dim,
        #                                 separate_params=separate_params)
        # self.model = SeparationParamSimpleLatent(x_param_dim=x_param_dim,
        #                                          t_param_dim=t_param_dim,
        #                                          separate_params=separate_params)
        self.model = SeparationParamSimplest(x_param_dim=x_param_dim,
                                                 t_param_dim=t_param_dim,
                                                 separate_params=separate_params)

        # else:
        #     self.model = SeparationParamSimpleLatent(param_dim=param_dim, separate_params=separate_params)

    def _get_aux_input(self, input):
        if isinstance(self.model, ParamModule):
            if self.separate_params:
                # aux = (input['x_params'], input['t_params'] )
                # aux = (input['t_params'], input['x_params'] )
                aux = {'x_params': input['x_params'],
                       't_params': input['t_params']}
            else:
                aux = input['params']
        elif isinstance(self.model, EmbeddingModule):
            aux = input['item']
        else:
            assert False
        return aux

    def forward(self, input):
        aux_inputs = self._get_aux_input(input)
        u0_partial = partial(self.u0, omega=aux_inputs['x_params'])
        u0_partial = partial(self.u0, omega=aux_inputs['x_params'])
        aux_inputs['x_params'] = u0_partial(coords=input['x'])
        output = self.model(input['x'], input['t'], aux_inputs)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0)
        return optimizer

    def u0(self, coords, omega):
        x = coords[..., [0]]
        return torch.sin(omega * x)

    def training_step(self, train_batch, batch_idx):
        input = train_batch
        aux_inputs = self._get_aux_input(input)
        v = partial(v_translation, coords_to_translate=[0])

        # u0_partial = partial(self.u0, omega=aux_inputs['x_params'])
        # aux_inputs['x_params'] = u0_partial(coords=input['x'])
        coords = torch.cat([input['x'], input['t']], -1)
        coords.requires_grad_(True)
        u0_partial = partial(self.u0, omega=aux_inputs['x_params'].unsqueeze(1))
        aux_inputs['x_params'] = u0_partial(coords)
        output = self.model(coords, aux_inputs)
        equiv_crit(self.model, u0_partial, v, coords, is_Q_onet=True)
        loss = F.mse_loss(output['u_pred'], train_batch['u'])
        return loss

    def _plot_solutions(self, input, output):
        u = output['u_pred']
        for i, ui in enumerate(u):
            j = input['item'][i]
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 2, 1)
            plot_solution_1d(to_numpy(ui).squeeze(),
                             to_numpy(input['x']).squeeze(),
                             to_numpy(input['t']).squeeze(),
                             title=f'{self.prefix}: sample {j}, {input["system"][i]},'
                                f' beta: {input["beta"][i]},'
                                f' nu: {input["nu"][i]}'
                                f' u0: {input["u0_str"][i]}')
            plt.subplot(1, 2, 2)
            plot_solution_1d(to_numpy(input['u'][i]).squeeze(),
                             to_numpy(input['x']).squeeze(),
                             to_numpy(input['t']).squeeze(),
                             title=f'{self.prefix}: sample {j}, {input["system"][i]},'
                                f' beta: {input["beta"][i]},'
                                f' nu: {input["nu"][i]}'
                                f' u0: {input["u0_str"][i]}')
            plt.show()

    def _plot_latents(self, input, output):
        if 'px' in output and 'zt' in output:
            if 'system' not in input:
                input['system'] = [''] * len(input['item'])
                input['beta'] = [''] * len(input['item'])
                input['nu'] = [''] * len(input['item'])
                input['u0_str'] = [''] * len(input['item'])

            plot_latents(
                to_numpy(output['px']), to_numpy(output['zt']),
                to_numpy(input['x']).squeeze(),
                to_numpy(input['t']).squeeze(),
                labels=[f'sample {j}, {input["system"][i]},'
                           f' beta: {input["beta"][i]},'
                           f' nu: {input["nu"][i]}'
                           f' u0: {input["u0_str"][i]}'
                    for i, j in enumerate(input['item'])],
                prefix=self.prefix,)

    def validation_step(self, val_batch, batch_idx):
        input = val_batch
        if isinstance(self.model, EmbeddingModule):
            self._plot_swapped_embeddings(input, 2, 12)
            plt.show()
            self._plot_swapped_embeddings(input, 12, 2)
            plt.show()

        aux_inputs = self._get_aux_input(input)

        # u0_partial = partial(self.u0, omega=aux_inputs['x_params'])
        # aux_inputs['x_params'] = u0_partial(coords=input['x'])
        coords = torch.cat([input['x'], input['t']], -1)
        aux_inputs['x_params'] = self.u0(coords, omega=aux_inputs['x_params'].unsqueeze(1))
        output = self.model(coords, aux_inputs)

        # output = self.model(input['x'], input['t'], self._get_aux_input(input))
        self._plot_solutions(input, output)
        if 'zt' in output and 'px' in output:
            self._plot_latents(input, output)
            plt.show(bbox_inches='tight')

    def val_dataloader(self):
        return val_loader






if __name__ == '__main__':
    import sys
    prefix = sys.argv[1]
    separate_params = True

    data_args_list, params, valid_args_list, valid_params = create_params('diffusion', separate_params=separate_params)
    # data_args_list, params, valid_args_list, valid_params = create_params('simple_swap_x2', separate_params=separate_params)
    # data_args_list, params, valid_args_list, valid_params = create_params('big', separate_params=True)

    dataset = SimplePDEDataset(data_args_list, params)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=1)

    val_dataset = SimplePDEDataset([(*d, 1) for d in valid_args_list], valid_params)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), num_workers=1)


    # model = SeparationExperiment(len(dataset))
    if separate_params:
        x_param_dim, t_param_dim = dataset[0]['x_params'].shape[0], dataset[0]['t_params'].shape[0]
        x_param_dim = dataset.data_args_list[0][5]
        model = SeparationExperiment(len(params), x_param_dim=x_param_dim, t_param_dim=t_param_dim, prefix=prefix)
    else:
        param_dim = dataset[0]['params'].shape[0]
        model = SeparationExperiment(len(params), param_dim=param_dim, prefix=prefix)

    # training
    trainer = pl.Trainer(gpus=1,
                         # limit_train_batches=6, limit_val_batches=6,
                         check_val_every_n_epoch=50, max_epochs=600, log_every_n_steps=len(dataset))
    trainer.fit(model, train_loader)
    trainer.validate(model)

