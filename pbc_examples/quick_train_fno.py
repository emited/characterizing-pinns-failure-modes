import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pbc_examples.data.params import to_numpy, create_params
from pbc_examples.data.simple_pde import SimplePDEDataset
from pbc_examples.data.plot import plot_solution_1d, plot_latents
from pbc_examples.modules.separation_embedding import SeparationEmbedding, EmbeddingModule
from pbc_examples.modules.separation_embedding_modulation import SeparationEmbeddingMod
from pbc_examples.modules.separation_param import SeparationParam, ParamModule
from pbc_examples.modules.separation_param_modulation import SeparationParamM
from pbc_examples.modules.separation_param_modulation_big import SeparationParamMod
from pbc_examples.modules.separation_param_simple_latents import SeparationParamSimpleLatent, ModulatedLinear
from pbc_examples.modules.separation_param_simple_latents_un import SeparationParamSimpleLatentUn
from pbc_examples.modules.separation_param_simplest import SeparationParamSimplest


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
        output = self.model(input['x'], input['t'], self._get_aux_input(input))
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0)
        # emb_params = [p for n, p in self.model.named_parameters() if n in ['ex.weight', 'et.weight']]
        # other_params = [p for n, p in self.model.named_parameters() if n not in ['ex.weight', 'et.weight']]
        # optimizer = torch.optim.Adam([
        #     {'params': emb_params, 'lr':0.002, 'weight_decay':0},
        #     {'params': other_params, 'lr':0.002, 'weight_decay':0},
        # ], lr=0.002)
        # optimizer = torch.optim.Adam([
        #     {'params': self.model.xt_param_group, 'lr':0.002, 'weight_decay':0},
        #     {'params': self.model.x_param_group, 'lr':0.002, 'weight_decay':0},
        #     {'params': self.model.t_param_group, 'lr':0.002, 'weight_decay':0},
        # ], lr=0.002)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input = train_batch
        output = self.model(input['x'], input['t'], self._get_aux_input(input))
        loss = F.mse_loss(output['u_pred'], train_batch['u'])
        return loss

    def _plot_swapped_embeddings(self, input, swap_index_1, swap_index_2):
        i, j = swap_index_1, swap_index_2
        exi, etj = self.model.get_ex(torch.tensor([i], device=input['x'].device)), \
                   self.model.get_et(torch.tensor([j], device=input['x'].device))
        output_ij = self.model(input['x'][[i]], input['t'][[j]], ex=exi, et=etj)
        plot_solution_1d(to_numpy(output_ij['u_pred']).squeeze(),
                         to_numpy(input['x']).squeeze(),
                         to_numpy(input['t']).squeeze(),
                         title=f'{self.prefix}: Extrap'
                               f' sample (px, zt): {i, j},\n {input["system"][i], input["system"][j]}, \n'
                               f' beta: {input["beta"][i].item(), input["beta"][j].item()},'
                               f' nu: {input["nu"][i].item(), input["nu"][j].item()}\n'
                               f' u0: {input["u0_str"][i], input["u0_str"][j]}\n')

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
        output = self.model(input['x'], input['t'], self._get_aux_input(input))
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

    # data_args_list = [
    #     # ('convection-diffusion', 1, 1, 0, 'sin(2x)', 100, 100, 0),
    #     # ('convection-diffusion', 1, 1, 0, 'sin(x)', 100, 100, 0),
    #     # ('convection', 0, 1, 0, 'np.sin(3*x)', 100, 100, 0),
    #     # ('convection', 0, 4, 0, 'np.sin(3*x)', 100, 100, 0),
    #     ('convection', 0, 1, 0, 'sin(x)', 100, 100, 0),
    #     ('convection', 0, 2, 0, 'sin(x)', 100, 100, 0),
    #     ('convection', 0, 2, 0, 'gauss', 100, 100, 0),
    #     ('convection', 0, 3, 0, 'gauss', 100, 100, 0),
    #     ('convection', 0, 3, 0, 'sin(x)', 100, 100, 0),
    #     ('convection', 0, 4, 0, 'gauss', 100, 100, 0),
    #     ('convection', 0, 4, 0, 'sin(x)', 100, 100, 0),
    # ]
    # p = np.linspace(-1, 1, len(data_args_list))
    # params = np.expand_dims(p, -1).astype(np.float32)


    # params = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]).astype(np.float32)
    # data_args_list = [('convection', 0, 1, 0, f'np.sin({1}*x)', 100, 100, 0),
    #                   ('convection', 0, 4, 0, f'np.sin({1}*x)', 100, 100, 0),
    #                   ('convection', 0, 1, 0, f'np.sin({2}*x)', 100, 100, 0),
    #                   ('convection', 0, 4, 0, f'np.sin({2}*x)', 100, 100, 0),
    #                   ]

    # params = np.array([[0], [1]])
    # # data_args_list = [('convection', 0, 1, 0, 'np.exp(-np.power((x - 0.5*np.pi)/(np.pi/64), 2.)/2.)', 100, 100, 0),]
    # data_args_list = [
    #     ('convection', 0, 1, 0, 'np.sin(2*x)', 100, 100, 0),
    #     ('convection', 0, 1, 0, 'np.sin(1*x)', 100, 100, 0),
    # ]



    # params = np.array([[[.1], [.1]], [[-.1], [-.1]]], dtype=np.float32)
    # data_args_list = [
    #     ('convection', 0, 1, 0, 'np.sin(2*x)', 100, 100, 0),
    #     ('convection', 0, 4, 0, 'np.sin(1*x)', 100, 100, 0),
    # ]
    data_args_list, params, valid_args_list, valid_params = create_params('big', separate_params=separate_params)
    # data_args_list, params, valid_args_list, valid_params = create_params('gauss', separate_params=separate_params)
    # data_args_list, params, valid_args_list, valid_params = create_params('simple_swap_x2', separate_params=separate_params)
    # data_args_list, params, valid_args_list, valid_params = create_params('big', separate_params=True)

    dataset = SimplePDEDataset(data_args_list, params)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=1)

    val_dataset = SimplePDEDataset([(*d, 2) for d in valid_args_list], valid_params)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), num_workers=1)


    # model = SeparationExperiment(len(dataset))
    if separate_params:
        x_param_dim, t_param_dim = dataset[0]['x_params'].shape[0], dataset[0]['t_params'].shape[0]
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

