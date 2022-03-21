import itertools

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pbc_examples.data.simple_pde import SimplePDEDataset
from pbc_examples.data.plot import plot_solution_1d, plot_latents
from pbc_examples.modules.separation_embedding import SeparationEmbedding
from pbc_examples.modules.separation_param import SeparationParam
from pbc_examples.modules.separation_param_modulation import SeparationParamM
from pbc_examples.modules.separation_param_modulation_big import SeparationParamMod


def to_numpy(t):
    return t.detach().cpu().numpy()


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, num_samples):
        super().__init__()
        # self.model = SeparationEmbedding(num_samples)
        # self.model = SeparationParam(1)
        # self.model = SeparationParamM(num_samples)
        self.model = SeparationParamMod(num_samples)

    def _get_aux_input(self, input):
        if isinstance(self.model, SeparationParam) or \
            isinstance(self.model, SeparationParamM) or \
            isinstance(self.model, SeparationParamMod):
            aux = input['params']
        elif isinstance(self.model, SeparationEmbedding):
            aux = input['item']
        else:
            assert False
        return aux

    def forward(self, input):
        output = self.model(input['x'], input['t'], self._get_aux_input(input))
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input = train_batch
        output = self.model(input['x'], input['t'], self._get_aux_input(input))
        loss = F.mse_loss(output['u_pred'], train_batch['u'])
        return loss

    def validation_step(self, val_batch, batch_idx):
        input = val_batch
        output = self.model(input['x'], input['t'], self._get_aux_input(input))
        u = output['u_pred']
        for i, ui in enumerate(u):
            j = input['item'][i]
            plot_solution_1d(to_numpy(ui).squeeze(),
                             to_numpy(input['x']).squeeze(),
                             to_numpy(input['t']).squeeze(),
                             title=f'sample {j}, {input["system"][i]},'
                                f' beta: {input["beta"][i]},'
                                f' nu: {input["nu"][i]}'
                                f' u0: {input["u0_str"][i]}'
                             )
            plt.show()
        if 'px' in output and 'zt' in output:
            plot_latents(
                to_numpy(output['px']), to_numpy(output['zt']),
                to_numpy(input['x']).squeeze(),
                to_numpy(input['t']).squeeze(),
                labels=[f'sample {j}, {input["system"][i]},'
                           f' beta: {input["beta"][i]},'
                           f' nu: {input["nu"][i]}'
                           f' u0: {input["u0_str"][i]}'
                    for i, j in enumerate(input['item'])]
            )
            plt.show(bbox_inches='tight')

    def val_dataloader(self):
        return val_loader


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


n = 3
def make_norm_denorm_fns(a, b):
    def renorm(x):
        return (x - a) / (b - a)
    def denorm(y):
        return y * (b - a) + a
    return renorm, denorm

betas = np.linspace(0, 5, 5).astype(np.float32)
beta2p, p2beta = make_norm_denorm_fns(0, 5)

omegas = np.linspace(1, n, n).astype(np.float32)
omega2p, p2omega = make_norm_denorm_fns(1, n)


# params = np.hstack([omega2p(omegas), beta2p(betas)])
params = np.array(list(itertools.product(omega2p(omegas), beta2p(betas))))
data_args_list = [('convection-diffusion', 1, p2beta(betap), 0, f'np.sin({p2omega(omegap)}*x)', 100, 100, 0)
                  for omegap, betap in params]
                  # for beta in betas]


dataset = SimplePDEDataset(data_args_list, params)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=1)
# valid_data_args_list = data_args_list +\
#                        [('convection', 0, 1, 0, f'np.sin({2.5}*x)', 100, 100, 0)] +\
#                        [('convection', 0, 1, 0, f'np.sin({2.75}*x)', 100, 100, 0)]
# valid_params = np.vstack([params, np.array([[0.5], [0.75]], dtype=np.float32)])
valid_params, valid_data_args_list = params, data_args_list
val_dataset = SimplePDEDataset([(*d, 2) for d in valid_data_args_list], valid_params)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), num_workers=1)


# model
# model = LitAutoEncoder(len(dataset))
model = LitAutoEncoder(params.shape[1])

# training
trainer = pl.Trainer(gpus=1,
                     # limit_train_batches=6, limit_val_batches=6,
                     check_val_every_n_epoch=50, max_epochs=600, log_every_n_steps=len(dataset))
trainer.fit(model, train_loader)
trainer.validate(model)

