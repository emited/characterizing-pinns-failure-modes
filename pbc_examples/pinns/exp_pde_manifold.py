import torch
import pytorch_lightning as pl
import deepxde as dde
import matplotlib.pyplot as plt


from pbc_examples.data.plot import plot_solution_1d
from pbc_examples.pinns.utils import generate_uniform_grid
from pbc_examples.data.params import to_numpy


class PDEManifoldExperiment(pl.LightningModule):
    def __init__(self, model, dataset, val_loader):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.val_loader = val_loader

    def forward(self, input):
        output = self.model(input)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # domain_batch, ic_batch, bc_batch = self.separate_batch(train_batch)
        (x_dom, x_bc), (x_all, y_all), aux = train_batch
        if isinstance(self.dataset.data, dde.data.TimePDE):
            x_ic, x_bc = x_bc
        x_dom.requires_grad_()
        # x = flatten_batch_dim(x)
        y_pred = self.model(x_dom)
        f = self.dataset.data.pde(x_dom, y_pred)
        # loss = self.dataset.losses_fn(x_dom, y_pred)
        # return sum(loss)
        return torch.mean(f ** 2)

    def validation_step(self, val_batch, batch_idx):
        X, x, t = generate_uniform_grid(self.dataset.data.geom, 101)
        X = torch.tensor(X).float().cuda()
        u = self.model(X)
        u = u.reshape((len(t), len(x)))
        plot_solution_1d(to_numpy(u).squeeze(),
                         x.squeeze(),
                         t.squeeze(),
                         title='ok')
        plt.show(bbox_inches='tight')
        print('ok')

    def val_dataloader(self):
        return self.val_loader
