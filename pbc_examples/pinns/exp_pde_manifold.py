import torch
import pytorch_lightning as pl
import deepxde as dde
import matplotlib.pyplot as plt


from pbc_examples.data.plot import plot_solution_1d
from pbc_examples.pinns.utils import generate_uniform_grid
from pbc_examples.data.params import to_numpy


class PINNExperiment(pl.LightningModule):
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
        (x_dom, x_bc), (x_all, y_all), aux = train_batch
        if isinstance(self.dataset.data, dde.data.TimePDE):
            x_ic, x_bc = x_bc
        x_dom.requires_grad_()
        # x = flatten_batch_dim(x)
        y_pred = self.model(x_dom)
        f = self.dataset.data.pde(x_dom, y_pred)
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

    def val_dataloader(self):
        return self.val_loader


class PDEManifoldExperiment(pl.LightningModule):
    def __init__(self, model, dataset, val_loader, base_distr='normal'):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.val_loader = val_loader
        self.base_distr = base_distr

    def forward(self, input):
        output = self.model(input)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        (x_dom, x_bc), (x_all, y_all), aux = train_batch
        if isinstance(self.dataset.data, dde.data.TimePDE):
            x_ic, x_bc = x_bc
        x_dom.requires_grad_()
        # x = flatten_batch_dim(x)
        z = self.sample_base_distr(x_dom.shape[0], self.model.z_dim)
        z = z.unsqueeze(1).expand(-1, x_dom.shape[1], -1)
        y_pred, style = self.model(x_dom, z)
        f = self.dataset.data.pde(x_dom, y_pred)
        # return torch.mean(1e-4 * (f ** 2) / (1e-8 + y_pred ** 2))
        return torch.mean(f ** 2)

    def sample_base_distr(self, nz, zdim, device=None):
        if device is None:
            device = 'cuda'
        if self.base_distr == 'normal':
            return torch.randn(nz, zdim, device=device)
        elif self.base_distr == 'uniform':
            return torch.zeros(nz, zdim, device=device).uniform_(-1, 1)
        raise NotImplementedError(self.base_distr)

    def validation_step(self, val_batch, batch_idx):
        X, x, t = generate_uniform_grid(self.dataset.data.geom, 101)
        X = torch.tensor(X).float().cuda() # (gridsize, coord_dim)
        z = self.sample_base_distr(5, self.model.z_dim)

        X = X.unsqueeze(0).expand(z.shape[0], -1, -1)
        z = z.unsqueeze(1).expand(-1, X.shape[1], -1)
        us, styles = self.model(X, z)
        for i, u in enumerate(us):
            u = u.reshape((len(t), len(x)))
            plot_solution_1d(to_numpy(u).squeeze(),
                             x.squeeze(),
                             t.squeeze(),
                             title=f'{i}')
            plt.show(bbox_inches='tight')

    def val_dataloader(self):
        return self.val_loader


class HPINNExperiment(pl.LightningModule):
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
        (x_dom, x_bc), (x_all, y_all), aux = train_batch
        if isinstance(self.dataset.data, dde.data.TimePDE):
            x_ic, x_bc = x_bc
        x_dom.requires_grad_()
        # x = flatten_batch_dim(x)
        y_pred = self.model(x_dom)
        f = self.dataset.data.pde(x_dom, y_pred)
        loss = torch.mean(f ** 2)
        u_lb_x  = torch.autograd.grad(
            loss, y_pred,
            grad_outputs=torch.ones_like(loss),
            retain_graph=True,
            create_graph=True,
        )[0]
        print('ok')

        return None

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

    def val_dataloader(self):
        return self.val_loader

