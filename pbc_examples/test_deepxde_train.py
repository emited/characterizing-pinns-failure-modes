import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import deepxde as dde
import matplotlib.pyplot as plt

from pbc_examples.data.params import to_numpy
from pbc_examples.data.plot import plot_solution_1d
from pbc_examples.pinns.datasets import SimpleDataset
# from pbc_examples.pinns.pde_fns import advection
from pbc_examples.pinns import pde_fns
from pbc_examples.pinns.utils import generate_uniform_grid, DummyBC
from pbc_examples.utils import set_seed


class PDEManifoldExperiment(pl.LightningModule):
    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.dataset = dataset

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

    def val_dataloader(self):
        return train_loader


def create_pde_data(pde_name):
    pde_fn = getattr(pde_fns, pde_name)
    if pde_name in pde_fns.time_pdes:
        num_domain = 100
        num_boundary = 100
        num_initial = 100
        geom = dde.geometry.Interval(-1, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        bc = DummyBC(geomtime, lambda _, on_boundary: on_boundary, None)
        ic = dde.icbc.IC(geomtime, lambda x: x, lambda _, on_initial: on_initial)
        data = dde.data.TimePDE(geomtime, pde_fn, [ic, bc],
                                num_domain=num_domain,
                                num_boundary=num_boundary,
                                num_initial=num_initial)
        return data
    else:
        raise NotImplementedError(pde_name)



if __name__ == '__main__':
    # set_seed(0)

    trainer = pl.Trainer(gpus=1,
                         # limit_train_batches=6,
                         limit_val_batches=1,
                         check_val_every_n_epoch=50, max_epochs=400, log_every_n_steps=1)

    bs = 13
    data = create_pde_data('advection')
    dataset = SimpleDataset(data, batch_size=bs)
    train_loader = DataLoader(dataset, batch_size=bs)

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = PDEManifoldExperiment(net, dataset)

    trainer.fit(model, train_loader)
    trainer.validate(model)