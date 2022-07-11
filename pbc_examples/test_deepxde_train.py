import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from deepxde import losses as losses_module
import deepxde as dde
import matplotlib.pyplot as plt


from pbc_examples.data.params import to_numpy
from pbc_examples.data.plot import plot_solution_1d


class Data(Dataset):
    def __init__(self, data, batch_size, loss='MSE'):
        self.data = data
        self.batch_size = batch_size
        self.loss_fn = losses_module.get(loss)

        def losses_train(inputs, outputs):
            return self.data.losses_train(None, outputs, self.loss_fn, inputs, None)

        self.losses_fn = losses_train

    def __getitem__(self, item):
        return self.data.train_points()

    def __len__(self):
        return self.batch_size

def flatten_batch_dim(x):
    return x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))


class SeparationExperiment(pl.LightningModule):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        self.X, self.y_true, self.u, self.x, self.t = gen_testdata()

    def forward(self, input):
        output = self.model(input)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        input = train_batch
        input.requires_grad_()
        input = flatten_batch_dim(input)
        output = self.model(input)
        loss = self.data.losses_fn(input, output)
        return sum(loss)

    def validation_step(self, val_batch, batch_idx):

        X = geomtime.uniform_points(100 * 100, boundary=True)
        X = torch.tensor(X).float().cuda()
        u = self.model(X)
        u = u.reshape((71, 142))
        plot_solution_1d(to_numpy(u).squeeze(),
                         self.x.squeeze(),
                         self.t.squeeze(),
                         title='ok')
        plt.show(bbox_inches='tight')

    def val_dataloader(self):
        return train_loader


if __name__ == '__main__':
    path = '/u/edebezenac/projects/deepxde/examples/dataset/Allen_Cahn.mat'

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        return dy_t + 3 * dy_x

    trainer = pl.Trainer(gpus=1,
                         # limit_train_batches=6,
                         limit_val_batches=1,
                         check_val_every_n_epoch=50, max_epochs=400, log_every_n_steps=1)
    num_domain = 100
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    data = dde.data.TimePDE(geomtime, pde, [], num_domain=num_domain, num_boundary=0, num_initial=0)
    bs = 13
    dataset = Data(data, batch_size=bs, loss='MSE')
    train_loader = DataLoader(dataset, batch_size=bs)
    device = torch.device('cuda')

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal").to(device)
    model = SeparationExperiment(net, dataset)

    trainer.fit(model, train_loader)
    trainer.validate(model)


    # f = pde(X, y_pred.to(device)).reshape(u.shape)
    # f = model.predict(X, operator=pde)
    # f = f.detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.imshow(f)
    # plt.show()