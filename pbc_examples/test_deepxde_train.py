import pytorch_lightning as pl
from torch.utils.data import DataLoader
import deepxde as dde


from pbc_examples.pinns.datasets import SimpleDataset, create_pde_data
from pbc_examples.pinns.exp_pde_manifold import PDEManifoldExperiment, PINNExperiment, HPINNExperiment
from pbc_examples.utils import set_seed
from pbc_examples.modules.nn_symmetries import SymmetryNet, ModulatedSymmetryNet


if __name__ == '__main__':
    # set_seed(0)
    trainer = pl.Trainer(gpus=1,
                         # limit_train_batches=6,
                         limit_val_batches=1,
                         check_val_every_n_epoch=100, max_epochs=5000, log_every_n_steps=1)

    bs = 64
    data = create_pde_data('advection')
    dataset = SimpleDataset(data, batch_size=bs)
    train_loader = DataLoader(dataset, batch_size=bs)

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = HPINNExperiment(net, dataset, train_loader)
    # net = ModulatedSymmetryNet(2, 12, hidden_dim=64)
    # model = PDEManifoldExperiment(net, dataset, train_loader)

    trainer.fit(model, train_loader)
    trainer.validate(model)