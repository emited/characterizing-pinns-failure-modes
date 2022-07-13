from torch.utils.data import Dataset
import deepxde as dde
import numpy as np

class SimpleDataset(Dataset):
    """Dataset that uses pde, a DeepXDE:PDE object
        Same underlying pde for samples across the batch
    """
    def __init__(self, data: dde.data.PDE, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        from deepxde import losses as losses_module

        self.loss_fn = losses_module.get("MSE")

        def losses_train(inputs, outputs):
            return self.data.losses_train(None, outputs, self.loss_fn, inputs, None)
        self.losses_fn = losses_train

    def __getitem__(self, item):
        x_all, y_all, aux = self.data.train_next_batch()
        # bcs = [bc.collocation_points(self.data.train_x_all) for bc in self.data.bcs]
        # x_bc = self.data.train_x_bc
        x_dom = self.data.geom.random_points(self.data.num_domain, random=self.data.train_distribution)
        x_bc = self.data.geom.random_boundary_points(
            self.data.num_boundary, random=self.data.train_distribution
        )
        if isinstance(self.data, dde.data.TimePDE):
            if self.data.num_initial > 0:
                if self.data.train_distribution == "uniform":
                    x_ic = self.data.geom.uniform_initial_points(self.data.num_initial)
                else:
                    x_ic = self.data.geom.random_initial_points(
                        self.data.num_initial, random=self.data.train_distribution
                    )
                if self.data.exclusions is not None:
                    def is_not_excluded(x):
                        return not np.any([np.allclose(x, y) for y in self.data.exclusions])

                    x_ic = np.array(list(filter(is_not_excluded, x_ic)))
            else:
                x_ic = np.empty((0, self.data.geom.dim), dtype=np.float32)
            x_bc = (x_ic, x_bc)
        y_all = [] if y_all is None else y_all
        aux = [] if aux is None else aux
        return (x_dom, x_bc), (x_all, y_all), aux


    def __len__(self):
        return self.batch_size
