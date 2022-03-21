import numpy as np
import torch.utils.data as data
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset

from pbc_examples.data.plot import plot_solution_1d
from pbc_examples.data.simple_pde import gen_coordinates
from pbc_examples.data.utilities_fno import MatReader


def ns_fno_dataset(n=None, timelen=10, train=True, startfrom=0):
    # TRAIN_PATH = "NavierStokes_V1e-5_N1200_T20.mat"
    # TEST_PATH = "NavierStokes_V1e-5_N1200_T20.mat"
    if train:
        PATH = "/r/ada-24/edebezenac/data/NavierStokes_V1e-5_N1200_T20.mat"
    else:
        PATH = "/r/ada-24/edebezenac/data/NavierStokes_V1e-5_N1200_T20.mat"
    # TRAIN_PATH = "/data/debezenac/data/ns_V1e-3_N5000_T50.mat"
    # TEST_PATH = "/data/debezenac/data/ns_V1e-3_N5000_T50.mat"
    # TRAIN_PATH = '/data/ns_data_V100_N1000_T50_1.mat'
    # TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'
    if n is None:
        n = 1200

    # ntest = 20
    #
    # modes = 12
    # width = 20

    # batch_size = 20
    # batch_size2 = batch_size
    #
    # # epochs = 500
    # # learning_ra:te = 0.001
    # # scheduler_step = 100
    # # scheduler_gamma = 0.5
    #
    # path = 'ns_fourier_2d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
    # path_model = 'model/'+path
    # path_train_err = 'results/'+path+'train.txt'
    # path_test_err = 'results/'+path+'test.txt'
    # path_image = 'image/'+path

    sub = 1
    S = 64
    T_in = 10
    T = timelen
    # step = 1

    ################################################################
    # load data
    ################################################################
    reader = MatReader(PATH)
    # train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
    train_u = reader.read_field('u')[startfrom : n + startfrom, ::sub, ::sub, T_in : T + T_in]

    assert (S == train_u.shape[-2])
    assert (T == train_u.shape[-1])

    return TensorDataset(train_u)


class ImplicitWrapper(data.Dataset):
    '''Takes a dataset of values on a grid and creates a new dataset where the inputs
    correspond to the coordinates of each point on the grid, and targets to their values '''
    def __init__(self, dataset, coords_args):
        self.dataset = dataset
        self.grids, self.mgrid = gen_coordinates(coords_args)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        u, = self.dataset[index]
        return {'u': u.permute(2, 0, 1),
                'item': index,
                'mgrid': self.mgrid, **self.grids}


def NSDataset():
    ns_dataset = ns_fno_dataset()
    nx, nt = ns_dataset[0][0].shape[-2:]
    return ImplicitWrapper(ns_dataset,
                           coords_args=(('x', np.linspace(-1, 1, nx)),
                                        ('y', np.linspace(-1, 1, nx)),
                                        ('t', np.linspace(0, 1, nt))))


if __name__ == '__main__':

    dataset = NSDataset()
    print(dataset)
    data = dataset[0]
    print(data['u'].shape)
    plot_solution_1d(data['u'][0], data['x'], data['t'])
    plt.show()
