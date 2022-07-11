"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch

Implementation of Allen-Cahn equation example in paper https://arxiv.org/abs/2111.02801.
"""
import deepxde as dde
import numpy as np
from scipy.io import loadmat
# Import tf if using backend tensorflow.compat.v1 or tensorflow
# from deepxde.backend import torch
# Import torch if using backend pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from deepxde import losses as losses_module

def gen_testdata():
    data = loadmat("../dataset/Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    return dy_t - 3 * dy_x


class Data(Dataset):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

        def loss_train(inputs, outputs):
            return self.data.losses_train(None, outputs, loss_fn, inputs, None)

        self.loss_fn = loss_train

    def __getitem__(self, item):
        return self.data.train_points()

    def __len__(self):
        return self.batch_size


num_domain = 100
data = dde.data.TimePDE(geomtime, pde, [], num_domain=num_domain, num_boundary=0, num_initial=0)
bs = 13
ds = Data(data, batch_size=bs)
dl = DataLoader(ds, batch_size=bs)
x = next(iter(dl))

loss_fn = losses_module.get("MSE")


net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")




# with torch.no_grad():
#     x = x[0]
x.requires_grad = True
x.requires_grad_()
output = net(x)

def flatten_batch_dim(x):
    return x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))

def unflatten_batch_dim(x, batch_size):
    return x.reshape((batch_size, x.shape[0] // batch_size,  *x.shape[1:]))

x_flat = flatten_batch_dim(x)
loss = ds.loss_fn(x_flat, net(x_flat))


import matplotlib.pyplot as plt

plt.scatter(*x[0].cpu().numpy().T)
plt.scatter(*x[1].cpu().numpy().T)
plt.show()




model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
# model(x)
# model.train(epochs=40000)
losshistory, train_state = model.train(epochs=40)
# model.compile("L-BFGS")
# losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))