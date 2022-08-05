import deepxde as dde
import numpy as np


def flatten_batch_dim(x):
    return x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))

def generate_uniform_grid(gxt: dde.geometry.GeometryXTime, nt: int):
    nx = int(
        np.ceil(
            nt
            * np.prod(gxt.geometry.bbox[1] - gxt.geometry.bbox[0])
            / gxt.timedomain.diam
        )
    )
    X = gxt.uniform_points(nx * nt, boundary=True)
    t = gxt.timedomain.uniform_points(nt, boundary=True)
    x = gxt.geometry.uniform_points(nx, boundary=True)
    return X, x, t

class DummyBC(dde.icbc.BC):
    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return None
