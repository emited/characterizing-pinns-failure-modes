from functools import partial

import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import math
from torch.autograd.functional import jacobian
from torch.nn.functional import interpolate, grid_sample

from pbc_examples.interp_1d import Interp1d


def interpolate_1d(x, y, x_new):
    return Interp1d()(x.squeeze(-1).unsqueeze(0),
                     y.squeeze(-1).unsqueeze(0),
                     x_new.squeeze(-1).unsqueeze(0)
                     ).squeeze(0).unsqueeze(-1)

def u0(x):
    return torch.cos(x)

def u0_deriv(x):
    return -torch.sin(x)

def translate_op_continuous(fx_in, x_in, x_out):
    index = torch.arange(0, len(x_in))
    pindex = (index + 10) % len(x_in)
    Qf = fx_in[pindex]
    Qfx = interpolate_1d(x_in, Qf, x_out)
    return Qfx


def translate_op(fx):
    # fx = f(x)
    index = torch.arange(0, len(fx))
    pindex = (index + 10) % len(fx)
    Qf = fx[pindex]
    return Qf


def v(x):
    return torch.ones_like(x)


def diff(F_func, v_func, x):
    y, v = F_func(x), v_func(x)
    dFdx = grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]
    return torch.einsum('...i,...i->...', dFdx, v).unsqueeze(-1)


def equiv(Q, u, v, x):
    ux = u(x)
    Qux = Q(ux)
    vux = diff(u, v, x)
    dQvu = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]
    # jac = jacobian(
    #     Q, ux,
    #     create_graph=True,
    # )
    # jac = jac.squeeze(3).squeeze(1)
    # dQvuu = torch.einsum('ij,jl -> il', jac.T, vux)

    # vQu = diff(****** , v, x)
    return dQvu - vQu, dQvu, -vQu


def equiv_cont(Q, u, v, x):
    ux = u(x)
    vux = diff(u, v, x)
    x_new = x
    Qux = Q(ux, x, x_new)
    dQvu = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]

    vQu = diff(partial(Q, ux, x), v, x_new)
    return dQvu - vQu, dQvu, -vQu

if __name__ == '__main__':
    n = 5000
    x = torch.linspace(-math.pi, math.pi, n).unsqueeze(-1)
    x_new = torch.linspace(-math.pi, math.pi, n).unsqueeze(-1)
    x.requires_grad = True
    out = diff(u0, v, x)
    fx = u0(x)
    # interp = interpolate_1d(x, fx, x_new)
    # plt.plot(x.detach().numpy(), fx.detach().numpy(), label='equiv')
    # plt.plot(x_new.detach().numpy(), interp.detach().numpy(), label='interp')
    # plt.legend()
    # plt.show()

    e, dQvu, vQu = equiv_cont(translate_op_continuous, u0, v, x)


    plt.plot(x.detach().numpy(), u0(x).detach().numpy())
    # plt.plot(x.detach().numpy(), u0_deriv(x).detach().numpy())
    plt.plot(x.detach().numpy(), -e.detach().numpy(), label='equiv')
    plt.plot(x.detach().numpy(), -vQu.detach().numpy(), label='vQu')
    plt.plot(x.detach().numpy(), -dQvu.detach().numpy(), label='dQvu')
    plt.legend()
    plt.show()