from functools import partial

import torch
from torch import nn
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
    # index = torch.arange(0, len(x_in))
    # pindex = (index + 1) % len(x_in)
    # fx_in: (nx, nc) -> (nb, nc, nx)
    fx_in_reshaped = fx_in.T.unsqueeze(0)
    conv = nn.Conv1d(fx_in.shape[1], fx_in.shape[1], 3, padding=1, padding_mode='reflect')
    Qf_reshaped = conv(fx_in_reshaped)
    Qf = Qf_reshaped.squeeze(0).T
    # Qf = fx_in[pindex]
    # translate continuously
    # Qf = fx_in
    # x_out  = x_out - math.pi / 8
    Qfx = interpolate_1d(x_in, Qf, x_out)
    return Qfx


# def translate_op(fx):
#     # fx = f(x)
#     index = torch.arange(0, len(fx))
#     pindex = (index + 100) % len(fx)
#     Qf = fx[pindex]
#     return Qf


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


# def equiv(Q, u, v, x):
#     ux = u(x)
#     Qux = Q(ux)
#     vux = diff(u, v, x)
#     dQvu = grad(
#         Qux * vux.detach(), ux,
#         grad_outputs=torch.ones_like(Qux),
#         retain_graph=True,
#         create_graph=True,
#     )[0]
#     jacQ = jacobian(
#         Q, ux,
#         create_graph=True,
#     )
#     jacQ = jacQ.squeeze(3).squeeze(1)
#     dQvuu = torch.einsum('ij,jl -> il', jacQ.T, vux)
#
#     # vQu = diff(u, v, x)
#     return dQvu - vQu, dQvu, -vQu


def equiv_cont(Q, u, v, x):
    x.requires_grad_(True)
    ux = u(x)
    vux = diff(u, v, x)
    plt.plot(x.detach().cpu().numpy(), vux.detach().cpu().numpy(), label='vux')

    ux = ux.detach()
    ux.requires_grad_(True)
    # vux = vux.detach()
    # vux.requires_grad_(True)
    # x_new = x.clone().detach()
    # x = x.clone().detach()
    # x_new.requires_grad_(True)
    # x.requires_grad_(True)
    # ux = ux.detach()
    # ux.requires_grad_(True)
    x.requires_grad_(False)

    Qux = Q(ux, x.detach(), x.detach())
    dQvu = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]


    ux.requires_grad_(True)
    def partial_Q(x_in, x_out):
        def QQ(ux):
            return Q(ux, x_in, x_out)
        return QQ
    jacQ = jacobian(
        partial_Q(x, x), ux,
        create_graph=True,
    )
    jacQ = jacQ.squeeze(3).squeeze(1)
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(jacQ.detach().cpu().numpy())
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.plot(vux.squeeze().detach().cpu().numpy())
    # plt.show()
    dQvuu = torch.einsum('ij,jl -> il', jacQ.T, vux)
    x.requires_grad_(False)

    x_new = x.clone().detach()
    x_new.requires_grad_(True)
    ux = u(x).detach()
    ux.requires_grad_(False)
    vQu = diff(partial(Q, ux, x.detach()), v, x_new)

    x = x.clone().detach()
    x_new = x_new.clone().detach()
    x_new.requires_grad_(True)
    vx = v(x)
    # y, v = F_func(x), v_func(x)
    y = Q(ux, x, x_new)
    dFdx = grad(
        y, x_new,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]
    vQuu = torch.einsum('...i,...i->...', dFdx, vx).unsqueeze(-1)

    return dQvu - vQu, dQvu, -vQu




def equiv_cont_(Q, u, v, x):
    x.requires_grad_(True)
    ux = u(x)
    vux = diff(u, v, x)
    plt.plot(x.detach().cpu().numpy(), vux.detach().cpu().numpy(), label='vux')

    ux = ux.detach()
    ux.requires_grad_(True)
    # vux = vux.detach()
    # vux.requires_grad_(True)
    # x_new = x.clone().detach()
    # x = x.clone().detach()
    # x_new.requires_grad_(True)
    # x.requires_grad_(True)
    # ux = ux.detach()
    # ux.requires_grad_(True)
    x.requires_grad_(False)

    Qux = Q(ux, x.detach(), x.detach())
    dQvu = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]


    ux = u(x).detach()
    ux = ux.detach()
    ux.requires_grad_(True)
    jacQ = jacobian(
        partial(Q, x_in=x.detach(), x_out=x.detach()), ux,
        create_graph=True,
    )
    jacQ = jacQ.squeeze(3).squeeze(1)
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(jacQ.detach().cpu().numpy())
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.plot(vux.squeeze().detach().cpu().numpy())
    # plt.show()
    dQvuu = torch.einsum('ij,jl -> il', jacQ.T, vux)
    x.requires_grad_(False)

    x_new = x.clone().detach()
    x_new.requires_grad_(True)
    ux = u(x).detach()
    ux.requires_grad_(False)
    vQu = diff(partial(Q, ux, x.detach()), v, x_new)

    x = x.clone().detach()
    x_new = x_new.clone().detach()
    x_new.requires_grad_(True)
    vx = v(x)
    # y, v = F_func(x), v_func(x)
    y = Q(ux, x, x_new)
    dFdx = grad(
        y, x_new,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]
    vQuu = torch.einsum('...i,...i->...', dFdx, vx).unsqueeze(-1)

    return dQvu - vQu, dQvu, -vQu



if __name__ == '__main__':
    torch.manual_seed(1)
    n = 500
    x = torch.linspace(-math.pi, math.pi, n).unsqueeze(-1)
    # x_new = torch.linspace(-math.pi, math.pi, n).unsqueeze(-1)
    # x.requires_grad = True
    # out = diff(u0, v, x)
    # fx = u0(x)
    # interp = interpolate_1d(x, fx, x_new)
    # plt.plot(x.detach().numpy(), fx.detach().numpy(), label='equiv')
    # plt.plot(x_new.detach().numpy(), interp.detach().numpy(), label='interp')
    # plt.legend()
    # plt.show()

    e, dQvu, vQu = equiv_cont(translate_op_continuous, u0, v, x)
    y = translate_op_continuous(u0(x), x, x)

    plt.plot(x.detach().numpy(), u0(x).detach().numpy(), label='u0')
    plt.plot(x.detach().numpy(), u0_deriv(x).detach().numpy(), label='deriv', alpha=0.5)
    plt.plot(x.detach().numpy(), -e.detach().numpy(), label='equiv')
    plt.plot(x.detach().numpy(), y.detach().numpy(), label='y')
    # plt.plot(x.detach().numpy(), -(u0_deriv(x - 2 * math.pi / 5)).detach().numpy(), label="u0'(x-e)")
    plt.plot(x.detach().numpy(), -vQu.detach().numpy(), label='vQu')
    plt.plot(x.detach().numpy(), -dQvu.detach().numpy(), label='dQvu')
    plt.legend()
    plt.grid()
    plt.show()