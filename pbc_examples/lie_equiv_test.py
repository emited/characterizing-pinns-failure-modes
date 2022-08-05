import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import math

def u0(x):
    return torch.cos(x)

def u0_deriv(x):
    return -torch.sin(x)

def Q(f):
    def func(x):
        fx = f(x)
        return 100 * fx + fx.sum(), fx
    return func

def v(x):
    return torch.ones_like(x)

def diff(F_func, v_func, x):
    y, v = F_func(x), v_func(x)
    if isinstance(y, tuple):
        y, _ = y
    dFdx = grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]

    return torch.einsum('...i,...i->...', dFdx, v).unsqueeze(-1)


def equiv(Q, u, v, x):
    Qux, ux = Q(u)(x)
    vux = diff(u, v, x)
    dQvu = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]
    vQu = diff(Q(u), v, x)
    return dQvu - vQu


if __name__ == '__main__':
    n = 100
    x = torch.linspace(-math.pi, math.pi, n).unsqueeze(-1)
    x.requires_grad = True
    out = diff(u0, v, x)

    e = equiv(Q, u0, v, x)

    plt.plot(x.detach().numpy(), u0(x).detach().numpy())
    plt.plot(x.detach().numpy(), u0_deriv(x).detach().numpy())
    plt.plot(x.detach().numpy(), -e.detach().numpy(), label='equiv')
    plt.legend()
    plt.show()