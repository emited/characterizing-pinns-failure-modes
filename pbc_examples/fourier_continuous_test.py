import torch
from torch.fft import rfft, irfft, rfft2, irfft2, irfftn, rfftn
import math
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.autograd.functional import jacobian
from functools import partial

def low_pass_filter_1d(ux):
    uhat = rfft(ux.squeeze(-1))
    filterhat = torch.zeros_like(uhat)
    filterhat[:5] = 0.50
    out = irfft(uhat * filterhat)
    return out.unsqueeze(-1)


def conv_2d(ux, filter):
    uhat = rfft2(ux.squeeze(-1), norm='forward')
    filterhat = rfft2(filter.squeeze(-1), norm='forward')
    out = irfft2(uhat * filterhat, norm='forward')
    out = torch.fft.fftshift(out)
    return out.unsqueeze(-1) * 80


def v_translation(x):
    return torch.ones_like(x)
    # u = torch.ones_like(x)
    # u[..., 0] = 0
    # return u


def v_rotation(x):
    print('ok')
    assert False
    return torch.ones_like(x)


def change_along_flow(Fx, vx, x):
    dFdx = grad(
        Fx, x,
        grad_outputs=torch.ones_like(Fx),
        retain_graph=True,
        create_graph=True,
    )[0]
    return torch.einsum('...i,...i->...', dFdx, vx).unsqueeze(-1)



def equiv_cont(cont_layer, u, v, x):

    x.requires_grad_(True)
    vx = v(x).detach()
    ux = u(x)
    vux = change_along_flow(ux, vx, x)

    # x = x.detach()
    # x.requires_grad_(True)
    # ux = u(x)
    Qux = cont_layer(ux)
    # dQux = grad(
    #     Qux * vux.detach(), ux,
    #     grad_outputs=torch.ones_like(Qux),
    #     retain_graph=True,
    #     create_graph=True,
    # )[0]


    dQux = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]

    vQu = change_along_flow(Qux, vx.detach(), x)

    dQvu = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]

    return dQvu - vQu, dQvu, -vQu


def run_1d_test():
    x = torch.linspace(-math.pi, math.pi, 50).unsqueeze(-1)
    u0 = torch.sin
    y = low_pass_filter_1d(u0(x))

    e, dQvu, mvQu = equiv_cont(low_pass_filter_1d, u0, v_translation, x)
    eps = .2
    gQu = y - eps * mvQu
    Qgu = y + eps * dQvu

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(x.detach().numpy(), y.detach().numpy(), label='ut')
    plt.plot(x.detach().numpy(), u0(x).detach().numpy(), label='u0')
    plt.plot(x.detach().numpy(), gQu.detach().numpy(), label='gQu')
    plt.plot(x.detach().numpy(), Qgu.detach().numpy(), label='Qgu')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x.detach().numpy(), mvQu.detach().numpy(), label='mvQu')
    plt.plot(x.detach().numpy(), dQvu.detach().numpy(), label='dQvu')
    plt.plot(x.detach().numpy(), e.detach().numpy(), label='dQvu - vQu')

    plt.legend()
    plt.show()


def run_2d_test():
    n = 1000
    x_ = torch.linspace(-math.pi, math.pi, n)
    y_ = torch.linspace(-math.pi, math.pi, n)
    x_grid, y_grid = torch.meshgrid(x_, y_)
    x = torch.stack([y_grid, x_grid], -1)

    def u0(x):
        r = x[..., 0] ** 2 + x[..., 1] ** 2
        scale = .6
        return torch.exp(-(r - 1) ** 2 / scale).unsqueeze(-1) + 1

    center = [0, 0]
    r = (x[..., 0] - center[0]) ** 2 + (x[..., 1] - center[1]) ** 2
    scale = .6
    gauss = torch.exp(-r  ** 2 / scale).unsqueeze(-1)
    gauss_filter = partial(conv_2d, filter=gauss)
    y = gauss_filter(u0(x))

    e, dQvu, mvQu = equiv_cont(gauss_filter, u0, v_translation, x)
    eps = .3
    gQu = y - eps * mvQu
    # gQu = mvQu
    Qgu = y + eps * dQvu

    plt.figure(figsize=(6, 12))
    plt.subplot(4, 2, 1)
    plt.title('u0')
    plt.imshow(u0(x).squeeze(-1).detach().numpy(), origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 2)
    plt.title('ut')
    plt.imshow(y.squeeze(-1).detach().numpy(), origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 3)
    plt.title('gQu')
    plt.imshow(gQu.squeeze(-1).detach().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 4)
    plt.title('Qgu')
    plt.imshow(Qgu.squeeze(-1).detach().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 5)
    plt.title('mvQu')
    plt.imshow(mvQu.squeeze(-1).detach().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 6)
    plt.title('dQvu')
    plt.imshow(dQvu.squeeze(-1).detach().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 7)
    plt.title('e')
    plt.imshow(e.squeeze(-1).detach().numpy(), origin='lower')
    plt.colorbar()
    plt.show()
    exit()

    plt.subplot(1, 2, 2)
    plt.plot(x.detach().numpy(), mvQu.detach().numpy(), label='mvQu')
    plt.plot(x.detach().numpy(), dQvu.detach().numpy(), label='dQvu')
    plt.plot(x.detach().numpy(), e.detach().numpy(), label='dQvu - vQu')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # run_1d_test()
    run_2d_test()