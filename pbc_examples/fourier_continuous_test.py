import torch
from torch.fft import rfft, irfft, rfft2, irfft2, irfftn, rfftn
import math
import matplotlib.pyplot as plt
from torch.autograd import grad
# from torch.autograd.functional import jacobian
from functools import partial
import numpy

def low_pass_filter_1d(ux):
    uhat = rfft(ux.squeeze(-1))
    filterhat = torch.zeros_like(uhat)
    filterhat[:5] = 0.50
    out = irfft(uhat * filterhat)
    return out.unsqueeze(-1)


def conv_2d_fft(ux, filter):
    uhat = rfft2(ux.squeeze(-1), norm='forward')
    filterhat = rfft2(filter.squeeze(-1), norm='forward')
    out = irfft2(uhat * filterhat, norm='forward')
    out = torch.fft.fftshift(out)
    return out.unsqueeze(-1) * 8


def gauss_conv_2d(ux, batch_dim=True):
    import torch.nn as nn
    channels = ux.shape[-1]
    # Set these to whatever you want for your gaussian filter
    kernel_size = 31
    sigma = 5

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                padding=(kernel_size - 1) // 2, padding_mode='reflect',
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    # gaussian_filter.weight.requires_grad = False

    # (N, C_{in}, H_{in}, W_{in})
    if batch_dim:
        ux_reshaped = ux.permute(0, 3, 1, 2)
    else:
        ux_reshaped = ux.permute(2, 0, 1).unsqueeze(0)
    out = gaussian_filter(ux_reshaped)

    if batch_dim:
        out_reshaped = out.permute(0, 2, 3, 1)
    else:
        out_reshaped = out.squeeze(0).permute(1, 2, 0)
    return out_reshaped


def v_translation(x, u=None):
    v = torch.ones_like(x)
    if u is not None:
        v = torch.cat([v, torch.ones_like(u(x))], -1)
    return v


def v_scale(x, ux):
    v_x = torch.zeros_like(x)
    # v_u = torch.ones_like(ux)
    v_u = ux
    return torch.cat([v_x, v_u], -1)


def v_rotation(x, ux=None):
    assert x.shape[-1] == 2
    v = torch.ones_like(x)
    v[..., 0] = x[..., 1]
    v[..., 1] = -x[..., 0]
    if ux is not None:
        v = torch.cat([v, torch.ones_like(ux)], -1)
    return v


def v_galilean_boost(x, ux):
    """Galilean boost of the heat equation"""
    x, t = x[..., [0]], x[..., [1]]
    vks = [
        2 * t * torch.ones_like(x),
        torch.zeros_like(t),
        - x * torch.ones_like(ux)
    ]
    return torch.cat(vks, -1)


# TODO: implement infinite dimensional subalgebras

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

    Qux = cont_layer(ux)

    # this: vQu = <dQudx, vx> where
    # dQudx = Jac_u(Q).T . dudx = grad_x( Qux * dudx)
    dudx = grad(
        ux, x,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]
    dQudx = []
    for i in range(dudx.shape[-1]):
        dQudxi = grad(
            Qux * dudx[..., [i]].detach(), ux,
            grad_outputs=torch.ones_like(Qux),
            retain_graph=True,
            create_graph=True,
        )[0]
        dQudx.append(dQudxi)
    dQudx = torch.cat(dQudx, -1)

    vQu = torch.einsum('...i,...i->...', dQudx, vx).unsqueeze(-1)

    # instead of this, which does not work
    # because implicitly we are taking the derivative of
    # Qux.sum() wrt x and this mixes things up
    # vQu = change_along_flow(Qux, vx.detach(), x)

    dQvu = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]

    return dQvu - vQu, dQvu, -vQu


def equiv_cont_with_u(Q, u, v, x):
    '''
    TODO: make v a basis of the Lie Algebra and not just a
    single vector field.
    :param Q: continuous operator taking as input u(x)s
    :param u: input function defined on x, should be differentiable
    :param v: vector field function, defined on (x, u(x)),
        associated to the group action
    :param x: coordinates
    deals with arbitrary batch dimensions
    :return: dQvu - vQu, a (signed) infinitesimal measure of
     equivariance of Q wrt to vector field v
    '''
    x.requires_grad_(True)
    ux = u(x)
    vx = v(x, ux).detach()
    dudx = grad(
        ux, x,
        grad_outputs=torch.ones_like(ux),
        retain_graph=True,
        create_graph=True,
    )[0]
    dudx = torch.cat([dudx, torch.ones_like(dudx[..., [0]])], -1)
    vux = torch.einsum('...i,...i->...', dudx, vx).unsqueeze(-1)

    Qux = Q(ux)

    # this: vQu = <dQudx, vx> where
    # dQudx = Jac_u(Q).T . dudx = grad_x( Qux * dudx)
    dudx = grad(
        ux, x,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]

    # computing derivative wrt coordinates
    dQudx = []
    for i in range(dudx.shape[-1]):
        dQudxi = grad(
            Qux * dudx[..., [i]].detach(), ux,
            grad_outputs=torch.ones_like(Qux),
            retain_graph=True,
            create_graph=True,
        )[0]
        dQudx.append(dQudxi)
    dQudx = torch.cat(dQudx, -1)

    # computing jacobian of Q wrt output ux
    # # here there is a bug
    # jacQ = jacobian(
    #     cont_layer, ux,
    #     create_graph=True,
    # )
    # j = jacQ.squeeze(5).squeeze(2).reshape((jacQ.shape[0] * jacQ.shape[1], jacQ.shape[2]* jacQ.shape[3]))
    # j = torch.diagonal(j, dim1=2, dim2=3)
    # J = j.reshape((jacQ.shape[0], jacQ.shape[1]))
    # dQdu = grad(
    #         Qux, ux,
    #         grad_outputs=torch.ones_like(Qux),
    #         retain_graph=True,
    #         create_graph=True,
    #     )[0]

    # approximation with finite difference
    eps = 1e-3
    dQdu = (Q(ux + eps) - Qux) / eps
    # dQdu = (cont_layer(ux - eps) - 2 * Qux + cont_layer(ux + eps)) / eps
    dQudx = torch.cat([dQudx, dQdu], -1)
    vQ = v(x, Qux).detach()
    vQu = torch.einsum('...i,...i->...', dQudx, vQ).unsqueeze(-1)

    # instead of this, which does not work
    # because implicitly we are taking the derivative of
    # Qux.sum() wrt x and this mixes things up
    # vQu = change_along_flow(Qux, vx.detach(), x)

    dQvu = grad(
        Qux * vux.detach(), ux,
        grad_outputs=torch.ones_like(Qux),
        retain_graph=True,
        create_graph=True,
    )[0]

    return dQvu - vQu, dQvu, -vQu



# def run_1d_test():
#     x = torch.linspace(-math.pi, math.pi, 50).unsqueeze(-1)
#     u0 = torch.sin
#     y = low_pass_filter_1d(u0(x))
#
#     e, dQvu, mvQu = equiv_cont(low_pass_filter_1d, u0, v_translation, x)
#     eps = .2
#     gQu = y - eps * mvQu
#     Qgu = y + eps * dQvu
#
#     plt.figure(figsize=(8, 3))
#     plt.subplot(1, 2, 1)
#     plt.plot(x.detach().numpy(), y.detach().numpy(), label='ut')
#     plt.plot(x.detach().numpy(), u0(x).detach().numpy(), label='u0')
#     plt.plot(x.detach().numpy(), gQu.detach().numpy(), label='gQu')
#     plt.plot(x.detach().numpy(), Qgu.detach().numpy(), label='Qgu')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(x.detach().numpy(), mvQu.detach().numpy(), label='mvQu')
#     plt.plot(x.detach().numpy(), dQvu.detach().numpy(), label='dQvu')
#     plt.plot(x.detach().numpy(), e.detach().numpy(), label='dQvu - vQu')
#
#     plt.legend()
#     plt.show()


def run_2d_test():
    n = 100
    batch_dim = True

    x_ = torch.linspace(-math.pi, math.pi, n)
    y_ = torch.linspace(-math.pi, math.pi, n)
    x_grid, y_grid = torch.meshgrid(x_, y_)
    x = torch.stack([y_grid, x_grid], -1)

    def ring(x):
        center = [0, 1]
        r = (x[..., 0] - center[0]) ** 2 + (x[..., 1] - center[1]) ** 2
        scale = .6
        return torch.exp(-(r - 3) ** 2 / scale).unsqueeze(-1)

    def lines(x):
        r = x[..., 1] ** 2
        scale = .6
        lines = torch.exp(-(r - 3) ** 2 / scale).unsqueeze(-1)
        # mask = torch.ones_like(lines)
        # mask[torch.abs(x[..., 0]) > 1] = 0
        # mask[torch.abs(x[..., 1]) > 2] = 0
        # compact_lines = mask * lines
        # return compact_lines
        return lines


    u0 = lines

    # center = [0, 0]
    # r = (x[..., 0] - center[0]) ** 2 + (x[..., 1] - center[1]) ** 2
    # scale = .06
    # gauss = torch.exp(-r  ** 2 / scale).unsqueeze(-1)
    # gauss_filter = partial(conv_2d_fft, filter=gauss)
    if batch_dim:
        x = x.unsqueeze(0)

    gauss_filter = partial(gauss_conv_2d)
    y = gauss_filter(u0(x))

    # e, dQvu, mvQu = equiv_cont(gauss_filter, u0, v_translation, x)
    # e, dQvu, mvQu = equiv_cont(gauss_filter, u0, v_rotation, x)
    # e, dQvu, mvQu = equiv_cont_with_u(gauss_filter, u0, v_scale, x)
    e, dQvu, mvQu = equiv_cont_with_u(gauss_filter, u0, v_galilean_boost, x)

    eps = .2
    gQu = y - eps * mvQu
    Qgu = y + eps * dQvu
    u0x = u0(x)

    if batch_dim:
        u0x = u0x.squeeze(0)
        y = y.squeeze(0)
        gQu = gQu.squeeze(0)
        Qgu = Qgu.squeeze(0)
        mvQu = mvQu.squeeze(0)
        dQvu = dQvu.squeeze(0)
        e = e.squeeze(0)

    plt.figure(figsize=(6, 12))
    plt.subplot(4, 2, 1)
    plt.title('u0')
    plt.imshow(u0x.squeeze(-1).detach().numpy(), origin='lower')
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
    plt.title('vQu')
    plt.imshow(-mvQu.squeeze(-1).detach().numpy(),  origin='lower')
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

if __name__ == '__main__':
    # run_1d_test()
    run_2d_test()