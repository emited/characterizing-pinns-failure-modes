import torch
from torch.autograd.functional import jacobian, vjp, jvp
from functorch import jvp as jpv_functorch
from torch.fft import rfft, irfft, rfft2, irfft2, irfftn, rfftn
import math
import matplotlib.pyplot as plt
from torch.autograd import grad
# from torch.autograd.functional import jacobian
from functools import partial
import numpy

def low_pass_filter_1d(ux):
    uhat = rfft(ux.squeeze(-1))
    filterhat = torch.zeros_like(uhat, device=ux.device)
    filterhat[:5] = 0.50
    out = irfft(uhat * filterhat)
    return out.unsqueeze(-1)


def conv_2d_fft(ux, filter):
    uhat = rfft2(ux.squeeze(-1), norm='forward')
    filterhat = rfft2(filter.squeeze(-1), norm='forward')
    out = irfft2(uhat * filterhat, norm='forward')
    out = torch.fft.fftshift(out)
    return out.unsqueeze(-1) * 8


def conv_2d_filter_given(channels=1, batch_dim=True, last_dim_scalar=False, filter='gaussian'):
    import torch.nn as nn

    # channels = ux.shape[-1]
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
    if filter == 'non-symmetric':
        where = (xy_grid - mean)[..., 0] > 0
        new_variance = variance
        new_xy_grid = xy_grid
        # new_xy_grid[..., 0] = xy_grid[..., 0] * 2
        new_variance = torch.ones(xy_grid.shape[-1], device=xy_grid.device) * variance
        new_variance[0] = variance * 5
        new_variance = new_variance.unsqueeze(0).unsqueeze(0)
        new_kernel = (1. / (2. * math.pi * variance)) * \
            torch.exp(
                -torch.sum(((new_xy_grid - mean) ** 2.) / new_variance, dim=-1) / 2
            )
        gaussian_kernel[where] = new_kernel[where]
        # gaussian_kernel = new_kernel

        # # plot as testing
        # import matplotlib.pyplot as plt
        # plt.imshow(gaussian_kernel.detach().cpu().numpy(), origin='lower')
        # plt.colorbar()
        # plt.show()

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                padding=(kernel_size - 1) // 2, padding_mode='reflect',
                                kernel_size=kernel_size, groups=channels, bias=False)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    gaussian_filter = gaussian_filter.cuda()

    def f(ux):

        if last_dim_scalar:
            ux = ux.unsqueeze(-1)

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

        if last_dim_scalar:
            out_reshaped = out_reshaped.squeeze(-1)

        return out_reshaped

    return f


def gauss_conv_2d(ux, batch_dim=True, last_dim_scalar=False):
    import torch.nn as nn

    if last_dim_scalar:
        ux = ux.unsqueeze(-1)

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
    gaussian_filter.weight.requires_grad = False

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

    if last_dim_scalar:
        out_reshaped = out_reshaped.squeeze(-1)

    return out_reshaped


def cont_layer_onet(ux, x):
    u2e = torch.nn.Linear(ux.shape[-1], 32)
    x2e = torch.nn.Linear(x.shape[-1], 32)
    e2o = torch.nn.Linear(64, ux.shape[-1])
    input = torch.cat([x2e(x), u2e(ux)], -1)
    out = e2o(input)
    return out


def v_translation(x, ux=None, coords_to_translate='all'):
    if coords_to_translate in ['x', 'all']:
        v = torch.ones_like(x, device=x.device)
    else:
        v = torch.zeros_like(x, device=x.device)
    if isinstance(coords_to_translate, (list, tuple)):
        for coord in coords_to_translate:
            v[coord] = 1
    if ux is not None:
        if coords_to_translate in ['all', 'u']:
            v_u = torch.ones_like(ux, device=ux.device)
        else:
            v_u = torch.zeros_like(ux, device=ux.device)
        v = torch.cat([v, v_u], -1)
    return v


def v_scale(x, ux):
    v_x = torch.zeros_like(x, device=x.device)
    # v_u = torch.ones_like(ux)
    v_u = ux
    return torch.cat([v_x, v_u], -1)


def v_rotation(x, ux=None):
    assert x.shape[-1] == 2
    v = torch.ones_like(x, device=x.device)
    v[..., 0] = x[..., 1]
    v[..., 1] = -x[..., 0]
    if ux is not None:
        v = torch.cat([v, torch.zeros_like(ux, device=ux.device)], -1)
    return v


def v_galilean_boost(x, ux):
    """Galilean boost of the heat equation"""
    x, t = x[..., [0]], x[..., [1]]
    vks = [
        2 * t * torch.ones_like(x, device=x.device),
        torch.zeros_like(t),
        - x * torch.ones_like(ux, device=x.device)
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


# def equiv_cont(cont_layer, u, v, x):
#
#     x.requires_grad_(True)
#     vx = v(x).detach()
#     ux = u(x)
#     vux = change_along_flow(ux, vx, x)
#
#     Qux = cont_layer(ux)
#
#     # this: vQu = <dQudx, vx> where
#     # dQudx = Jac_u(Q).T . dudx = grad_x( Qux * dudx)
#     dudx = grad(
#         ux, x,
#         grad_outputs=torch.ones_like(Qux),
#         retain_graph=True,
#         create_graph=True,
#     )[0]
#     dQudx = []
#     for i in range(dudx.shape[-1]):
#         dQudxi = grad(
#             Qux * dudx[..., [i]].detach(), ux,
#             grad_outputs=torch.ones_like(Qux),
#             retain_graph=True,
#             create_graph=True,
#         )[0]
#         dQudx.append(dQudxi)
#     dQudx = torch.cat(dQudx, -1)
#
#     vQu = torch.einsum('...i,...i->...', dQudx, vx).unsqueeze(-1)
#
#     # instead of this, which does not work
#     # because implicitly we are taking the derivative of
#     # Qux.sum() wrt x and this mixes things up
#     # vQu = change_along_flow(Qux, vx.detach(), x)
#
#     dQvu = grad(
#         Qux * vux.detach(), ux,
#         grad_outputs=torch.ones_like(Qux),
#         retain_graph=True,
#         create_graph=True,
#     )[0]
#
#     return dQvu - vQu, dQvu, -vQu


# def equiv_crit(Q, u, v, x, dudx=None, ux=None, Qux=None, Quxpeps=None, last_dim_scalar=False):
#     '''
#     TODO: make v a basis of the Lie Algebra and not just a
#     single vector field.
#     equivariance criterion, taking into account independant
#      and dependant variables
#     :param Q: continuous operator taking as input u(x)s
#     :param u: input function defined on x, should be differentiable
#     :param v: vector field function, defined on (x, u(x)),
#         associated to the group action
#     :param x: coordinates
#     deals with arbitrary batch dimensions
#     :return: dQvu - vQu, a (signed) infinitesimal measure of
#      equivariance of Q wrt to vector field v
#     '''
#     x.requires_grad_(True)
#     if ux is None:
#         ux = u(x)
#
#     if last_dim_scalar:
#         ux = ux.unsqueeze(-1)
#
#     vx = v(x, ux).detach()
#     if dudx is None:
#         dudx = grad(
#             ux, x,
#             grad_outputs=torch.ones_like(ux),
#             retain_graph=True,
#             create_graph=True,
#         )[0]
#     dudu = torch.ones_like(dudx[..., [0]], device=dudx.device)
#     dudxu = torch.cat([dudx, dudu], -1)
#     vux = torch.einsum('...i,...i->...', dudxu, vx).unsqueeze(-1)
#
#     # if is_Q_onet:
#     #     x_ = x.clone().detach()
#     #     x_.requires_grad_(True)
#     #     Qux = Q(ux, x_)
#     # else:
#     #     Qux = Q(ux)
#     if last_dim_scalar:
#         ux = ux.squeeze(-1)
#
#     if Qux is None:
#         Qux = Q(ux)
#
#     if last_dim_scalar:
#         Qux = Qux.unsqueeze(-1)
#
#     # this: vQu = <dQudx, vx> where
#     # dQudx = Jac_u(Q).T . dudx = grad_x( Qux * dudx )
#     # dudx = grad(
#     #     ux, x,
#     #     grad_outputs=torch.ones_like(Qux),
#     #     retain_graph=True,
#     #     create_graph=True,
#     # )[0]
#
#     # computing derivative wrt coordinates
#     dQudx = []
#     for i in range(dudx.shape[-1]):
#         dQudxi = grad(
#             Qux * dudx[..., [i]].detach(), ux,
#             grad_outputs=torch.ones_like(Qux),
#             retain_graph=True,
#             create_graph=True,
#         )[0]
#         if last_dim_scalar:
#             dQudxi = dQudxi.unsqueeze(-1)
#         dQudx.append(dQudxi)
#     dQudx = torch.cat(dQudx, -1)
#
#     # if is_Q_onet:
#     #     grad_x = grad(
#     #         Qux, x_,
#     #         grad_outputs=torch.ones_like(Qux),
#     #         retain_graph=True,
#     #         create_graph=True,
#     #     )[0]
#     #     dQudx = grad_x
#
#     # computing jacobian of Q wrt output ux
#     # # here there is a bug
#     # jacQ = jacobian(
#     #     cont_layer, ux,
#     #     create_graph=True,
#     # )
#     # j = jacQ.squeeze(5).squeeze(2).reshape((jacQ.shape[0] * jacQ.shape[1], jacQ.shape[2]* jacQ.shape[3]))
#     # j = torch.diagonal(j, dim1=2, dim2=3)
#     # J = j.reshape((jacQ.shape[0], jacQ.shape[1]))
#     # dQdu = grad(
#     #         Qux, ux,
#     #         grad_outputs=torch.ones_like(Qux),
#     #         retain_graph=True,
#     #         create_graph=True,
#     #     )[0]
#
#     # approximation with finite difference
#     eps = 1e-3
#     # if is_Q_onet:
#     #     dQdu = (Q(ux + eps, x) - Qux) / eps
#     # else:
#     if Quxpeps is None:
#         Quxpeps = Q(ux + eps)
#
#     if last_dim_scalar:
#         Quxpeps = Quxpeps.unsqueeze(-1)
#
#     dQdu = (Quxpeps - Qux) / eps
#     dQudx = torch.cat([dQudx, dQdu], -1)
#     vQ = v(x, Qux).detach()
#     vQu = torch.einsum('...i,...i->...', dQudx, vQ).unsqueeze(-1)
#
#     vQ * Qux
#
#
#     # instead of this, which does not work
#     # because implicitly we are taking the derivative of
#     # Qux.sum() wrt x and this mixes things up
#     # vQu = change_along_flow(Qux, vx.detach(), x)
#
#     dQvu = grad(
#         Qux * vux.detach(), ux,
#         grad_outputs=torch.ones_like(Qux),
#         retain_graph=True,
#         create_graph=True,
#     )[0]
#
#     if last_dim_scalar:
#         vQu = vQu.squeeze(-1)
#
#     return dQvu - vQu, dQvu, -vQu


def equiv_crit_full_jac(Q, u, v, x, dudx=None, ux=None, Qux=None, Quxpeps=None, last_dim_scalar=False):
    '''
    TODO: make v a basis of the Lie Algebra and not just a
    single vector field.
    equivariance criterion, taking into account independant
     and dependant variables
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
    if ux is None:
        ux = u(x)

    if last_dim_scalar:
        ux = ux.unsqueeze(-1)

    vx = v(x, ux).detach()
    if dudx is None:
        dudx = grad(
            ux, x,
            grad_outputs=torch.ones_like(ux),
            retain_graph=True,
            create_graph=True,
        )[0]
    dudu = torch.ones_like(dudx[..., [0]], device=dudx.device)
    dudxu = torch.cat([dudx, dudu], -1)
    vux = torch.einsum('...i,...i->...', dudxu, vx).unsqueeze(-1)

    # if is_Q_onet:
    #     x_ = x.clone().detach()
    #     x_.requires_grad_(True)
    #     Qux = Q(ux, x_)
    # else:
    #     Qux = Q(ux)
    if last_dim_scalar:
        ux = ux.squeeze(-1)

    # dQdu = jacobian(
    #     Q, ux,
    #     create_graph=True,
    # )

    # _, vjp_fn = vjp(gaussian_filter, ux, torch.ones(Qux.shape))
    ux = ux.cuda()
    from time import time
    t0 = time()
    Qux, tangent_out = jvp(Q, (ux,), (torch.ones(ux.shape, device=ux.device),))
    print('time to compute jvp', time() - t0)

    from time import time
    t0 = time()
    Qux, tangent_out = jpv_functorch(Q, (ux,), (torch.ones(ux.shape, device=ux.device),))
    print('time to compute jpv_functorch', time() - t0)
    # unit_vector = torch.ones(Qux.shape)
    # vjp_value = vjp_fn(torch.ones(Qux.shape))
    t0 = time()
    _, vjp_fn = vjp(Q, ux)
    out = vjp_fn(torch.ones(Qux.shape))
    print('time to compute vjp', time() - t0)


    # jvp_value = jvp_fn(torch.ones(ux.shape))
    # ft_jacobian, = vmap(vjp_fn)(unit_vectors)

    Q_reduced = lambda ux, batch_dim=0: Q(ux).sum(batch_dim)
    dQdu_reduced = jacobian(
        Q_reduced, ux,
        vectorize=True,
        create_graph=True,
    )

    if last_dim_scalar:
        Qux = Qux.unsqueeze(-1)

    # this: vQu = <dQudx, vx> where
    # dQudx = Jac_u(Q).T . dudx = grad_x( Qux * dudx )
    # dudx = grad(
    #     ux, x,
    #     grad_outputs=torch.ones_like(Qux),
    #     retain_graph=True,
    #     create_graph=True,
    # )[0]

    # computing derivative wrt coordinates
    dQudx = []
    for i in range(dudx.shape[-1]):
        dQudxi = grad(
            Qux * dudx[..., [i]].detach(), ux,
            grad_outputs=torch.ones_like(Qux),
            retain_graph=True,
            create_graph=True,
        )[0]
        if last_dim_scalar:
            dQudxi = dQudxi.unsqueeze(-1)
        dQudx.append(dQudxi)
    dQudx = torch.cat(dQudx, -1)

    # if is_Q_onet:
    #     grad_x = grad(
    #         Qux, x_,
    #         grad_outputs=torch.ones_like(Qux),
    #         retain_graph=True,
    #         create_graph=True,
    #     )[0]
    #     dQudx = grad_x

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
    # if is_Q_onet:
    #     dQdu = (Q(ux + eps, x) - Qux) / eps
    # else:
    if Quxpeps is None:
        Quxpeps = Q(ux + eps)

    if last_dim_scalar:
        Quxpeps = Quxpeps.unsqueeze(-1)

    dQdu = (Quxpeps - Qux) / eps
    dQudx = torch.cat([dQudx, dQdu], -1)
    vQ = v(x, Qux).detach()
    vQu = torch.einsum('...i,...i->...', dQudx, vQ).unsqueeze(-1)

    vQ * Qux


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

    if last_dim_scalar:
        vQu = vQu.squeeze(-1)

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

def finite_diff_grid_2d(ux, dx, last_dim_scalar=False):
    dudx1_fd = (ux[..., :, 1:] - ux[..., :, :-1]) / dx
    dudx2_fd = (ux[..., 1:, :] - ux[..., :-1, :]) / dx

    if last_dim_scalar:
        dudx1_fd = dudx1_fd.unsqueeze(-1)
        dudx2_fd = dudx2_fd.unsqueeze(-1)

    pad1 = torch.zeros_like(dudx1_fd[..., :, [0], :], device=dudx1_fd.device)
    dudx1_fd = torch.cat([dudx1_fd, pad1], -2)
    pad2 = torch.zeros_like(dudx1_fd[..., [0], :, :], device=dudx2_fd.device)
    dudx2_fd = torch.cat([dudx2_fd, pad2], -3)
    dudx = torch.cat([dudx1_fd, dudx2_fd], -1)
    return dudx

def run_2d_test():
    n = 1000
    batch_dim = True
    # method_to_compute_dudx =  'finite_differences_on_grid'
    # method_to_compute_dudx =  'finite_differences_with_function'
    # method_to_compute_dudx =  'autograd'
    method_to_compute_dudx = 'none'
    last_dim_scalar = False

    x_ = torch.linspace(-math.pi, math.pi, n)
    y_ = torch.linspace(-math.pi, math.pi, n)
    x_grid, y_grid = torch.meshgrid(x_, y_)
    x = torch.stack([y_grid, x_grid], -1)

    def ring(x, last_dim_scalar=False):
        center = [0, 1]
        r = (x[..., 0] - center[0]) ** 2 + (x[..., 1] - center[1]) ** 2
        scale = .6
        out = torch.exp(-(r - 3) ** 2 / scale).unsqueeze(-1)
        if last_dim_scalar:
            out = out.squeeze(-1)
        return out

    def lines(x, last_dim_scalar=False):
        r = x[..., 1] ** 2
        scale = .6
        lines = torch.exp(-(r - 3) ** 2 / scale).unsqueeze(-1)
        # mask = torch.ones_like(lines)
        # mask[torch.abs(x[..., 0]) > 1] = 0
        # mask[torch.abs(x[..., 1]) > 2] = 0
        # compact_lines = mask * lines
        # return compact_lines
        if last_dim_scalar:
            lines = lines.squeeze(-1)

        return lines

    u0 = partial(lines, last_dim_scalar=last_dim_scalar)

    # center = [0, 0]
    # r = (x[..., 0] - center[0]) ** 2 + (x[..., 1] - center[1]) ** 2
    # scale = .06
    # gauss = torch.exp(-r  ** 2 / scale).unsqueeze(-1)
    # gauss_filter = partial(conv_2d_fft, filter=gauss)
    if batch_dim:
        x = x.unsqueeze(0)

    # gauss_filter = partial(gauss_conv_2d, last_dim_scalar=last_dim_scalar)
    gauss_filter = conv_2d_filter_given(channels=1, last_dim_scalar=last_dim_scalar)
    u0x = u0(x)


    # eps_x1 = eps * torch.cat([torch.ones_like(x[..., [0]]),
    #                           torch.zeros_like(x[..., [1]])], -1)
    # eps_x2 = eps * torch.cat([torch.zeros_like(x[..., [0]]),
    #                           torch.ones_like(x[..., [1]])], -1)

    if method_to_compute_dudx == 'finite_differences_on_grid':
        dx = 2 * math.pi / n
        du0dx = finite_diff_grid_2d(u0x, dx, last_dim_scalar=last_dim_scalar)
    elif method_to_compute_dudx == 'autograd':
        x.requires_grad_(True)
        du0dx = grad(
            u0x, x,
            grad_outputs=torch.ones_like(u0x),
            retain_graph=True,
            create_graph=True,
        )[0]
    elif method_to_compute_dudx == 'finite_differences_with_function':
        assert u0x.shape[-1] == 1
        eps = 1e-4
        eps_xs = torch.zeros((x.shape[-1], *x.shape), device=x.device)
        for i in range(eps_xs.shape[0]):
            eps_xs[i, ..., i] = eps
        du0dx_fd_ = (u0(x.unsqueeze(0) + eps_xs) - u0x.unsqueeze(0)) / eps
        du0dx = torch.swapaxes(du0dx_fd_, 0, -1).squeeze(0)
        # du0dx1_fd = (u0(x + eps_xs[0]) - u0x) / eps
        # du0dx2_fd = (u0(x + eps_xs[1]) - u0x) / eps
    elif method_to_compute_dudx == 'none':
        du0dx = None

    if method_to_compute_dudx != 'none':
        j = 1
        plt.title('method_to_compute_dudx')
        plt.imshow(du0dx[..., j].squeeze().detach().cpu().numpy())
        plt.colorbar()
        plt.show()

    y = gauss_filter(u0x.cuda())
    # y = cont_layer_onet(u0(x), x)
    e, dQvu, mvQu = equiv_crit_full_jac(gauss_filter, u0,
                                      # partial(v_translation, coords_to_translate='x'),
                                      v_rotation,
                                      x, dudx=None, last_dim_scalar=last_dim_scalar)
    # e, dQvu, mvQu = equiv_crit(gauss_filter, u0,
    #                                   # partial(v_translation, coords_to_translate='x'),
    #                                   v_rotation,
    #                                   x, dudx=du0dx, last_dim_scalar=last_dim_scalar)
    # e, dQvu, mvQu = equiv_crit(gauss_filter, u0, v_rotation, x)
    # e, dQvu, mvQu = equiv_crit(gauss_filter, u0, v_scale, x)
    # e, dQvu, mvQu = equiv_crit(gauss_filter, u0, v_galilean_boost, x)
    # e, dQvu, mvQu = equiv_crit(cont_layer_onet, u0, v_translation, x, is_Q_onet=True)

    eps = .2
    gQu = y - eps * mvQu
    Qgu = y + eps * dQvu
    # u0x = u0(x)

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