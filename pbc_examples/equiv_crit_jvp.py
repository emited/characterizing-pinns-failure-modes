"""
TODO:
- make a filtering operator which is non-symmetric and non-linear for tests ---> OK
- implement jacobian version of equiv crit and compare with jvp version
- implement diagonal approximation of jacobian, and compare with full jacobian computation
    ---> OK but to no use now
"""

import torch
from torch.autograd.functional import jacobian, vjp as vjp_torch, jvp as jvp_torch
from functorch import jvp, jacrev, jacfwd, vjp, grad, vmap
import math
import matplotlib.pyplot as plt
from torch.autograd import grad as grad_torch
# from torch.autograd.functional import jacobian
from functools import partial
import numpy as np
from functorch import jacrev
from time import time

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
from pbc_examples.fourier_continuous_test import conv_2d_filter_given, finite_diff_grid_2d, v_rotation, v_translation


# def equiv_crit_full_jac(Q, u, v, x, dudx=None, ux=None, Qux=None, Quxpeps=None, last_dim_scalar=False):
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
#         dudx = grad_torch(
#             ux, x,
#             grad_outputs=torch.ones_like(ux),
#             retain_graph=True,
#             create_graph=True,
#         )[0]
#     dudu = torch.ones_like(dudx[..., [0]], device=dudx.device)
#     dudxu = torch.cat([dudx, dudu], -1)
#     vux = torch.einsum('...i,...i->...', dudxu, vx).unsqueeze(-1)
#
#     if last_dim_scalar:
#         ux = ux.squeeze(-1)
#
#     # dQdu = jacobian(
#     #     Q, ux,
#     #     create_graph=True,
#     # )
#
#     # _, vjp_fn = vjp(gaussian_filter, ux, torch.ones(Qux.shape))
#     ux = ux.cuda()
#     from time import time
#     t0 = time()
#     Qux, tangent_out = jvp(Q, (ux,), (torch.ones(ux.shape, device=ux.device),))
#     print('time to compute jvp', time() - t0)
#
#     from time import time
#     t0 = time()
#     Qux, tangent_out = jpv_functorch(Q, (ux,), (torch.ones(ux.shape, device=ux.device),))
#     print('time to compute jpv_functorch', time() - t0)
#     # unit_vector = torch.ones(Qux.shape)
#     # vjp_value = vjp_fn(torch.ones(Qux.shape))
#     t0 = time()
#
#
#     # jvp_value = jvp_fn(torch.ones(ux.shape))
#     # ft_jacobian, = vmap(vjp_fn)(unit_vectors)
#
#     Q_reduced = lambda ux, batch_dim=0: Q(ux).sum(batch_dim)
#     dQdu_reduced = jacobian(
#         Q_reduced, ux,
#         vectorize=True,
#         create_graph=True,
#     )
#
#     if last_dim_scalar:
#         Qux = Qux.unsqueeze(-1)
#
#     # this: vQu = <dQudx, vx> where
#     # dQudx = Jac_u(Q).T . dudx = grad_x( Qux * dudx )
#     # dudx = grad_torch(
#     #     ux, x,
#     #     grad_outputs=torch.ones_like(Qux),
#     #     retain_graph=True,
#     #     create_graph=True,
#     # )[0]
#
#     # computing derivative wrt coordinates
#     dQudx = []
#     for i in range(dudx.shape[-1]):
#         dQudxi = grad_torch(
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
#     #     grad_x = grad_torch(
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
#     # dQdu = grad_torch(
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
#     dQvu = grad_torch(
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
from pbc_examples.jdiag_approx import diag_jac_approx


def equiv_crit_full_jac(Q, u, v, x, backend='functorch', strategy='reverse-mode'):
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
    assert backend in ['functorch', 'torch']
    assert strategy in ['reverse-mode', 'forward-mode']

    # x: (b, w, h, d)
    x.requires_grad_(True)

    # computing dQdx
    if backend == 'torch':
        # Qu(x): (b, w, h, 1)
        Qr = lambda ux: Q(ux).sum(0)

        ux = u(x)
        Qux = Q(ux)
        dQdu = jacobian(
            Qr, ux,
            create_graph=True,
            strategy=strategy,
        )
        dudx = grad_torch(
            ux, x,
            grad_outputs=torch.ones_like(ux),
            retain_graph=True,
            create_graph=True,
        )[0]

    elif backend == 'functorch':
        if strategy == 'reverse-mode':
            jac_func = jacrev
        elif strategy == 'forward-mode':
            jac_func = jacfwd

        def Qr_aux(u):
            Qu = Q(u)
            # we reduce the output, assuming the computations
            # are independant for every sample in the batch
            return Qu.sum(0), Qu

        def ur_aux(x):
            ux = u(x)
            return ux.sum(), ux

        # ux_shape = (*x.shape[:-1], 1)
        # ux_, dudx_ = vjp_torch(u, x, torch.ones(ux_shape, device=x.device), create_graph=True)
        dudx, ux = grad(ur_aux, has_aux=True)(x)
        assert ux.shape[-1] == 1
        # _, vjpfunc, ux = vjp(ur_aux, x, has_aux=True)
        # ux_shape = (*x.shape[1:-1], 1)
        # dudx, ux = vjpfunc(torch.ones(ux_shape, device=x.device))
        # dudx, ux

        dQdu, Qux = jac_func(Qr_aux, has_aux=True)(ux)

    # selecting diagonals
    # dQdu_diag: (b, w, h, i), assuming o = 1 (for now)
    dQdu_diag_test = torch.einsum('whobwhi -> bwhoi', dQdu)
    # assert dQdu_diag.shape[-2] == 1
    # dQdu_diag = dQdu_diag.squeeze(-2)
    # dQdu_dudx_diag = torch.einsum('bwhoi, bwhi -> bwhoi', dQdu_diag, dudx)
    # dQdu_dudx_diag = (dQdu_diag * dudx.unsqueeze(-2)).squeeze(-1)
    DQu_dudx = torch.einsum('lkobwhi, bwhi -> blkoi', dQdu, dudx)
    dudu = torch.ones_like(ux, device=ux.device)
    DQu_dudu = torch.einsum('lkobwhi, bwhi -> blkoi', dQdu, dudu)
    dQdx_diag = DQu_dudx

    # dQdx_diag = dQdu_diag_test * dudx.unsqueeze(-2)
    dQdu_diag = DQu_dudu

    # assert torch.allclose(dQdu_dudx_diag, dQdx_diag)
    # TO DO: normally vx should have shape (b, w, h, k, i) and not (b, w, h, i)
    # we are basically assuming that u: R^i -> R^1 but  u: R^i -> R^k in the general case
    vx = v(x, ux).detach()
    dudu = torch.ones_like(dudx[..., [0]], device=dudx.device)
    dudxu = torch.cat([dudx, dudu], -1)
    vux = torch.einsum('...i,...i->...', dudxu, vx).unsqueeze(-1)

    # dQdu: (b, w, h, o, b, w, h, i)
    # vux: (b, w, h, i)
    # dQvu: (b, w, h, o)
    # should be this instead of whats on the bottim
    # dQvu = torch.einsum('lkobwhi, bwhi -> blkoi', dQdu, vux)
    dQvu = torch.einsum('lkobwhi, bwhi -> blko', dQdu, vux)
    vQ = v(x, Qux).detach()
    # vQu = torch.einsum('...i,...i->...', dQudx, vQ).unsqueeze(-1)
    dQdxu = torch.cat([dQdx_diag, dQdu_diag], -1)
    # TO DO: normally vQ should have shape (b, w, h, o, i) and not (b, w, h, i)
    # quick fix
    vQ = vQ.unsqueeze(-2)
    vQu = torch.einsum('...i,...i->...', dQdxu, vQ)

    return dQvu - vQu, dQvu, -vQu, Qux



def equiv_crit_fast(Q, u, v_coeff_fn, x):
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

    # x: (b, w, h, d)
    x.requires_grad_(True)

    def ur_aux(x):
        ux = u(x)
        # assume u(x) is in R
        assert ux.shape[-1] == 1
        # here we compute u(x_bwh), that is why we can
        # do the sum, (given u(x_bwh) in R^1)
        return ux.sum(), ux

    def Q_i(u_i):
        return Q(u_i.unsqueeze(0)).squeeze(0)

    def Q_ij(u_ij):
        Qu_ij = Q_i(u_ij.unsqueeze(-1))
        # we reduce the output, assuming the computations
        # are independant for every sample in the batch
        # here the dims wh are not independant, as Q
        # mixed them together
        return Qu_ij.squeeze(-1)

    # ux_shape = (*x.shape[:-1], 1)
    # ux_, dudx_ = vjp_torch(u, x, torch.ones(ux_shape, device=x.device), create_graph=True)
    # dudx: (b, w, h, d)
    # TO DO: make dudx have dims (b, w, h, k, d) with u(x) in R^k, x in R^d
    dudx, ux = grad(ur_aux, has_aux=True)(x)

    # for every coordinate of x
    # dudx: (b, w, h, d)

    # jvp_xs = vmap(jvp, in_dims=(None, (-1,), (-1,)))
    DQu_dudx_ij_fn = jvp
    DQu_dudx_i_fn_ = vmap(DQu_dudx_ij_fn, in_dims=(None, (None,), (-1,)))
    def DQu_dudx_i_fn(*x):
        out, jvp_out = DQu_dudx_i_fn_(*x)
        return torch.swapaxes(out.unsqueeze(-1), 0, -1).squeeze(0),\
               torch.swapaxes(jvp_out.unsqueeze(-1), 0, -1).squeeze(0)
    DQu_dudx_fn = vmap(DQu_dudx_i_fn, in_dims=(None, (0,), (0,)))
    # DQu_dudx: (b, w, h, d)
    dudu = torch.ones_like(ux, device=dudx.device)
    dudxu = torch.cat([dudx, dudu], -1)
    Quxs, DQu_dudxu = DQu_dudx_fn(Q_ij, (ux.squeeze(-1),), (dudxu,))
    Qux = Quxs[..., [0]]
    dQdxu = DQu_dudxu
    dQdxu = dQdxu.unsqueeze(-2)

    vQ_coeff = v_coeff_fn(x, Qux).detach()
    vQ_coeff = vQ_coeff.unsqueeze(-2)
    vQu = torch.einsum('...i,...i->...', dQdxu, vQ_coeff)

    vx_coeff = v_coeff_fn(x, ux).detach()
    dudu = torch.ones_like(dudx[..., [0]], device=dudx.device)
    dudxu = torch.cat([dudx, dudu], -1)
    vux = torch.einsum('...i,...i->...', dudxu, vx_coeff).unsqueeze(-1)

    DQu_vux_fn = vmap(jvp, in_dims=(None, (0,), (0,)))
    Quxx, DQu_vux = DQu_vux_fn(Q_i, (ux,), (vux,))
    dQvu = DQu_vux

    return dQvu - vQu, dQvu, -vQu, Qux


def plot_g(u, gu, Qgu, Qu, gQu, batch_dim=True, title=''):

    if batch_dim:
        u = u[0]
        Qu = Qu[0]
        gQu = gQu[0]
        Qgu = Qgu[0]
        gu = gu[0]

    plt.figure(figsize=(6, 12))
    plt.suptitle(title, fontsize=16)
    plt.subplot(3, 2, 1)
    plt.title('u0')
    plt.imshow(u.squeeze(-1).detach().cpu().numpy(), origin='lower')
    plt.colorbar()
    plt.subplot(3, 2, 2)
    plt.title('Qu')
    plt.imshow(Qu.squeeze(-1).detach().cpu().numpy(), origin='lower')
    plt.colorbar()
    plt.subplot(3, 2, 3)
    plt.title('gu')
    plt.imshow(gu.squeeze(-1).detach().cpu().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(3, 2, 4)
    plt.title('Qgu')
    plt.imshow(Qgu.squeeze(-1).detach().cpu().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(3, 2, 5)
    plt.title('Qu')
    plt.imshow(Qu.squeeze(-1).detach().cpu().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(3, 2, 6)
    plt.title('gQu')
    plt.imshow(gQu.squeeze(-1).detach().cpu().numpy(),  origin='lower')
    plt.colorbar()
    plt.show()


def plot(e, mvQu, dQvu, ux, Qux, Qux_target=None, batch_dim=True, eps=0.2):
    gQu = Qux - eps * mvQu
    Qgu = Qux + eps * dQvu

    if batch_dim:
        ux = ux[0]
        Qux = Qux[0]
        gQu = gQu[0]
        Qgu = Qgu[0]
        mvQu = mvQu[0]
        dQvu = dQvu[0]
        e = e[0]
        if Qux_target is not None:
            Qux_target = Qux_target[0]

    plt.figure(figsize=(6, 12))
    plt.subplot(4, 2, 1)
    plt.title('u0')
    plt.imshow(ux.squeeze(-1).detach().cpu().numpy(), origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 2)
    plt.title('Qu')
    plt.imshow(Qux.squeeze(-1).detach().cpu().numpy(), origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 3)
    plt.title('gQu')
    plt.imshow(gQu.squeeze(-1).detach().cpu().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 4)
    plt.title('Qgu')
    plt.imshow(Qgu.squeeze(-1).detach().cpu().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 5)
    plt.title('vQu')
    plt.imshow(-mvQu.squeeze(-1).detach().cpu().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 6)
    plt.title('dQvu')
    plt.imshow(dQvu.squeeze(-1).detach().cpu().numpy(),  origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 7)
    plt.title('e')
    plt.imshow(e.squeeze(-1).detach().cpu().numpy(), origin='lower')
    plt.colorbar()
    if Qux_target is not None:
        plt.subplot(4, 2, 8)
        plt.title('Qu_target')
        plt.imshow(Qux_target.squeeze(-1).detach().cpu().numpy(), origin='lower')
        plt.colorbar()
    plt.show()


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


def run_2d_test(v_coeff_fn=v_translation, n=100, batch_size=1, strategy='fast', do_plot=False):
    equiv_crit = equiv_crit_fast if strategy == 'fast' else equiv_crit_full_jac
    batch_dim = True
    # method_to_compute_dudx =  'finite_differences_on_grid'
    # method_to_compute_dudx =  'finite_differences_with_function'
    # method_to_compute_dudx =  'autograd'
    method_to_compute_dudx = 'none'
    last_dim_scalar = False
    device = 'cuda'

    x_ = torch.linspace(-math.pi, math.pi, n)
    y_ = torch.linspace(-math.pi, math.pi, n)
    x_grid, y_grid = torch.meshgrid(x_, y_)
    x = torch.stack([y_grid, x_grid], -1).to(device)

    u0 = partial(ring, last_dim_scalar=last_dim_scalar)

    # center = [0, 0]
    # r = (x[..., 0] - center[0]) ** 2 + (x[..., 1] - center[1]) ** 2
    # scale = .06
    # gauss = torch.exp(-r  ** 2 / scale).unsqueeze(-1)
    # gauss_filter = partial(conv_2d_fft, filter=gauss)
    if batch_dim:
        x = x.unsqueeze(0)

    x = torch.cat([x.clone() for _ in range(batch_size)], 0)

    # gauss_filter = partial(gauss_conv_2d, last_dim_scalar=last_dim_scalar)
    # gauss_filter = conv_2d_filter_given(channels=1, last_dim_scalar=last_dim_scalar, filter="non-symmetric")
    gauss_filter = conv_2d_filter_given(channels=1,
                                        last_dim_scalar=last_dim_scalar,
                                        filter="non-symmetric",
                                        # filter="gaussian",
    )
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

    # Qu0x = gauss_filter(u0x.cuda())
    # from time import time
    # for backend in ['torch', 'functorch']:
    #     for strategy in ['forward-mode', 'reverse-mode']:
    #     # for strategy in ['reverse-mode']:
    #         if backend == 'torch' and strategy == 'forward-mode':
    #             continue
    #         t0 = time()
    #         if not(backend == 'functorch' and strategy == 'reverse-mode'):
    #             continue
    #         # e, dQvu, mvQu = equiv_crit_full_jac(gauss_filter, u0,
    #         #                                     v_rotation,
    #         #                                     # v_translation,
    #         #                                     x,
    #         #                                     backend=backend,
    #         #                                     strategy=strategy,
    #         #                                     )
    #         e, dQvu, mvQu = equiv_crit_fast(gauss_filter, u0,
    #                                         # v_rotation,
    #                                         partial(v_translation, coords_to_translate='u'),
    #                                         x,
    #                                         # backend=backend,
    #                                         # strategy=strategy,
    #                                         )
    #         # plot(e, mvQu, dQvu, u0x, Qu0x, batch_dim=batch_dim, eps=0.2)
    #         print(f'backend: {backend},'
    #               f' strategy: {strategy},'
    #               f' time: {time() - t0} seconds')

    Qu0x = gauss_filter(u0x.cuda())
    t0 = time()
    e, dQvu, mvQu = equiv_crit(gauss_filter, u0,
                                    # v_rotation,
                                    v_coeff_fn,
                                    x,
                                    # backend=backend,
                                    # strategy=strategy,
                                    )
    t_f = time() - t0
    # plot(e, mvQu, dQvu, u0x, Qu0x, batch_dim=batch_dim, eps=0.2)
    print(f' time: {t_f} seconds')
    if do_plot:
        plot(e, mvQu, dQvu, u0x, Qu0x, batch_dim=batch_dim, eps=0.5)

    return (e, dQvu, mvQu), t_f


def profile_varying_batchsize(v_coeff_fn, strategy):
    n = 64
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    times = []
    for batch_size in batch_sizes:
        print(f'batch size: {batch_size}')
        try:
            etime = run_2d_test(v_coeff_fn, n, batch_size=batch_size, strategy=strategy)
            times.append(etime)
        except:
            print('Out of mem /!\ ')
            times.append(np.nan)
    times, batch_sizes = np.array(times), np.array(batch_sizes)
    if strategy == 'full':
        strategy_str = 'Full Jacobian'
    elif strategy == 'fast':
        strategy_str = 'Fast JVP'
    plt.title(f'Computational time of Equiv Criteria with {strategy_str}, grid: {n}²')
    plt.ylabel('seconds')
    plt.xlabel('batch size')
    plt.plot(batch_sizes, times)
    plt.scatter(batch_sizes[times!=times],
                np.zeros_like(batch_sizes)[times!=times],
                c='r', label='Out of memory')
    plt.xticks(batch_sizes)
    plt.legend()
    plt.show()


def profile_varying_gridsize(v_coeff_fn, strategy):
    ns = [32, 64, 128, 256, 512, 1024, 2048]
    batch_size = 1
    times = []
    for n in ns:
        print(f'n: {n}')
        try:
            etime = run_2d_test(v_coeff_fn, n, batch_size=batch_size, strategy=strategy)
            times.append(etime)
        except:
            print('Out of mem /!\ ')
            times.append(np.nan)
    times, ns = np.array(times), np.array(ns)
    if strategy == 'full':
        strategy_str = 'Full Jacobian'
    elif strategy == 'fast':
        strategy_str = 'Fast JVP'
    plt.title(f'Computational time of Equiv Criteria with {strategy_str}, batch: {batch_size}')
    plt.ylabel('seconds')
    plt.xlabel('n²')
    ns2 = ns ** 2
    plt.plot(ns2, times)
    plt.scatter(ns2[times!=times],
                np.zeros_like(ns2)[times!=times],
                c='r', label='Out of memory')
    plt.xticks(ticks=ns2, labels=[f'{n}²' for n in ns])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # strategy = 'full'
    strategy = 'fast'
    batch_size = 1
    n = 80
    # v_coeff_fn = partial(v_translation, coords_to_translate='u')
    # profile_varying_batchsize(v_coeff_fn, strategy)
    # profile_varying_gridsize(v_coeff_fn, strategy)
    v_coeff_fn = partial(v_translation, coords_to_translate=[0])
    # v_coeff_fn = v_rotation
    # (e, dQvu, mvQu), t_f = run_2d_test(
    #     v_coeff_fn, n=n, batch_size=batch_size, strategy='full', do_plot=True
    # )

    (e_fast, dQvu_fast, mvQu_fast), t_f_fast = run_2d_test(
        v_coeff_fn, n=n, batch_size=batch_size, strategy='fast', do_plot=True
    )

    # print('allclose(e, e_fast)', torch.allclose(e, e_fast))
    # print('max(abs(e - e_fast))', (e - e_fast).abs().max().item())




