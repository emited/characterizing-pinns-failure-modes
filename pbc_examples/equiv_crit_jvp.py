"""
TODO:
- make a filtering operator which is non-symmetric and non-linear for tests ---> OK
- implement jacobian version of equiv crit and compare with jvp version
- implement diagonal approximation of jacobian, and compare with full jacobian computation
"""

import torch
from torch.autograd.functional import jacobian, vjp, jvp
from functorch import jvp as jpv_functorch, jacrev, jacfwd
import math
import matplotlib.pyplot as plt
from torch.autograd import grad
# from torch.autograd.functional import jacobian
from functools import partial
import numpy
from functorch import jacrev

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
from pbc_examples.fourier_continuous_test import conv_2d_filter_given, finite_diff_grid_2d, v_rotation


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



def equiv_crit_full_jac(Q, u, v, x, backend='jacrev', strategy='reverse-mode', compose_jacs=False):
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
        Qur = lambda x: Qr(u(x))
        if not compose_jacs:
            dQdx = jacobian(
                Qur, x,
                create_graph=True,
                strategy=strategy,
            )
        ux = u(x)
        Qux = Q(ux)
        dQdu = jacobian(
            Qr, ux,
            create_graph=True,
            strategy=strategy,
        )
    elif backend == 'functorch':
        if strategy == 'reverse-mode':
            jac_func = jacrev
        elif strategy == 'forward-mode':
            jac_func = jacfwd
        def Qr(u):
            Qu = Q(u)
            # we reduce the output, assuming the computations
            # are independant for every sample in the batch
            return Qu.sum(0), (Qu, u)

        Qur = lambda x: Qr(u(x))
        Qur_unb = lambda *x_unb: Qur(torch.stack(x_unb, -1))

        x_unbindded = x.unbind(-1)
        argnums = tuple(range(len(x_unbindded)))
        for x_ in x_unbindded:
            x_.requires_grad_(True)
        if not compose_jacs:
            dQdx_tuple, (Qux, ux) = jac_func(Qur_unb, has_aux=True, argnums=argnums)(*x_unbindded)
            dQdx = torch.stack(dQdx_tuple, -1)
        dQdu, (Qux, ux) = jac_func(Qr, has_aux=True)(ux)

    # computing dudx
    if backend == 'torch':
        dudx = grad(
            ux, x,
            grad_outputs=torch.ones_like(ux),
            retain_graph=True,
            create_graph=True,
        )[0]
    elif backend == 'functorch':
        ux, dudx = vjp(u, x, torch.ones_like(ux), create_graph=True)
    # assert torch.allclose(dudx, dudx_)

    # selecting diagonals
    # dQdu_diag: (b, w, h, i), assuming o = 1 (for now)
    dQdu_diag = torch.einsum('whobwhi->bwhoi', dQdu)
    assert dQdu_diag.shape[-2] == 1
    # dQdu_diag = dQdu_diag.squeeze(-2)
    if compose_jacs:
        dQdu_dudx_diag = torch.einsum('bwhoi, bwhi -> bwhoi', dQdu_diag, dudx)
        dQdx_diag = dQdu_dudx_diag
    else:
        dQdx_diag = torch.einsum('whobwhi -> bwhoi', dQdx)

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
    dQvu = torch.einsum('lkobwhi, bwhi -> bwho', dQdu, vux)
    vQ = v(x, Qux).detach()
    # vQu = torch.einsum('...i,...i->...', dQudx, vQ).unsqueeze(-1)
    dQdxu = torch.cat([dQdx_diag, dQdu_diag], -1)
    # TO DO: normally vQ should have shape (b, w, h, o, i) and not (b, w, h, i)
    # quick fix
    vQ = vQ.unsqueeze(-2)
    vQu = torch.einsum('bwhoi, bwhoi-> bwho', dQdxu, vQ)

    return dQvu - vQu, dQvu, -vQu


def run_2d_test():
    n = 60
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
    # gauss_filter = conv_2d_filter_given(channels=1, last_dim_scalar=last_dim_scalar, filter="non-symmetric")
    gauss_filter = conv_2d_filter_given(channels=1, last_dim_scalar=last_dim_scalar, filter="gaussian")
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
                                        v_rotation, x,
                                        backend='torch', compose_jacs=False)
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
    plt.imshow(u0x.squeeze(-1).detach().cpu().numpy(), origin='lower')
    plt.colorbar()
    plt.subplot(4, 2, 2)
    plt.title('ut')
    plt.imshow(y.squeeze(-1).detach().cpu().numpy(), origin='lower')
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
    plt.show()

if __name__ == '__main__':
    # run_1d_test()
    run_2d_test()