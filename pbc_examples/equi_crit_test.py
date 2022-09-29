import functools

import math

import torch
import functorch as ft
from functorch import vmap
from torch import nn

from LPSDA.common.augmentation import SpaceTranslate, fourier_shift, Galileo
from pbc_examples.equiv_crit_jvp import ring, equiv_crit_fast, equiv_crit_full_jac, plot, plot_g
from pbc_examples.fourier_continuous_test import v_translation, v_rotation, v_galilean_boost

# primal0 = torch.randn(10, 10)
# tangent0 = torch.randn(10, 10)
# primal1 = torch.randn(10, 10)
# tangent1 = torch.randn(10, 10)
#
# def fn(x, y):
#     return x ** 2 + y ** 2
#
# # Here is a basic example to compute the JVP of the above function.
# # The jvp(func, primals, tangents) returns func(*primals) as well as the
# # computed jvp. Each primal must be associated with a tangent of the same shape.
# primal_out, tangent_out = ft.jvp(fn, (primal0, primal1), (tangent0, tangent1))
#
# # functorch.jvp requires every primal to be associated with a tangent.
# # If we only want to associate certain inputs to `fn` with tangents,
# # then we'll need to create a new function that captures inputs without tangents:
# primal = torch.randn(10, 10)
# tangent = torch.randn(10, 10)
# y = torch.randn(10, 10)
#
# import functools
# new_fn = functools.partial(fn, y=y)
# primal_out, tangent_out = ft.jvp(new_fn, (primal,), (tangent,))
#
# from functorch import jvp
# x = torch.randn(5)
# y = torch.randn(5)
#
# import torch.nn as nn
# f = lambda x, y: (x * y)
# _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
# assert torch.allclose(output, x + y)
# lin = nn.Linear(1, 1)
# input = torch.randn(2, 1)
# print('output', lin(input))
# tangent = torch.ones(2, 1)


# from torch.optim import Adam
# optimizer = Adam(lin.parameters())
# with torch.enable_grad():
#     optimizer.zero_grad()
#     output = lin(input)
#     l = output.sum()
#     l.backward()
#     optimizer.step()

torch.ones(1)

eps = 0.05

# def space_translate(u, x, eps, dim=-2):
#     gu = fourier_shift(u, eps=eps, dim=dim)
#     # aug = SpaceTranslate(eps)
#     # gdata = aug((data, data_coords), eps)[0]
#     return gu


def make_group_transform(name, *args, **kwargs):
    if name == 'space_translate':
        g = SpaceTranslate(*args, **kwargs)
    elif name == 'galileo':
        g = Galileo(*args, **kwargs)
    else:
        raise NotImplementedError(name)
    def group_transform(ux, x, eps):
        assert ux.shape[-1] == 1
        # (B, t, x, d=1) -> (B, t, x)
        x = x.squeeze(-1)
        ux = ux.squeeze(-1)
        # (B, t, x) -> (B, t, x, d=1)
        g_batched = vmap(g, in_dims=((0, 0), None))
        gux = g_batched((ux.cpu(), x.cpu()), eps)[0].to('cuda').unsqueeze(-1)
        return gux
    return group_transform


# x: axis: epsilon of group transform
# y axis: error

def norm(x):
    xp = torch.pow(x, 2)
    for dim in reversed(range(1, x.dim())):
        xp = torch.sum(xp, dim)
    return xp


def error(Q, u0x, x, g, extent: float = 1., n: int = 10):
    epss = torch.linspace(-extent, extent, n)
    g_batched = vmap(g, in_dims=(None, None, 0))
    gu0 = g_batched(u0x, x, epss).squeeze(1)
    Qgu0 = Q(gu0)
    Qu0 = Q(u0x)
    gQu0 = g_batched(Qu0, x, epss).squeeze(1)
    Q_err = norm(Qgu0 - gQu0) / norm(gQu0)
    return (epss, Q_err), (Qu0, gQu0, gu0, Qgu0)


if __name__ == '__main__':
    # strategy = 'full'
    strategy = 'fast'
    group_transform_name = 'space_translate'
    # group_transform_name = 'galileo'
    group_transform = make_group_transform(group_transform_name)
    if group_transform_name == 'space_translate':
        v_group_transform = functools.partial(v_translation, coords_to_translate=[0])
    elif group_transform_name == 'galileo':
        v_group_transform = v_galilean_boost
    equiv_crit = equiv_crit_fast if strategy == 'fast' else equiv_crit_full_jac

    batch_size = 1
    n = 80
    # v_coeff_fn = partial(v_translation, coords_to_translate='u')
    # profile_varying_batchsize(v_coeff_fn, strategy)
    # profile_varying_gridsize(v_coeff_fn, strategy)
    # v_coeff_fn = v_rotation

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            kernel_size = 11
            self.kernel_size = kernel_size
            padding = (kernel_size - 1) // 2
            self.model = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            ).cuda()

            def init_weights(m):
                if isinstance(m, nn.Conv2d):
                    # torch.nn.init.orthogonal_(m.weight, gain=1.)
                    torch.nn.init.xavier_uniform_(m.weight, gain=1.)
                    m.weight.data.fill_(1 / kernel_size ** 2)
                    # m.bias.data.fill_(0.)
                    # gain = 0
                    # torch.nn.init.uniform_(m.bias.data, a=-gain, b=gain)
            self.model.apply(init_weights)

        def forward(self, input):
            input = torch.swapaxes(input, -1, 1)
            output = self.model(input)
            output = torch.swapaxes(output, -1, 1)
            return output

    batch_dim = True
    method_to_compute_dudx = 'none'
    last_dim_scalar = False
    device = 'cuda'

    x_ = torch.linspace(-math.pi, math.pi, n)
    y_ = torch.linspace(-math.pi, math.pi, n)
    x_grid, y_grid = torch.meshgrid(x_, y_)
    x = torch.stack([y_grid, x_grid], -1).to(device)
    if batch_dim:
        x = x.unsqueeze(0)

    x = torch.cat([x.clone() for _ in range(batch_size)], 0)
    u0 = functools.partial(ring, last_dim_scalar=last_dim_scalar)
    u0x = u0(x)
    # def space_translate_cuda(u0x, x, eps):
    #     return space_translate(u0x.cpu(), x.cpu(), eps).to('cuda')
    #
    # gu0 = space_translate_cuda(u0x, x, torch.tensor(eps))
    model = Model()
    Qu0x = model(u0x.cuda())
    (epss, err), (Qu0, gQu0, gu0, Qgu0) = error(model, u0x, x, group_transform, extent=0.1)

    import matplotlib.pyplot as plt
    plt.plot(epss.detach().cpu().numpy(), err.detach().cpu().numpy())
    plt.show()

    i, j = 0, -1
    plot_g(u0x[[i]], gu0[[i]], Qgu0[[i]], Qu0x[[i]], gQu0[[i]], title='first')
    plot_g(u0x[[j]], gu0[[j]], Qgu0[[j]], Qu0x[[j]], gQu0[[j]], title='last')

    e, dQvu, mvQu = equiv_crit(model, u0, v_group_transform, x)

    plot(e, mvQu, dQvu, u0x, Qu0x, batch_dim=batch_dim, eps=0.5)
    # import matplotlib.pyplot as plt
    # plt.imshow(e.detach().cpu().numpy().squeeze())
    # plt.show()

    # (e_fast, dQvu_fast, mvQu_fast), t_f_fast = run_2d_test(
    #     v_coeff_fn, n=n, batch_size=batch_size, strategy='fast', do_plot=True
    # )

    # print('allclose(e, e_fast)', torch.allclose(e, e_fast))
    # print('max(abs(e - e_fast))', (e - e_fast).abs().max().item())
