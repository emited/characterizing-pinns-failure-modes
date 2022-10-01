import functools

import math

import torch
import functorch as ft
from functorch import vmap
from torch import nn

from LPSDA.common.augmentation import SpaceTranslate, fourier_shift, Galileo
from pbc_examples.equiv_crit_jvp import ring, equiv_crit_fast, equiv_crit_full_jac, plot, plot_g
from pbc_examples.fourier_continuous_test import v_translation, v_rotation, v_galilean_boost, v_galilean_boost_kdv
from pbc_examples.resnet import Generator, WrapperSwapAxes


class Model(nn.Module):
    def __init__(self, is_target=False):
        super().__init__()
        kernel_size = 3
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        if is_target:
            self.model = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            )
        else:
            # self.model = nn.Sequential(
            #     nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            #     nn.GELU(),
            #     nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, padding_mode='replicate'),
            # ).cuda()
            self.model = Generator(1, 64, 1, 6)

        self.model = WrapperSwapAxes(self.model).cuda()


        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.orthogonal_(m.weight, gain=1.)
                torch.nn.init.xavier_uniform_(m.weight, gain=1.)
                # m.weight.data.fill_(1 / kernel_size ** 2)
                gain = 1 / kernel_size ** 2
                eps = 10.1 * gain
                torch.nn.init.uniform_(m.weight.data, a=gain - eps, b=gain + eps)
                # m.bias.data.fill_(0.)
                # gain = 0
                # torch.nn.init.uniform_(m.bias.data, a=-gain, b=gain)
        if is_target:
            self.model.apply(init_weights)

    def forward(self, input):
        # input = torch.swapaxes(input, -1, 1)
        output = self.model(input)
        # output = torch.swapaxes(output, -1, 1)
        return output


def make_group_transform(name, *args, **kwargs):
    if name == 'space_translate':
        g = SpaceTranslate(*args, **kwargs)
    elif name == 'galileo_kdv':
        g = Galileo(*args, **kwargs)
    else:
        raise NotImplementedError(name)
    def group_transform(ux, x, eps):
        assert ux.shape[-1] == 1
        # (t, x, c=1) -> (t, x)
        x = x.squeeze(-1)
        ux = ux.squeeze(-1)
        # (t, x) -> (t, x, d=1)
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
    torch.manual_seed(11)
    # strategy = 'full'
    strategy = 'fast'
    # group_transform_name = 'space_translate'
    group_transform_name = 'galileo_kdv'
    group_transform = make_group_transform(group_transform_name)
    if group_transform_name == 'space_translate':
        v_group_transform = functools.partial(v_translation, coords_to_translate=[0])
    elif group_transform_name == 'galileo_kdv':
        v_group_transform = v_galilean_boost_kdv
    equiv_crit = equiv_crit_fast if strategy == 'fast' else equiv_crit_full_jac

    batch_size = 1
    n = 80

    batch_dim = True
    method_to_compute_dudx = 'none'
    last_dim_scalar = False
    device = 'cuda'
    with torch.no_grad():
        x_ = torch.linspace(-math.pi, math.pi, n)
        y_ = torch.linspace(-math.pi, math.pi, n)
        x_grid, y_grid = torch.meshgrid(x_, y_)
        x = torch.stack([y_grid, x_grid], -1).to(device)
        if batch_dim:
            x = x.unsqueeze(0)

        x = torch.cat([x.clone() for _ in range(batch_size)], 0)
        u0 = functools.partial(ring, last_dim_scalar=last_dim_scalar)
        u0x = u0(x)
        Qu0x_target = Model(is_target=True)(u0x.cuda()).detach() # using another model as target
        # model = Model()
        # input_dim, num_filter, output_dim, num_resnet
        model = Model()
        Qu0x = model(u0x.cuda())
        (epss, err), (Qu0, gQu0, gu0, Qgu0) = error(model, u0x, x, group_transform, extent=0.05)
        # import matplotlib.pyplot as plt
        # plt.plot(epss.detach().cpu().numpy(), err.detach().cpu().numpy())
        # plt.show()
        i, j = 0, -1
        plot_g(u0x[[i]], gu0[[i]], Qgu0[[i]], Qu0x[[i]], gQu0[[i]], title='first')
        plot_g(u0x[[j]], gu0[[j]], Qgu0[[j]], Qu0x[[j]], gQu0[[j]], title='last')
        # e, dQvu, mvQu = equiv_crit(model, u0, v_group_transform, x)
        # plot(e, mvQu, dQvu, u0x, Qu0x, batch_dim=batch_dim, eps=0.02)

    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=0.0002)
    nepochs = 1000
    for epoch in range(nepochs):
        with torch.enable_grad():
            optimizer.zero_grad()
            # output = model(u0x)
            e, dQvu, mvQu, Qux = equiv_crit(model, u0, v_group_transform, x)
            equiv_loss = (e ** 2).mean()
            # l = 0
            regression_loss = ((Qu0x_target - Qux) ** 2).mean()
            l = 0 * equiv_loss + regression_loss
            print(f'equiv_loss: {equiv_loss.item()}, regress: {regression_loss.item()}')
            l.backward()
            optimizer.step()
        with torch.no_grad():
            if epoch % 100 == 0:
                Qu0x = model(u0x.cuda())
                (epss, err), (Qu0, gQu0, gu0, Qgu0) = error(model, u0x, x, group_transform, extent=0.05)
                plot(e, mvQu, dQvu, u0x, Qu0x, Qux_target=Qu0x_target, batch_dim=batch_dim, eps=0.05)
