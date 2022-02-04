import math
from functools import partial
import torch
import torch.nn as nn

# from modules import torch.nn.Linear



def phase_init(m, bandlimit):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            # num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-bandlimit, bandlimit)
        if hasattr(m, 'bias'):
            c = 50
            m.bias.uniform_(-math.pi * c, math.pi * c)


def lin_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-math.sqrt(6 / num_input), math.sqrt(6 / num_input))


def get_rot_mat(theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack([torch.stack([cos, -sin], -1),
                        torch.stack([cos, sin], -1)], -2)


class WaveletFeatures(nn.Module):
    def __init__(self, dh, coord_dim, bandlimit):
        super().__init__()
        assert coord_dim == 2
        self.dh = dh
        self.theta = nn.Parameter(torch.zeros((self.dh,),),)
        self.sigma = nn.Parameter(torch.zeros((1, self.dh, 1)))
        self.lamda = nn.Parameter(torch.zeros((1, self.dh, 1)))
        self.psi = nn.Parameter(torch.zeros((1, self.dh, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, self.dh, 1)))
        self.shift = nn.Parameter(torch.zeros((1, self.dh, 1, 2))) # (b, dh, res, dim)
        with torch.no_grad():
            self.theta.uniform_(-math.pi, math.pi)
            self.gamma.uniform_(.5, 1)
            # self.sigma.uniform_(1.1, 2.1)
            # self.lamda.uniform_(.01 * math.pi, .1 * math.pi)
            self.sigma.uniform_(3.01, 15.02)
            self.lamda.uniform_(1.02 * math.pi, 2. * math.pi)
            # import torch.distributions as d
            # b = d.Beta(torch.tensor([2.]), torch.tensor([4.]))
            # lamda = b.sample(sample_shape=(1, self.dh, 1))
            # self.lamda = nn.Parameter(lamda.squeeze(-1))
            self.psi.uniform_(-math.pi, math.pi)
            self.shift.uniform_(0, 4*math.pi)

    def forward(self, x): # x: (b, res**2, coord_dim)
        x = x.unsqueeze(0)
        sigma_x = self.sigma
        sigma_y = self.sigma / (self.gamma + 1e-7)
        xf = x.unsqueeze(1) - self.shift # (b, dh, res**2, dim)
        rotmat = get_rot_mat(self.theta) # (dh, 2, 2)
        coord_theta = torch.matmul(rotmat.unsqueeze(0).unsqueeze(2), xf.unsqueeze(-1)).squeeze(-1) #x: (b, dh, res**2, dim) coord_theta: (dh, res**2, 2)
        x_theta, y_theta = coord_theta[..., 0], coord_theta[..., 1] # (b, dh, res**2)
        # gb : should be (b, dh, res**2)
        gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * \
             torch.cos(2 * math.pi / self.lamda * x_theta + self.psi)
        return gb.squeeze(0)


class FourierFeatureBand(nn.Module):
    def __init__(self, dh, coord_dim, bandlimit):
        super().__init__()
        self.dh = dh
        self.lin = torch.nn.Linear(coord_dim, self.dh, bias=True)
        self.lin.apply(partial(phase_init, bandlimit=bandlimit))

    def forward(self, x):
        return torch.sin(self.lin(x))


class BaconBlock(nn.Module):
    def __init__(self, dh, coord_dim, bandlimit, is_first):
        super().__init__()
        self.is_first = is_first
        self.dh = dh
        if not self.is_first:
            self.lin = torch.nn.Linear(dh, dh, bias=True)
            self.lin.apply(lin_init)
        # self.coord_feats = WaveletFeatures(dh, coord_dim, bandlimit)
        self.coord_feats = FourierFeatureBand(dh, coord_dim, bandlimit)

    def forward(self, z, x):
        assert z is None and self.is_first or \
               z is not None and not self.is_first

        if not self.is_first:
            wz = self.lin(z)
            # wz = wz / self.dh
        else:
            wz = 1

        gi = self.coord_feats(x)
        if isinstance(self.coord_feats, WaveletFeatures):
            gi = gi.permute(1, 0) # (b,  res**2, dh)
        return wz * gi


class Bacon(nn.Module):
    def __init__(self, dh=128, dout=3, nblocks=2, coord_dim=2):
        super().__init__()
        # d = 256
        d = .1
        self.blocks = nn.ModuleList([
            BaconBlock(dh, coord_dim, d * math.sqrt(coord_dim), i == 0)
            for i in range(nblocks)])
        self.lin = torch.nn.Linear(dh, dout)

    def _compute_last_block_index(self, iteration):
        every = 500
        # print(iteration, every, iteration//every)
        return iteration // every

    def forward(self, model_input):
        # Enables us to compute gradients w.r.t. coordinates
        # coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        # coords = coords_org
        x = model_input
        z = None
        for i, b in enumerate(self.blocks):
            if z is None:
                r = 0
            else:
                r = z
            z = r + b(z, x)
            # print('ok')
            if self.iter is not None and\
                    i > self._compute_last_block_index(self.iter):
                # print('bprke at ',  i, self._compute_last_block_index(self.iter), self.iter)
                break
        out = self.lin(z)
        return out