import torch

from pbc_examples.modules.nn_symmetries import ModulatedLinear, ModulatedSymmetryNet

# from pbc_examples.modules.separation_param_simple_latents import FactorizedMultiplicativeModulation as FMM

if __name__ == '__main__':
    # in_features, out_features, in_mod_features, rank, bias = True,):
    # lin = FMM(2, 3, 3, 3)
    # input = torch.randn(5, 2)
    # mod = torch.randn(5, 3)
    # print(lin(input, mod).shape)
    #
    # mlin = ModulatedLinear(2, 3, bias=True)
    # input = torch.ones(1, 2)
    # style = torch.ones(1, 2) * 2
    # print(mlin(input, style))

    coord_dim = 2
    z_dim = 3
    B = 5
    net = ModulatedSymmetryNet(coord_dim=coord_dim, z_dim=z_dim, out_dim=1)
    z = torch.randn(B, 7, z_dim)
    coords = torch.randn(B, 7, coord_dim)
    out = net(coords, z)
    print(out.shape)
