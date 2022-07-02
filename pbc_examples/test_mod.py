import torch

from pbc_examples.modules.nn_symmetries import ModulatedLinear
# from pbc_examples.modules.separation_param_simple_latents import FactorizedMultiplicativeModulation as FMM

if __name__ == '__main__':
    # in_features, out_features, in_mod_features, rank, bias = True,):
    # lin = FMM(2, 3, 3, 3)
    # input = torch.randn(5, 2)
    # mod = torch.randn(5, 3)
    # print(lin(input, mod).shape)

    mlin = ModulatedLinear(2, 3, bias=True)
    input = torch.ones(5, 2)
    style = torch.ones(5, 2) * 2
    print(mlin(input, style).shape)
