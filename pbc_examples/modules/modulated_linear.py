import torch
import torch.nn as nn


def param_linear(input, weight, bias=None):
    '''
    The difference with the other ParamLinear linear in the code is the fact that
    there are no weights that are saved in memory that are not used anyway
    '''
    output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
    if bias is not None:
        output += bias.unsqueeze(-2)
    return output


class ModulatedLinear(nn.Linear):
    """Fully connected version of stylegan2's modulated conv"""
    def __init__(self, in_features, out_features, bias=True):
        super(ModulatedLinear, self).__init__(in_features, out_features, bias=bias)
        self.eps = 1e-8

    def forward(self, input, style, style_bias=None):
        """
        style: (B, I), weight: (O, I), input: (B, I), bias: (B, O)
        output: (B, O)
        """
        assert self.weight.shape[-1] == style.shape[-1]
        # mod_weight: (B, O, I)
        # mod_weight = self.weight.unsqueeze(0) * (1 + style.unsqueeze(-2))
        mod_weight = self.weight.unsqueeze(0) * style.unsqueeze(-2)
        norm_weight = mod_weight.pow(2).sum(-1, keepdim=True)
        demod_weight = mod_weight * torch.rsqrt(norm_weight + self.eps)
        output = torch.einsum('...i, ...ji->...j', input, demod_weight)
        # output = output + self.bias * (1 + style_bias)
        output = output + self.bias * style_bias
        return output