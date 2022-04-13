import torch
import torch.nn as nn

from pbc_examples.net_pbc import DNN


class Gating(nn.Module):
    def __init__(self, gated_funcs, dims):
        super().__init__()
        self.gated_funcs = gated_funcs
        self.gates = [DNN(d) for d in dims]

    def forward(self, input, gate_input):
        out = None
        for g_func, s_func in zip(self.gated_funcs, self.gates):
            s = s_func(gate_input)
            out = s * g_func(input) + (1 - s) * input
        return out