"""
Approximation of the diagonal of the jacobian of a function f: x -> f(x); i.e. diag^{-1](Jac_x(f)).
Makes use of the fact that diag^{-1](A) = diag^{-1}(A^T) =E_v[v elementprod A^T.v],
if the components of v are uncorrelated. Replacing A by  Jac_x(f), as Jac_x(f)^T.v can be computed
efficiently using reverse mode auto-differentiation, we do not have to explicit the full jacobian,
making the approximation computable in linear time, instead of quadratic.
"""

import torch
from functorch import grad, vjp, vmap
from torch import nn
from torch.autograd.functional import jacobian
import numpy as np


def diag_jac(f, x):
    jac = jacobian(
            f, x,
            create_graph=True,
    )
    return torch.diag(jac, 0)


def diag_jac_approx(f, x, nsamples=10, timeit=False):
    y, vjp_fn = vjp(f, x)
    vs = torch.randint(0, 2, size=(nsamples, *y.shape), device=y.device) * 2 - 1

    # # using for loop
    # if timeit:
    #     from time import time
    #     t0 = time()
    # total = 0
    # for i in range(nsamples):
    #     v = vs[i]
    #     g = vjp_fn(v)[0]
    #     s = v * g
    #     total += s
    # diag_jac_loop = total / nsamples
    # if timeit:
    #     print(f'Time with for loop: {time() - t0}')

    # using vmap
    if timeit:
        from time import time
        t0 = time()
    gs = vmap(vjp_fn)(vs)[0]
    diag_jac_vmap = torch.mean(gs * vs, 0)
    if timeit:
        print(f'Time with vmap: {time() - t0}')

    # assert torch.allclose(diag_jac_loop, diag_jac_vmap, atol=1e-5)

    return diag_jac_vmap


if __name__ == '__main__':
    # testing here

    in_dim, h_dim, out_dim = 21, 42, 21
    module = nn.Sequential(
        nn.Linear(in_dim, h_dim),
        nn.GELU(),
        nn.Linear(h_dim, h_dim),
        nn.GELU(),
        nn.Linear(h_dim, out_dim),
    )
    module = module.cuda()
    f = lambda x: module(x) * 100
    # f = module

    x = torch.randn(in_dim, device='cuda')
    dj = diag_jac(f, x)

    nrepeats = 100
    nsampless = list(range(1, 5000, 100))
    mean_errors, std_errors = [], []
    for nsamples in nsampless:
        current_exp_errors = []
        for e in range(nrepeats):
            dj_approx = diag_jac_approx(f, x, nsamples=nsamples, timeit=True)
            error = torch.abs(dj_approx - dj).mean() / (dj.abs().mean() + 1e-8)
            # error = torch.abs(dj_approx - dj).mean()
            current_exp_errors.append(error.item())

        mean_errors.append(np.mean(current_exp_errors))
        std_errors.append(np.std(current_exp_errors))

    import matplotlib.pyplot as plt
    plt.title('Relative error of approximation of diagonal components of Jacobian')
    plt.errorbar(nsampless, mean_errors, yerr=std_errors)
    plt.xlabel('Number of samples')
    plt.show()