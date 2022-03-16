import torch
import torch.utils.data as data
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import systems_pbc as systems


def gen_simple_pde_data(system, nu, beta, rho, u0_str, xgrid, nt, source, T=1, ):
    if system == 'diffusion':  # just diffusion
        beta = 0.0
        rho = 0.0
    elif system == 'convection':
        nu = 0.0
        rho = 0.0
    elif system == 'rd':  # reaction-diffusion
        beta = 0.0
    elif system == 'reaction':
        nu = 0.0
        beta = 0.0
    elif system == 'wave':
        # simple wave with beta as speed
        nu = 0.0
        rho = 0.0

    ############################
    # Process data
    ############################
    x = np.linspace(0, 2 * np.pi, xgrid, endpoint=False).reshape(-1, 1) # not inclusive
    t = np.linspace(0, T, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data
    XT_grid = np.stack([T, X], axis=-1)

    # remove initial and boundaty data from X_star
    t_noinitial = t[1:]
    # remove boundary at x=0
    x_noboundary = x[1:]
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)

    if 'convection' in system or 'diffusion' in system:
        u_vals = systems.convection_diffusion(u0_str, nu, beta, source, xgrid, nt)
        G = np.full(u_vals.shape, float(source))
    elif 'rd' in system:
        u_vals = systems.reaction_diffusion_discrete_solution(u0_str, nu, rho, xgrid, nt)
        G = np.full(u_vals.shape, float(source))
    elif 'reaction' in system:
        u_vals = systems.reaction_solution(u0_str, rho, xgrid, nt)
        G = np.full(u_vals.shape, float(source))
    elif system == 'wave':
        u_vals = systems.wave_solution(u0_str, beta, xgrid, nt)
        G = np.full(u_vals.shape, float(source))
    else:
        print("WARNING: System is not specified.")

    u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
    Exact = u_star.reshape(len(t), len(x), 1) # Exact on the (x,t) grid

    return {
        'system': system,
        'beta': beta,
        'nu': nu,
        'rho': rho,
        'u0_str': u0_str,
        'source': source,
        'G': G.astype(np.float32),
        'xt_grid': XT_grid.astype(np.float32), # (nt, xgrid)
        'u': Exact.astype(np.float32), # (nt, xgrid, 1)
        'x': x.astype(np.float32), # (xgrid, 1)
        't': t.astype(np.float32), # (nt, 1)
    }


def plot_solution(U_pred, x, t, ax=None, title=None):
    import matplotlib.pyplot as plt
    """Visualize u_predicted."""

    if ax is None:
        fig, ax = plt.gcf(), plt.gca()

    # colorbar for prediction: set min/max to ground truth solution.
    # U_pred.sum(1, keepdims=True).dot(U_pred.sum(0, keepdims=True))
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=U_pred.min().item(), vmax=U_pred.max().item())
    # if X_collocation is not None:
    #     ax.scatter(X_collocation[..., [1]], X_collocation[..., [0]], marker='x', c='black')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel('t', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)
    ax.set_title(title)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)


def plot_latents(px, zt, x, t, labels=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18,6))
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    cmaps = [c + '_r' for c in cmaps]
    plt.subplot(1, 2, 1)
    plt.title(f'zt')
    for i, zti in enumerate(zt):
        plt.scatter(zti[:, 0], zti[:, 1], c=t[i],
                label=labels[i],
                cmap=cmaps[i], alpha=1)  # yellow: 1, blue:  0
    # plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.subplot(1, 2, 2)

    plt.title(f'px')
    for i, pxi in enumerate(px):
        plt.scatter(pxi[:, 0], pxi[:, 1], c=x[i],
                label=labels[i],
                cmap=cmaps[i], alpha=1)  # yellow: 1, blue:  0
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.tight_layout()


class SimplePDEDataset(data.Dataset):
    def __init__(self, data_args_list, params=None):
        self.data_args_list = data_args_list
        self.data_list = [gen_simple_pde_data(*args) for args in self.data_args_list]
        self.params = params

    def __getitem__(self, item_index):
        item = {**self.data_list[item_index], 'item': item_index}
        if self.params is not None:
            item['params'] = self.params[item_index]
        return item

    def __len__(self):
        return len(self.data_args_list)




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def data_repr(data):
        str = f''
        for k, v in data.items():
            str += f'{k}: '
            if isinstance(v, np.ndarray) or False:
                str += f'{v.shape} \n'
            else:
                str += f'{v} \n'
        return str + '\n'

    # data = gen_simple_pde_data('convection', 0, 1, 0, 'sin(x)', 50, 100, 0)
    # for k, v in data.items():
    #     print(k)
    #     if isinstance(v, np.ndarray) or False:
    #         print(v.shape)
    #     else:
    #         print(v)

    dataset = SimplePDEDataset([
        ('convection', 0, 1, 0, 'sin(x)', 50, 100, 0),
        ('convection-diffusion', 1, 1, 0, 'sin(x)', 50, 100, 0)
    ])
    print(data_repr(dataset[0]))

    loader = data.DataLoader(dataset, batch_size=2)

    d =next(iter(loader))

    print(d)

    # plot_solution(data['u'], data['x'], data['t'])
    # plt.show()

