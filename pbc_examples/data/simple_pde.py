import torch.utils.data as data
import numpy as np

from pbc_examples.data import systems_pbc as systems


def gen_coordinates(coords_args):
    grids, coordnames = [], []
    for coord_args in coords_args:
        coordname, coord = coord_args
        grid = np.expand_dims(coord, -1)
        grids.append(grid)
        coordnames.append(coordname)
    mgrid = np.stack(np.meshgrid(*grids), axis=-1)
    return {c:g for c, g in zip(coordnames, grids)}, mgrid


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
    grids, XT_grid = gen_coordinates((('x', np.linspace(0, 2 * np.pi, xgrid, endpoint=False)),
                                      ('t', np.linspace(0, T, nt))))
    x, t = grids['x'], grids['t']

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

