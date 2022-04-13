import itertools

import numpy as np


def to_numpy(t):
    return t.detach().cpu().numpy()


def make_norm_denorm_fns(a, b):
    def renorm(x):
        return (x - a) / (b - a)
    def denorm(y):
        return y * (b - a) + a
    return renorm, denorm


def create_params(name, separate_params=False):
    valid_args_list, valid_params = [], {'x_params': [], 't_params': [], 'params': []}
    if name == 'gauss':
        assert separate_params
        x_params = np.array([[0],], dtype=np.float32)
        t_params = np.array([[0],], dtype=np.float32)
        params = {'x_params': x_params, 't_params': t_params}
        # params = np.array([[0],])
        data_args_list = [
            ('convection', 0, 1, 0, 'np.exp(-np.power((x - 0.5*np.pi)/(np.pi/16), 2.)/2.)', 100, 100, 0),
        ]
    elif name == 'big':
        n = 3
        beta_min, beta_max = 0, 5
        betas = np.linspace(beta_min, beta_max, n).astype(np.float32)
        beta2p, p2beta = make_norm_denorm_fns(beta_min, beta_max)

        omega_min, omega_max = 1, n
        omegas = np.linspace(omega_min, omega_max, n).astype(np.float32)
        omega2p, p2omega = make_norm_denorm_fns(omega_min, omega_max)

        nu_min, nu_max = 0, 1
        nus = np.linspace(nu_min, nu_max, n).astype(np.float32)
        nu2p, p2nu = make_norm_denorm_fns(nu_min, nu_max)

        params = np.array(list(itertools.product(omega2p(omegas), beta2p(betas), nu2p(nus))))
        data_args_list = [
            ('convection-diffusion', p2nu(nup), p2beta(betap), 0, f'np.sin({p2omega(omegap)}*x)', 100, 100, 0)
            for omegap, betap, nup in params]
        if separate_params:
            x_params, t_params = params[..., [0]], params[..., 1:]
            params = {'x_params': x_params, 't_params': t_params}

    elif name == 'simple_swap':
        assert separate_params
        x_params = np.array([[1], [-1]], dtype=np.float32)
        t_params = np.array([[1], [-1]], dtype=np.float32)
        params = {'x_params': x_params, 't_params': t_params}
        data_args_list = [
            ('convection', 0, 1, 0, 'np.sin(2*x)', 100, 101, 0),
            ('convection', 0, 4, 0, 'np.sin(1*x)', 100, 101, 0),
        ]
        x_valid_params = np.array([[-1], [1]], dtype=np.float32)
        t_valid_params = np.array([[1], [-1]], dtype=np.float32)
        valid_params = {'x_params': x_valid_params, 't_params': t_valid_params}
        valid_args_list = [
            ('convection', 0, 1, 0, 'np.sin(1*x)', 100, 101, 0),
            ('convection', 0, 4, 0, 'np.sin(2*x)', 100, 101, 0),
        ]
    elif name == 'simple_swap_x2':
        assert separate_params
        x_params = np.array([[1], [1], [-1], [-1]], dtype=np.float32)
        t_params = np.array([[1], [-1], [1], [-1]], dtype=np.float32)
        params = {'x_params': x_params, 't_params': t_params}
        data_args_list = [
            ('convection', 0, 1, 0, 'np.sin(2*x)', 100, 101, 0),
            ('convection', 0, 4, 0, 'np.sin(2*x)', 100, 101, 0),
            ('convection', 0, 1, 0, 'np.sin(1*x)', 100, 101, 0),
            # ('convection', 0, 4, 0, 'np.sin(1*x)', 100, 101, 0),
        ]
        x_valid_params = np.array([[-1], [1]], dtype=np.float32)
        t_valid_params = np.array([[1], [-1]], dtype=np.float32)
        valid_params = {'x_params': x_valid_params, 't_params': t_valid_params}
        valid_args_list = [
            ('convection', 0, 4, 0, 'np.sin(1*x)', 100, 101, 0),
        ]
    elif name == 'simple_swap_x3':
        assert separate_params
        x_params = np.array([[-1], [-1], [-1],
                             [0], [1],],
                            dtype=np.float32)
        t_params = np.array([[-1], [0], [1],
                             [-1], [-1],
                             ], dtype=np.float32)
        params = {'x_params': x_params, 't_params': t_params}
        data_args_list = [
            ('convection', 0, 1, 0, 'np.sin(1*x)', 100, 101, 0),
            ('convection', 0, 2, 0, 'np.sin(1*x)', 100, 101, 0),
            ('convection', 0, 4, 0, 'np.sin(1*x)', 100, 101, 0),
            # ('convection', 0, 1, 0, 'np.sin(1*x)', 100, 101, 0),
            ('convection', 0, 1, 0, 'np.sin(3*x)', 100, 101, 0),
            ('convection', 0, 1, 0, 'np.sin(2*x)', 100, 101, 0),
            # ('convection', 0, 4, 0, 'np.sin(1*x)', 100, 101, 0),
        ]
        x_valid_params = np.array([[0], [0], [1], [1]], dtype=np.float32)
        t_valid_params = np.array([[0], [1], [1], [0]], dtype=np.float32)
        valid_params = {'x_params': x_valid_params, 't_params': t_valid_params}
        valid_args_list = [
            ('convection', 0, 2, 0, 'np.sin(3*x)', 100, 101, 0),
            ('convection', 0, 4, 0, 'np.sin(3*x)', 100, 101, 0),
            ('convection', 0, 4, 0, 'np.sin(2*x)', 100, 101, 0),
            ('convection', 0, 2, 0, 'np.sin(2*x)', 100, 101, 0),
        ]
    else:
        return NotImplementedError(name)

    # concat params and valid_params
    all_params = {}
    for k in params:
        if k in valid_params and len(valid_params[k]):
            all_params[k] = np.vstack([params[k], valid_params[k]])
        else:
            all_params[k] = params[k]

    return data_args_list, params,\
           data_args_list + valid_args_list, all_params