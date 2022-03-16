"""Run PINNs for convection/reaction/reaction-diffusion with periodic boundary conditions."""

import argparse
from net_pbc_aug import *
import numpy as np
import os
import random
import torch

from pbc_examples.net_pbc_aug import PhysicsInformedNN_pbc_aug
from systems_pbc import *
import torch.backends.cudnn as cudnn
from utils import *
from visualize import *
import matplotlib.pyplot as plt

################
# Arguments
################
parser = argparse.ArgumentParser(description='Characterizing/Rethinking PINNs')

parser.add_argument('--system', type=str, default='convection', help='System to study.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

parser.add_argument('--xgrid', type=int, default=100, help='Number of points in the xgrid.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
parser.add_argument('--nu', type=float, default=1.0, help='nu value that scales the d^2u/dx^2 term. 0 if only doing advection.')
parser.add_argument('--rho', type=float, default=1.0, help='reaction coefficient for u*(1-u) term.')
parser.add_argument('--beta', type=float, default=1.0, help='beta value that scales the du/dx term. 0 if only doing diffusion.')
parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")

parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus the first layer.')
parser.add_argument('--net', type=str, default='DNN', help='The net architecture that is to be used.')
parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')
parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')

parser.add_argument('--visualize', default=True, help='Visualize the solution.')
parser.add_argument('--train_method', default='regression', choices=['regression', 'pinns'],)
parser.add_argument('--save_model', default=False, help='Save the model for analysis later.')

args = parser.parse_args()

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



# parse the layers list here
orig_layers = args.layers
layers = [int(item) for item in args.layers.split(',')]

# def gen_data(system, u0_str, nu, beta):
def gen_data(system, nu, beta, rho, u0_str):
    # nu = args.nu
    # beta = args.beta
    # rho = args.rho

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

    print('nu', nu, 'beta', beta, 'rho', rho)

    ############################
    # Process data
    ############################
    x = np.linspace(0, 2 * np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

    # remove initial and boundaty data from X_star
    t_noinitial = t[1:]
    # remove boundary at x=0
    x_noboundary = x[1:]
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)
    X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f)

    if 'convection' in system or 'diffusion' in system:
        u_vals = convection_diffusion(u0_str, nu, beta, args.source, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif 'rd' in system:
        u_vals = reaction_diffusion_discrete_solution(u0_str, nu, rho, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif 'reaction' in system:
        u_vals = reaction_solution(u0_str, rho, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    elif system == 'wave':
        u_vals = wave_solution(u0_str, beta, args.xgrid, args.nt)
        G = np.full(X_f_train.shape[0], float(args.source))
    else:
        print("WARNING: System is not specified.")

    u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
    Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid
    title = f'{system}, {nu}, {beta}, {rho}, {u0_str}'
    u_predict(None, Exact, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr,
              u0_str, system, path=None, prefix=f'target: {title}', X_collocation=None)
    plt.show()

    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
    uu1 = Exact[0:1,:].T # u(x, t) at t=0
    bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
    uu2 = Exact[:,0:1] # u(-end, t)

    # generate the other BC, now at x=2pi
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
    bc_ub = np.hstack((x_bc_ub, t))

    u_train = uu1 # just the initial condition
    X_u_train = xx1 # (x,t) for initial condition

    return system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho, u_star, X_star, u0_str

############################
# Train the model
############################

set_seed(args.seed) # for weight initialization
# u_star_train, X_star_train = sample_random(u_star, args.N_f), sample_random(X_star, args.N_f)
# X_star_train, u_star_train = X_star, u_star
# model = PhysicsInformedNN_pbc(system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho,
#                             args.optimizer_name, args.lr, args.net, args.L, args.activation, args.loss_style,
#                               u_star=u_star_train, train_method=args.train_method, X_star=X_star_train)

# data_list = [gen_data(args) for args in [args.u0_str, 'sin(2x)',  'sin(6x)',  'sin(x/2)']]
# u0_strs = [args.u0_str, 'sin(2x)', 'sin(x/2)', 'sin(6x)']
# u0_strs = ['sin(x)', 'sin(2x)', 'np.sin(1.5*x)', ]
# u0_strs = ['sin(x)', '0.5sin(x)', '0.1sin(x)']
# u0_strs = [f'np.sin({c}*x)' for c in np.linspace(1, 4, 4)/2]

# variables = []
# for u0_str in [f'np.sin({c}*x)' for c in [1, 2]]:
#     for (system, nu, beta, rho) in [('convection', 0, 1, 0),
#                                     ('convection',  0, 2, 0),
#                                     ('convection', 0, 4, 0),
#                                     ('convection', 0, 5, 0),
#                                     ]:
# variables.append(*v)
# variables = [
#     ('convection', 0, 1, 0, 'np.sin(1*x)'),
#     # ('convection-diffusion', 1, 2, 0, 'np.sin(1*x)'),
#     ('convection', 0, 4, 0, 'np.sin(2*x)'),
#     # ('convection', 0, 2, 0, 'np.sin(2*x)'),
#           ]

variables = [
    # ('convection', 0, 2, 0, 'np.sin(2*x)'),
    ('convection-diffusion', 1, 1, 0, 'np.sin(1*x)'),
    # ('convection-diffusion', 1, 4, 0, 'np.sin(1*x)'),
    # ('convection-diffusion', 2, 4, 0, 'np.sin(1*x)'),
    # ('convection-diffusion', 2, 1, 0, 'np.sin(1*x)'),
    # ('convection-diffusion', 1, 1, 0, 'np.sin(2*x)'),
    # ('convection-diffusion', 1, 4, 0, 'np.sin(2*x)'),
    ('convection-diffusion', 2, 4, 0, 'np.sin(2*x)'),
    # ('convection-diffusion', 2, 1, 0, 'np.sin(2*x)'),


    # ('convection', 0, 2, 0, 'np.sin(1*x)'),
    # ('convection', 0, 4, 0, 'np.sin(2*x)'),
    # ('convection', 0, 2, 0, 'np.sin(2*x)'),
          ]


data_list = [gen_data(*v) for v in variables]
data = DataList(('system', 'nu', 'beta', 'rho', 'u0_str'), [Data(*data) for data in data_list])
model = PhysicsInformedNN_pbc_aug(data, args.optimizer_name, args.lr, args.net, args.L, args.activation, args.loss_style,
                                  train_method=args.train_method)


def eval(variables, e=-1):
    system, nu, beta, rho, u0_str = variables
    # new data points
    nt, xgrid = args.nt, args.xgrid
    # nt, xgrid = 256, 256
    x = np.linspace(0, 2 * np.pi, xgrid, endpoint=False).reshape(-1, 1)  # not inclusive
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)  # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # all the x,t "test" data

    if u0_str is None:
        u0_str = args.u0_str
    u_pred = model.predict(X_star, [(system, nu, beta, rho, u0_str)])
    # f_pred = model.predict_f(X_star)

    if 'convection' in system or 'diffusion' in system:
        u_vals = convection_diffusion(u0_str, nu, beta, args.source, xgrid, nt)
        # G = np.full(data.X_f_train.shape[0], float(args.source))
    elif 'rd' in system:
        u_vals = reaction_diffusion_discrete_solution(u0_str, nu, rho, xgrid, nt)
        # G = np.full(data.X_f_train.shape[0], float(args.source))
    elif 'reaction' in system:
        u_vals = reaction_solution(u0_str, rho, xgrid, args.nt)
        # G = np.full(data.X_f_train.shape[0], float(args.source))
    elif system == 'wave':
        u_vals = wave_solution(u0_str, beta, xgrid, args.nt)
    else:
        print("WARNING: System is not specified.")

    u_star = u_vals.reshape(-1, 1)  # Exact solution reshaped into (n, 1)
    Exact = u_star.reshape(len(t), len(x))  # Exact on the (x,t) grid

    error_u_relative = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_u_abs = np.mean(np.abs(u_star - u_pred))
    error_u_linf = np.linalg.norm(u_star - u_pred, np.inf) / np.linalg.norm(u_star, np.inf)

    print('Error u rel: %e' % (error_u_relative))
    print('Error u abs: %e' % (error_u_abs))
    print('Error u linf: %e' % (error_u_linf))

    if args.visualize:
        path = f"heatmap_results/{system}"
        if not os.path.exists(path):
            os.makedirs(path)
        print('saving to ', path)

        u_pred = u_pred.reshape(len(t), len(x))
        # f_pred = f_pred.reshape(len(t), len(x))
        # exact_u(Exact, x, t, nu, beta, rho, orig_layers, args.N_f, args.L, args.source, u0_str, system,
        #         path=path)
        # u_diff(Exact, u_pred, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr,
        #        u0_str, system, path=path)
        prefix = f'epoch:{e}'
        fn = f"{path}/{prefix}_zt_{system}_nu{nu}_beta{beta}" \
             f"_rho{rho}_Nf{args.N_f}_{orig_layers}_L{ args.L}_seed{args.seed}_source{args.source}" \
             f"_{u0_str}_lr{args.lr}.png"
        import matplotlib.pyplot as plt
        title = f'u0_str:{u0_str}, {system}, beta: {beta}, nu:{nu} epoch:{e}'
        if True:
            zt = model.dnn.zt.detach().cpu().numpy()
            px = model.dnn.px.detach().cpu().numpy()
            pr = model.dnn.prod.detach().cpu().numpy()
            plt.suptitle(title, fontsize=12)
            plt.subplot(1, 2, 1)
            plt.title(r'$\{z(t)\}_{t \in [0, 1]}$')
            plt.gca().set_aspect('equal')
            if model.dnn.latent_dim == 1:
                plt.scatter(T.flatten(), zt.squeeze())
            else:
                plt.scatter(zt[:, 0], zt[:, 1], c=model.dnn.t.squeeze().detach().cpu().numpy(), label='zt')  # yellow: 1, blue:  0
            plt.subplot(1, 2, 2)
            plt.gca().set_aspect('equal')
            plt.title(r'$\{p(x)\}_{x \in [0, 2 \pi]}$')
            if model.dnn.latent_dim == 1:
                plt.scatter(X.flatten(), px.squeeze())
            else:
                plt.scatter(px[:, 0], px[:, 1], c= model.dnn.x.squeeze().detach().cpu().numpy(), label='px', cmap='inferno')  # yellow: 1, blue:  0
            # plt.subplot(2, 2, 3)
            # plt.gca().set_aspect('equal')
            # plt.title(r'$\{z(t) \cdot p(x)\}_{t \in [0, 1], x \in [0, 2 \pi]}$')
            # if model.dnn.latent_dim == 1:
            #     pass
            #     # plt.plot(T.squeeze(), zt.squeeze())
            # else:
            #     plt.scatter(pr[:, 0], pr[:, 1], c= model.dnn.t.squeeze().detach().cpu().numpy(), label='t')  # yellow: 1, blue:  0
            #     plt.colorbar()
            # plt.subplot(2, 2, 4)
            # plt.gca().set_aspect('equal')
            # plt.title(r'$\{z(t) \cdot p(x)\}_{t \in [0, 1], x \in [0, 2 \pi]}$')
            # if model.dnn.latent_dim == 1:
            #     pass
            #     # plt.plot(T.squeeze(), zt.squeeze())
            # else:
            #     plt.scatter(pr[:, 0], pr[:, 1], c= model.dnn.x.squeeze().detach().cpu().numpy(), label='x', cmap='inferno')  # yellow: 1, blue:  0
            #     plt.colorbar()
            plt.tight_layout()
            plt.show()
            # print(model.dnn.linear.weight)
            plt.clf()

        # plt.savefig(fn)
        u_predict(u_vals, u_pred, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr,
                  u0_str, system, path=path, prefix=title)

        plt.show()
        # u_predict(u_vals, f_pred, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr,
        #           u0_str, system, path=path, prefix=f'f_epoch:{e}')
        # plt.show()

        return zt, px

cmaps =  ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps = [c + '_r' for c in cmaps]

if args.optimizer_name != 'LBFGS':
    for e in range(10000):
        if e % 200 == 0:
            print('e', e)
            zts, pxs = [], []
            for v in variables:
                zt, px = eval(v, e=e)
                zts.append(zt)
                pxs.append(px)

            plt.clf()
            plt.subplot(1, 2, 1)
            plt.title(f'zt epoch{e}')
            for i, zt in enumerate(zts):
                plt.scatter(zt[:, 0], zt[:, 1], c=model.dnn.t.squeeze().detach().cpu().numpy(),
                            label=' '.join([str(v) for v in variables[i]]),
                        cmap=cmaps[i], alpha=1)  # yellow: 1, blue:  0
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.title(f'px epoch{e}')
            for i, px in enumerate(pxs):
                plt.scatter(px[:, 0], px[:, 1], c=model.dnn.x.squeeze().detach().cpu().numpy(),
                            label=' '.join([str(v) for v in variables[i]]),
                        cmap=cmaps[i], alpha=1)  # yellow: 1, blue:  0
            plt.tight_layout()
            plt.legend()
            plt.show()

            x = np.linspace(0, 2 * np.pi, args.xgrid, endpoint=False).reshape(-1, 1)  # not incllendusive
            t = np.linspace(0, 1, args.nt).reshape(-1, 1)
            X, T = np.meshgrid(x, t)  # all the X grid points T times, all the T grid points X times
            X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # all the x,t "test" data

            # model.predict(X_star, [('convection', 0, 1, 0, 'np.sin(2*x)')])
            model.predict(X_star, [('convection-diffusion', 1, 1, 0, 'np.sin(1*x)')])
            evx = model.dnn.evx
            model.predict(X_star, [('convection-diffusion', 2, 4, 0, 'np.sin(2*x)')])
            # model.predict(X_star, [('convection', 0, 3, 0, 'np.sin(1*x)')])
            evt = model.dnn.evt
            X = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
            T = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
            u_pred = model.dnn(torch.cat([X, T], dim=1), None, evx=evx, evt=evt).reshape(100, 100).detach().cpu().numpy()
            # title ='swap: np.sin(2*x), beta 3'
            title =f'swap: np.sin(1*x), beta 4 nu 1  epoch{e}'
            u_predict(None, u_pred, x, t, 'nu', 'beta', 'rho', args.seed, orig_layers, args.N_f, args.L, args.source,
                      args.lr,
                      'u0_str', 'system', path='path', prefix=title)
            plt.show()

            # model.predict(X_star, [('convection', 0, 1, 0, 'np.sin(2*x)')])
            model.predict(X_star, [('convection-diffusion', 2, 4, 0, 'np.sin(2*x)')])
            evx = model.dnn.evx
            # model.predict(X_star, [('convection', 0, 4, 0, 'np.sin(1*x)')])
            model.predict(X_star, [('convection-diffusion', 1, 1, 0, 'np.sin(1*x)')])
            evt = model.dnn.evt
            X = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
            T = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
            u_pred = model.dnn(torch.cat([X, T], dim=1), None, evx=evx, evt=evt).reshape(100, 100).detach().cpu().numpy()
            title =f'swap: np.sin(2*x), beta 1 nu 1 epoch{e}'
            u_predict(None, u_pred, x, t, 'nu', 'beta', 'rho', args.seed, orig_layers, args.N_f, args.L, args.source,
                      args.lr,
                      'u0_str', 'system', path='path', prefix=title)
            plt.show()
            if False:
                #     ('convection', 0, 2, 0, 'np.sin(2*x)'),
                #     ('convection-diffusion', 1, 2, 0, 'np.sin(1*x)'),

                # testing with random temporal dynamics in latent sapce
                model.predict(X_star, [('convection-diffusion', 2, 4, 0, 'np.sin(2*x)')])
                evx = model.dnn.evx
                # model.predict(X_star, [('convection', 0, 4, 0, 'np.sin(1*x)')])
                model.predict(X_star, [('convection-diffusion', 1, 1, 0, 'np.sin(1*x)')])
                evt = torch.cat([T, T], -1)
                # evt = model.dnn.evt
                X = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
                T = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
                u_pred = model.dnn(torch.cat([X, T], dim=1), None, evx=evx, evt=evt).reshape(100, 100).detach().cpu().numpy()
                title =f'swap: np.sin(2*x), random time direction epoch{e}'
                u_predict(None, u_pred, x, t, 'nu', 'beta', 'rho', args.seed, orig_layers, args.N_f, args.L, args.source,
                          args.lr,
                          'u0_str', 'system', path='path', prefix=title)
                plt.show()

                # testing with sin(2x) but random spatial dynamics in latent sapce
                # model.predict(X_star, [('convection', 0, 4, 0, 'np.sin(2*x)')])
                # evx = model.dnn.evx
                evx = torch.cat([X, X], -1) - np.pi / np.pi * 0.5
                # model.predict(X_star, [('convection', 0, 4, 0, 'np.sin(1*x)')])
                model.predict(X_star, [('convection-diffusion', 2, 4, 0, 'np.sin(2*x)')])
                evt = model.dnn.evt
                X = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
                T = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
                u_pred = model.dnn(torch.cat([X, T], dim=1), None, evx=evx, evt=evt).reshape(100, 100).detach().cpu().numpy()
                title =f'swap: np.sin(2*x), random space direction epoch{e}'
                u_predict(None, u_pred, x, t, 'nu', 'beta', 'rho', args.seed, orig_layers, args.N_f, args.L, args.source,
                          args.lr,
                          'u0_str', 'system', path='path', prefix=title)
                plt.show()

            # see interpolation capabilities
            # eval(system,  nu, beta, rho, 'interp', e=e)
        model.train()

else:
    model.train()

# eval(v)


# if args.save_model: # whether or not to save the model
#     path = "saved_models"
#     if not os.path.exists(path):
#         os.makedirs(path)
#     if 'pretrained' not in args.net: # only save new models
#         torch.save(model, f"saved_models/pretrained_{args.system}_u0{args.u0_str}_nu{nu}_beta{beta}_rho{rho}_Nf{args.N_f}_{args.layers}_L{args.L}_source{args.source}_seed{args.seed}.pt")
