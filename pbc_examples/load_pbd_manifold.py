"""Run PINNs for convection/reaction/reaction-diffusion with periodic boundary conditions."""

import argparse
import math

import torch

from net_pbc_manifold import *
import os
from pbc_examples.data.systems_pbc import *
from utils import *
from visualize import *

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

parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
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
parser.add_argument('--save_model', default=False, help='Save the model for analysis later.')

args = parser.parse_args()

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

nu = args.nu
beta = args.beta
rho = args.rho

if args.system == 'diffusion': # just diffusion
    beta = 0.0
    rho = 0.0
elif args.system == 'convection':
    nu = 0.0
    rho = 0.0
elif args.system == 'rd': # reaction-diffusion
    beta = 0.0
elif args.system == 'reaction':
    nu = 0.0
    beta = 0.0

print('nu', nu, 'beta', beta, 'rho', rho)

# parse the layers list here
orig_layers = args.layers
layers = [int(item) for item in args.layers.split(',')]

############################
# Process data
############################

x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
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

if 'convection' in args.system or 'diffusion' in args.system:
    u_vals = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
elif 'rd' in args.system:
    u_vals = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
elif 'reaction' in args.system:
    u_vals = reaction_solution(args.u0_str, rho, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
else:
    print("WARNING: System is not specified.")

u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid

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




# set_seed(args.seed) # for weight initialization

# sample z
nz = 1
zdim = 12
# z = torch.randn(nz, zdim)
# z0s = torch.zeros(nz, zdim).unsqueeze(0)
z0s = torch.randn(nz, zdim).unsqueeze(0)

# z = torch.clip(z, -3, 3)

# layers.insert(0, 2 * X_u_train.shape[-1] + 2 * zdim)
layers.insert(0, 2 * 64)

model = PhysicsInformedNN_pbc(args.system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho,
                            args.optimizer_name, args.lr, args.net, args.L, args.activation, args.loss_style, UB=0, z=z0s[0])


#####################################################################
######################## latent component analysis ####################
###################################################################
eps = torch.linspace(-3, 3, 10).unsqueeze(1).unsqueeze(2)
zs = z0s + eps
i = 0
zs[..., :i] = z0s[..., :i]
zs[..., i+1:] = z0s[..., i+1:]

for iz, z in enumerate(zs):
    u_pred = model.predict(X_star, z=z)
    if args.visualize:
        path = f"heatmap_results/{args.system}"
        u_pred = u_pred.reshape(-1, len(t), len(x))
        for i, u_pred_z in enumerate(u_pred):
            if i > 5:
                break
            u_predict(u_vals, u_pred_z, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path, prefix=f'u_pred_reloaded_{iz}')
            plt.show()
            plt.clf()

#####################################################################
######################## continuous interpoltion ####################
#####################################################################
z0 = torch.randn(1, zdim)
z1 = torch.randn(1, zdim)
a = torch.linspace(0, 1, 6)
zs = torch.stack([z0 * (1- ai) + z1 * ai for ai in a], 0)

for iz, z in enumerate(zs):
    u_pred = model.predict(X_star, z=z)
    if args.visualize:
        path = f"heatmap_results/{args.system}"
        u_pred = u_pred.reshape(-1, len(t), len(x))
        for i, u_pred_z in enumerate(u_pred):
            if i > 5:
                break
            u_predict(u_vals, u_pred_z, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, path=path, prefix=f'u_pred_reloaded_{iz}')
            plt.show()
            plt.clf()
#####################################################################
#####################################################################
#####################################################################


#####################################################################
######################## generator inversion ####################
#####################################################################

# with torch.no_grad():
#     z0s = torch.randn(nz, zdim).unsqueeze(0)
#     z = z0s[0].to(device)
#     z.requires_grad = True
#     x0 = model.x_u
#     u0 = torch.sin(2*x0)
#     # u0[x0 < math.pi] = 0
#     optimizer = torch.optim.Adam([z], lr=0.08)
#
#
# def renorm(x, old_min, old_max, new_min, new_max):
#     return (new_max - new_min) * ((x - old_min) / (old_max - old_min)) + new_min
#
# epochs = 600
# for e in range(1, epochs + 1):
#     old_min, old_max = 0.0935, 0.0970
#     new_min, new_max = 1, -1
#     optimizer.zero_grad()
#     z_u = z.unsqueeze(1).expand(-1, model.z_u.shape[1], -1)
#     u0_rec = renorm(model.net_u(model.x_u, model.t_u, z_u), old_min, old_max, new_min, new_max)
#     loss = torch.mean((u0 - u0_rec) ** 2)
#
#     z_bc_lb = z.unsqueeze(1).expand(-1, model.t_bc_lb.shape[1], -1)
#     z_bc_ub = z.unsqueeze(1).expand(-1, model.t_bc_ub.shape[1], -1)
#     u_pred_lb = renorm(model.net_u(model.x_bc_lb, model.t_bc_lb, z_bc_lb), old_min, old_max, new_min, new_max)
#     u_pred_ub = renorm(model.net_u(model.x_bc_ub, model.t_bc_ub, z_bc_ub), old_min, old_max, new_min, new_max)
#     loss += torch.mean((u_pred_lb - u_pred_ub) ** 2)
#     loss.backward()
#     optimizer.step()
#     if e % 100 == 0 or e == epochs:
#         path = f"heatmap_results/{args.system}"
#         u_pred = renorm(model.predict(X_star, z=z), old_min, old_max, new_min, new_max)
#         u_pred = u_pred.reshape(-1, len(t), len(x))
#         for i, u_pred_z in enumerate(u_pred):
#             if i > 5:
#                 break
#             u_predict(u_vals, u_pred_z, x, t, nu, beta, rho, args.seed, orig_layers, args.N_f, args.L, args.source, args.lr,
#                       args.u0_str, args.system, path=path, prefix=f'u_pred_reloaded_{e}')
#             plt.show()
#             plt.clf()


