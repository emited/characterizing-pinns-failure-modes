import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from choose_optimizer import *

# CUDA support
from pbc_examples.data.systems_pbc import convection_diffusion, reaction_solution, reaction_diffusion_discrete_solution
from pbc_examples.modules.discr import ToyDiscriminator, set_requires_grad, GANLoss
from pbc_examples.modules.features import FourierFeatures
from pbc_examples.modules.nn_symmetries import SymmetryNet, ModulatedSymmetryNet
from pbc_examples.modules.separation_param_simple_latents_un import SeparationParamSimpleLatentUn
from pbc_examples.net_pbc import Sine, SymmetricInitDNN
from pbc_examples.utils import sample_random

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def make_grid(system, nu, beta, rho, u0_str, xgrid, nt, N_f, source):

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

    print('nu', nu, 'beta', beta, 'rho', rho)

    ############################
    # Process data
    ############################

    x = np.linspace(0, 2 * np.pi, xgrid, endpoint=False).reshape(-1, 1)  # not inclusive
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)  # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # all the x,t "test" data

    # remove initial and boundaty data from X_star
    t_noinitial = t[1:]
    # remove boundary at x=0
    x_noboundary = x[1:]
    X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
    X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

    # sample collocation points only from the interior (where the PDE is enforced)
    X_f_train = sample_random(X_star_noinitial_noboundary, N_f)

    if 'convection' in system or 'diffusion' in system:
        u_vals = convection_diffusion(u0_str, nu, beta, source, xgrid, nt)
        G = np.full(X_f_train.shape[0], float(source))
    elif 'rd' in system:
        u_vals = reaction_diffusion_discrete_solution(u0_str, nu, rho, xgrid, nt)
        G = np.full(X_f_train.shape[0], float(source))
    elif 'reaction' in system:
        u_vals = reaction_solution(u0_str, rho, xgrid, nt)
        G = np.full(X_f_train.shape[0], float(source))
    else:
        print("WARNING: System is not specified.")

    u_star = u_vals.reshape(-1, 1)  # Exact solution reshaped into (n, 1)
    Exact = u_star.reshape(len(t), len(x))  # Exact on the (x,t) grid

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  # initial condition, from x = [-end, +end] and t=0
    uu1 = Exact[0:1, :].T  # u(x, t) at t=0
    bc_lb = np.hstack((X[:, 0:1], T[:, 0:1]))  # boundary condition at x = 0, and t = [0, 1]
    uu2 = Exact[:, 0:1]  # u(-end, t)

    # generate the other BC, now at x=2pi
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    x_bc_ub = np.array([2 * np.pi] * t.shape[0]).reshape(-1, 1)
    bc_ub = np.hstack((x_bc_ub, t))

    u_train = uu1  # just the initial condition
    X_u_train = xx1  # (x,t) for initial condition



# model = PhysicsInformedNN_pbc(args.system, X_u_train, u_train, X_star_noinitial_noboundary, bc_lb, bc_ub, layers, G, nu, beta, rho,
#                             args.optimizer_name, args.lr, args.net, args.N_f, args.L, args.activation, args.loss_style, UB=0, nx=args.xgrid, nt=args.nt, nz=nz, zdim=zdim, base_distr='normal')

    return X_u_train, u_train, X_star_noinitial_noboundary, bc_lb, bc_ub, G

class PhysicsInformedNN_pbc():
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, system, nu, beta, rho, u0_str, xgrid, nt, N_f, source):

        # X_u_train, u_train, X_star_noinitial_noboundary, bc_lb, bc_ub, layers, G, nu, beta, rho,
        #         net, N_f, L=1, activation='tanh', loss_style='mean', UB=1, nx=None, nt=None, nz=None, zdim=None, base_distr='normal',

        X_u_train, u_train, X_star_noinitial_noboundary, bc_lb, bc_ub, G = \
            make_grid(system, nu, beta, rho, u0_str, xgrid, nt, N_f, source)


        self.N_f = N_f
        self.nx, self.nt = xgrid, nt
        self.system = system
        self.X_star_noinitial_noboundary = X_star_noinitial_noboundary
        # self.x_f = torch.tensor(X_f_train[:, 0:1]).float().to(device)
        # self.t_f = torch.tensor(X_f_train[:, 1:2]).float().to(device)
        self.x_u = torch.tensor(X_u_train[:, 0:1]).float().to(device)
        self.t_u = torch.tensor(X_u_train[:, 1:2]).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1]).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2]).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1]).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2]).float().to(device)
        if nz is not None:
            # self.z_f = self.z.unsqueeze(1).expand(-1, X_f_train.shape[0], -1)
            # self.z_u = self.z.unsqueeze(1).expand(-1, X_u_train.shape[0], -1)
            # self.z_bc_ub = z.unsqueeze(1).expand(-1, self.x_bc_ub.shape[0], -1)
            # self.z_bc_lb = z.unsqueeze(1).expand(-1, self.t_bc_ub.shape[0], -1)
            # self.x_f = self.x_f.unsqueeze(0).expand(nz, -1, -1)
            # self.t_f = self.t_f.unsqueeze(0).expand(nz, -1, -1)
            self.x_u = self.x_u.unsqueeze(0).expand(nz, -1, -1)
            self.t_u = self.t_u.unsqueeze(0).expand(nz, -1, -1)
            self.x_bc_lb = self.x_bc_lb.unsqueeze(0).expand(nz, -1, 1)
            self.t_bc_lb = self.t_bc_lb.unsqueeze(0).expand(nz, -1, 1)
            self.x_bc_ub = self.x_bc_ub.unsqueeze(0).expand(nz, -1, 1)
            self.t_bc_ub = self.t_bc_ub.unsqueeze(0).expand(nz, -1, 1)

        # self.x_f.requires_grad = True
        # self.t_f.requires_grad = True
        self.x_u.requires_grad = True
        self.t_u.requires_grad = True
        self.x_bc_lb.requires_grad = True
        self.t_bc_lb.requires_grad = True
        self.x_bc_ub.requires_grad = True
        self.t_bc_ub.requires_grad = True


        self.u = torch.tensor(u_train, requires_grad=True).float().to(device)
        self.nu = nu
        self.beta = beta
        self.rho = rho

        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)

    def sample_collocation_points(self):
        X_all = self.X_star_noinitial_noboundary
        idx = np.random.choice(X_all.shape[0], self.N_f, replace=False)
        X_f_train = X_all[idx, :]
        x_f = torch.tensor(X_f_train[:, 0:1]).float().to(device)
        t_f = torch.tensor(X_f_train[:, 1:2]).float().to(device)
        if self.nz is not None:
            x_f = x_f.unsqueeze(0).expand(self.nz, -1, -1)
            t_f = t_f.unsqueeze(0).expand(self.nz, -1, -1)
        return x_f, t_f

    def z_f(self, z, N_f):
        z_f = z.unsqueeze(1).expand(-1, N_f, -1)
        z_f.requires_grad = True
        return z_f

    def z_u(self, z):
        return z.unsqueeze(1).expand(-1, self.x_u.shape[1], -1)

    def z_bc_ub(self, z):
        return z.unsqueeze(1).expand(-1, self.x_bc_ub.shape[1], -1)

    def z_bc_lb(self, z):
        return z.unsqueeze(1).expand(-1, self.x_bc_lb.shape[1], -1)

    def net_u(self, x, t, z=None):
        """The standard DNN that takes (x,t) --> u."""
        if isinstance(self.dnn, SymmetryNet) \
                or isinstance(self.dnn, ModulatedSymmetryNet):
            coords = torch.cat([x, t], dim=-1)
            u = self.dnn(coords, z)
        else:
            input = torch.cat([x, t], dim=1)
            if z is not None:
                input = torch.cat([x, t, z], dim=-1)
            u = self.dnn(input)
        if isinstance(u, list) or isinstance(u, tuple):
            u, aux_outputs = u[0], u[1:]
            return u, aux_outputs
        return u, None

    def net_f(self, x, t, z=None):
        """ Autograd for calculating the residual for different systems."""
        u, w = self.net_u(x, t, z)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        if self.nu != 0:
            u_xx = torch.autograd.grad(
                u_x, x,
                grad_outputs=torch.ones_like(u_x),
                retain_graph=True,
                create_graph=True
            )[0]
        else:
            u_xx = 0

        if 'convection' in self.system or 'diffusion' in self.system:
            f = u_t - self.nu*u_xx + self.beta*u_x - self.G
        elif 'rd' in self.system:
            f = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif 'reaction' in self.system:
            f = u_t - self.rho*u + self.rho*u**2

        if z is None:
            return f
        else:
            # u_z = torch.autograd.grad(
            #     u, z,
            #     grad_outputs=torch.ones_like(u),
            #     retain_graph=True,
            #     create_graph=True
            # )[0]
            u_z = None
            return f, u_z, w

    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """For taking BC derivatives."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
        )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
        )[0]

        return u_lb_x, u_ub_x

    def net_u_derivatives(self, u0, x_u0):
        """For taking initial condition derivatives."""

        u0_x = torch.autograd.grad(
            u0, x_u0,
            grad_outputs=torch.ones_like(u0),
            retain_graph=True,
            create_graph=True
        )[0]

        if self.nu != 0:
            u0_xx = torch.autograd.grad(
                u0_x, x_u0,
                grad_outputs=torch.ones_like(u0),
                retain_graph=True,
                create_graph=True
            )[0]
        else:
            u0_xx = None

        return u0_x, u0_xx

    def loss_bc(self, u_pred_lb, u_pred_ub):
        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
        loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
        if self.nu != 0:
            loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
        return loss_b

    def loss_u0(self, u0_pred):
        loss_u = torch.mean((self.u - u0_pred) ** 2)

    def loss_pinn(self, u_pred):
        z_f = self.z_f(z, self.N_f)
        with torch.no_grad():
            x_f, t_f = self.sample_collocation_points()
        x_f.requires_grad = True
        t_f.requires_grad = True

        f_pred = self.net_f(x_f, t_f, z_f)
        f_pred, u_z, w_f = f_pred

        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
        if self.nu != 0:
            loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
        # loss_f = torch.mean(torch.log(f_pred ** 2))
        # loss_f = torch.mean(torch.max(f_pred ** 2))
        # loss_f = torch.mean(torch.softmax(f_pred ** 2, dim=-1))
        # loss_f = torch.mean(torch.max(f_pred ** 2, dim=-2)[0])
        # loss_f = torch.mean(torch.log(torch.abs(f_pred)))
        loss_f = torch.mean(f_pred ** 2)
        # loss_f = torch.sum(torch.log(torch.abs(f_pred)))/ f_pred.numel()
        # loss = 0
        loss = self.UB * (loss_u + loss_b) + self.L*loss_f


        gan_coeff = 1
        loss += gan_coeff * self.criterion_gan(self.discr(u_pred.squeeze(-1)), True)

        return loss


    def predict(self, X, z=None):

        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        if z is not None:
            z = z.to(device)
            z = z.unsqueeze(1).expand(-1, X.shape[0], -1)
            x = x.unsqueeze(0).expand(z.shape[0], -1, -1)
            t = t.unsqueeze(0).expand(z.shape[0], -1, -1)

        self.dnn.eval()
        u, _ = self.net_u(x, t, z)
        u = u.detach().cpu().numpy()

        return u