import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from choose_optimizer import *

# CUDA support
from pbc_examples.net_pbc import Sine, SymmetricInitDNN

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))
    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        elif activation == 'sin':
            self.activation = Sine
        elif activation == 'swish':
            self.activation = Swish
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                # torch.nn.init.orthogonal_(m.weight, gain=1.)
                torch.nn.init.xavier_uniform_(m.weight, gain=1.)
                # m.bias.data.fill_(0.)
                gain = 1
                torch.nn.init.uniform_(m.bias.data, a=-gain, b=gain)
        # self.layers.apply(init_weights)

    def forward(self, x):
        out = self.layers(x)
        return out

class PhysicsInformedNN_pbc():
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho, optimizer_name, lr,
        net, L=1, activation='tanh', loss_style='mean', UB=1, z=None, nx=None, nt=None):

        self.z = z.to(device)
        self.nx, self.nt = nx, nt
        self.system = system
        self.x_f = torch.tensor(X_f_train[:, 0:1]).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2]).float().to(device)
        self.x_u = torch.tensor(X_u_train[:, 0:1]).float().to(device)
        self.t_u = torch.tensor(X_u_train[:, 1:2]).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1]).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2]).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1]).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2]).float().to(device)
        if z is not None:
            # self.z_f = self.z.unsqueeze(1).expand(-1, X_f_train.shape[0], -1)
            # self.z_u = self.z.unsqueeze(1).expand(-1, X_u_train.shape[0], -1)
            # self.z_bc_ub = z.unsqueeze(1).expand(-1, self.x_bc_ub.shape[0], -1)
            # self.z_bc_lb = z.unsqueeze(1).expand(-1, self.t_bc_ub.shape[0], -1)
            self.x_f = self.x_f.unsqueeze(0).expand(z.shape[0], -1, -1)
            self.t_f = self.t_f.unsqueeze(0).expand(z.shape[0], -1, -1)
            self.x_u = self.x_u.unsqueeze(0).expand(z.shape[0], -1, -1)
            self.t_u = self.t_u.unsqueeze(0).expand(z.shape[0], -1, -1)
            self.x_bc_lb = self.x_bc_lb.unsqueeze(0).expand(z.shape[0], -1, 1)
            self.t_bc_lb = self.t_bc_lb.unsqueeze(0).expand(z.shape[0], -1, 1)
            self.x_bc_ub = self.x_bc_ub.unsqueeze(0).expand(z.shape[0], -1, 1)
            self.t_bc_ub = self.t_bc_ub.unsqueeze(0).expand(z.shape[0], -1, 1)

        self.x_f.requires_grad = True
        self.t_f.requires_grad = True
        self.x_u.requires_grad = True
        self.t_u.requires_grad = True
        self.x_bc_lb.requires_grad = True
        self.t_bc_lb.requires_grad = True
        self.x_bc_ub.requires_grad = True
        self.t_bc_ub.requires_grad = True


        self.net = net

        if self.net == 'DNN':
            self.dnn = DNN(layers, activation).to(device)
            # self.dnn = SymmetricInitDNN(layers, activation).to(device)
        else: # "pretrained" can be included in model path
            # the dnn is within the PINNs class
            self.dnn = torch.load(net).dnn

        self.u = torch.tensor(u_train, requires_grad=True).float().to(device)
        self.layers = layers
        self.nu = nu
        self.beta = beta
        self.rho = rho

        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)

        self.L = L
        self.UB = UB

        self.lr = lr
        self.optimizer_name = optimizer_name

        self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)

        self.loss_style = loss_style

        self.iter = 0

    @property
    def z_f(self):
        z_f = self.z.unsqueeze(1).expand(-1, self.x_f.shape[1], -1)
        z_f.requires_grad = True
        return z_f

    def calc_pl_lengths(self, styles, images):
        device = images.device
        # num_pixels = self.nx * self.nt
        num_pixels = images.shape[1]
        pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
        outputs = (images * pl_noise).sum()

        pl_grads = torch.autograd.grad(outputs=outputs, inputs=styles,
                              grad_outputs=torch.ones(outputs.shape, device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

    @property
    def z_u(self):
        return self.z.unsqueeze(1).expand(-1, self.x_u.shape[1], -1)

    @property
    def z_bc_ub(self):
        return self.z.unsqueeze(1).expand(-1, self.x_bc_ub.shape[1], -1)

    @property
    def z_bc_lb(self):
        return self.z.unsqueeze(1).expand(-1, self.x_bc_lb.shape[1], -1)

    def net_u(self, x, t, z=None):
        """The standard DNN that takes (x,t) --> u."""
        input = torch.cat([x, t], dim=1)
        if z is not None:
            # x_broad = x.unsqueeze(0).expand(z.shape[0], -1, -1)
            # t_broad = t.unsqueeze(0).expand(z.shape[0], -1, -1)
            # input_broad = input.unsqueeze(0).expand(z.shape[0], -1, -1)
            # z_broad = z.unsqueeze(1).expand(-1, input.shape[0], -1)
            input = torch.cat([x, t, z], dim=-1)
        u = self.dnn(input)
        return u

    def net_f(self, x, t, z=None):
        """ Autograd for calculating the residual for different systems."""
        u = self.net_u(x, t, z)
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

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]

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
            return f, u_z

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

        # u0_xx = torch.autograd.grad(
        #     u0_x, x_u0,
        #     grad_outputs=torch.ones_like(u0),
        #     retain_graph=True,
        #     create_graph=True
        #     )[0]
        #
        # return u0_x, u0_xx
        return u0_x, None

    def loss_pinn(self, verbose=True):
        """ Loss function. """
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u, self.z_u)

        u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb, self.z_bc_lb)
        u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub, self.z_bc_ub)
        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
        z_f = self.z_f
        f_pred = self.net_f(self.x_f, self.t_f, z_f)
        if self.z is not None:
            f_pred, u_z = f_pred
        else:
            u_z = None


        # p = 0.3
        if self.loss_style == 'mean':
            loss_u = torch.mean((self.u - u_pred) ** 2)
            loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            if self.nu != 0:
                loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.mean(f_pred ** 2)
            # loss_f = torch.mean(torch.log(torch.abs(f_pred)))
            # loss_f = torch.mean(torch.log(f_pred**2))
            # loss_f = torch.sum(torch.log(torch.abs(f_pred)))/ f_pred.numel()
        elif self.loss_style == 'sum':
            loss_u = torch.sum((self.u - u_pred) ** 2)
            loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
            if self.nu != 0:
                loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.sum(f_pred ** 2)
            # loss_f = torch.sum(torch.log((f_pred + 1) ** 2))
            # loss_f = torch.sum(torch.log(f_pred ** 2))
            # loss_f = torch.exp(torch.sum(torch.log(f_pred ** 2) ** p)) ** p

        u0_penalty_coeff = 1
        u0_x_penalty_coeff = 1
        # u0_xx_penalty_coeff = 1
        # jac_z_penalty_coeff = 1

        loss = self.UB * (loss_u + loss_b) + self.L*loss_f
        # if u_z is not None:
        #     jac_z_penalty = torch.mean(((u_z ** 2).mean(-1) - 1) ** 2)
        #     loss = loss + jac_z_penalty_coeff * jac_z_penalty
        #
        # # adding penalty on norm of gradients of u0 wrt x
        u0_x, u0_xx = self.net_u_derivatives(u_pred, self.x_u)
        u0_x_penalty = (torch.pow(u0_x, 2).mean(-2) - 1).pow(2).mean()
        u0_penalty = (torch.pow(u_pred, 2).mean(-2) - 1).pow(2).mean()
        # u0_xx_penalty = (torch.pow(u0_xx, 2).sum(-1) - 1).pow(2).mean()
        loss = loss\
               + u0_x_penalty_coeff * u0_x_penalty \
               + u0_penalty_coeff * u0_penalty
               # + u0_xx_penalty_coeff * u0_xx_penalty \

        pl_loss_coeff = 10
        pl_lengths = self.calc_pl_lengths(z_f, f_pred)
        pl_loss = ((pl_lengths - 1) ** 2).mean()
        loss += pl_loss_coeff * pl_loss

        if loss.requires_grad:
            loss.backward()

        grad_norm = 0
        for p in self.dnn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 100 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss: %.5e, loss_u: %.5e, loss_b: %.5e, loss_f: %.5e' % (self.iter, grad_norm, loss.item(), loss_u.item(), loss_b.item(), loss_f.item())
                )
            self.iter += 1

        return loss

    def train(self):
        self.dnn.train()
        self.optimizer.step(self.loss_pinn)

    def predict(self, X, z=None):

        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        if z is not None:
            z = z.to(device)
            z = z.unsqueeze(1).expand(-1, X.shape[0], -1)
            x = x.unsqueeze(0).expand(z.shape[0], -1, -1)
            t = t.unsqueeze(0).expand(z.shape[0], -1, -1)

        self.dnn.eval()
        u = self.net_u(x, t, z)
        u = u.detach().cpu().numpy()

        return u