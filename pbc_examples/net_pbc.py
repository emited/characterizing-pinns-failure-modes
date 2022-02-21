import math
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from choose_optimizer import *

# CUDA support
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

class LinearEstimator(nn.Module):
    def __init__(self, in_c, out_c, factor=1.0):
        super().__init__()
        self.factor = factor
        self.net = nn.Linear(in_c, out_c, bias=False)

    def forward(self, x):
        return self.net(x) * self.factor

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(10*input)


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
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU
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
        layer_list[-1][1].weight.data.zero_()
        print(layer_list[-1][1].weight)
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class Resnet(nn.Module):
    def __init__(self, in_features, out_features, block_layers, num_blocks, activation):
        super().__init__()
        self.blocks = []
        for _ in range(num_blocks):
            block = DNN(block_layers, activation)
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)
        self.first_linear = nn.Linear(in_features, block_layers[0])
        self.last_linear = nn.Linear(block_layers[-1], out_features)

    def forward(self, input):
        out = self.first_linear(input)
        for block in self.blocks:
            out = out + block(out)
        out =  self.last_linear(out)
        return out


class SymmetricInitDNN(DNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.cloned_layers = deepcopy(self.layers)

    def forward(self, inputs):
        return self.layers(inputs) - self.cloned_layers(inputs)


class LatentDNN(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False):
        super(LatentDNN, self).__init__()
        latent_dim = 2
        self.d = DNN([layers[0] - 1 + latent_dim] + layers[1:], activation, use_batch_norm, use_instance_norm)
        self.z = DNN([1, 128, latent_dim], activation, use_batch_norm, use_instance_norm)
        self.zphase = nn.Parameter(torch.linspace(0, 1, latent_dim).unsqueeze(0))
        self.zt = None

    def forward(self, x):
        x, t = x[:, [0]], x[:, [1]]
        zt = self.z(t)
        zt = torch.cos(2*5*math.pi*(zt + self.zphase))
        self.zt = zt
        self.t = t
        ut = self.d(torch.cat([x, zt], 1))
        return ut

class SeparationDNN(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False,
                 entangling_method='product'):
        super(SeparationDNN, self).__init__()
        self.entangling_method = entangling_method
        latent_dim = 2
        hidden_dim = 256
        if entangling_method == 'concat':
            latent_dim_factor = 2
        else:
            latent_dim_factor = 1

        # self.d = DNN([latent_dim] + [layers[-1]], 'identity', use_batch_norm, use_instance_norm)
        self.d = SymmetricInitDNN([latent_dim * latent_dim_factor, hidden_dim, 1],
                                  "relu", use_batch_norm, use_instance_norm)
        # self.d = Resnet(latent_dim * latent_dim_factor, 1, [hidden_dim, hidden_dim],
        #                 5, "relu")
        # self.d = SymmetricInitDNN([latent_dim] + [layers[-1]], 'identity', use_batch_norm, use_instance_norm)
        # self.d = DNN([latent_dim, hidden_dim, 1], "relu", use_batch_norm, use_instance_norm)
        # self.d = DNN([latent_dim, 512, 1], 'identity', use_batch_norm, use_instance_norm)
        # self.z = Resnet(1, latent_dim, [hidden_dim, hidden_dim], 5, activation)
        # self.p = Resnet(1, latent_dim, [hidden_dim, hidden_dim], 5, activation)
        self.z = DNN([1, hidden_dim, hidden_dim, latent_dim], activation, )
        self.p = DNN([1, hidden_dim, hidden_dim, latent_dim], activation,)
        # self.zphase = nn.Parameter(torch.linspace(0, 1, latent_dim).unsqueeze(0))
        self.zt = None

    def forward(self, x):
        x, t = x[:, [0]], x[:, [1]]
        zt, px = self.z(t), self.p(x)
        self.zt = zt
        self.px = px
        if self.entangling_method == 'product':
            entangling = zt * px
            self.prod = entangling
        elif self.entangling_method == 'concat':
            entangling = torch.concat([zt, px], dim=-1)
            self.prod = zt
        else:
            raise NotImplementedError(self.entangling_method)
        self.t = t
        self.x = x
        ut = self.d(entangling)
        return ut


class PhysicsInformedNN_pbc():
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho, optimizer_name, lr,
                 net, L=1, activation='tanh', loss_style='mean',
                 u_star=None, train_method='pinns', X_star=None):

        self.system = system
        self.train_method = train_method
        self.x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)
        self.net = net

        self.u_star = torch.tensor(u_star).float().to(device) # for regression to the solution directly
        self.X_star = torch.tensor(X_star).float().to(device)

        if self.net == 'DNN':
            # self.dnn = LatentDNN(layers, activation).to(device)
            self.dnn = SeparationDNN(layers, activation).to(device)
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

        self.lr = lr
        self.optimizer_name = optimizer_name

        self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)

        self.loss_style = loss_style

        self.iter = 0

    def net_u(self, x, t):
        """The standard DNN that takes (x,t) --> u."""
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        u = self.net_u(x, t)

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
            if any(self.G != 0):
                assert 'Neeeded to put back self.G'
            # f = u_t - self.nu*u_xx + self.beta*u_x - self.G
            f = u_t - self.nu*u_xx + self.beta*u_x
        elif 'rd' in self.system:
            f = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif 'reaction' in self.system:
            f = u_t - self.rho*u + self.rho*u**2
        return f

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

    def loss_regression(self, verbose=True):
        u_pred = self.net_u(self.X_star[:, [0]], self.X_star[:, [1]])
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        if self.loss_style == 'mean':
            loss_u = torch.mean((u_pred - self.u_star) ** 2)
        elif self.loss_style == 'sum':
            loss_u = torch.sum((u_pred - self.u_star) ** 2)
        loss = loss_u

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
                    'epoch %d, gradient: %.5e, loss: %.5e' % (self.iter, grad_norm, loss.item(), )
                )
            self.iter += 1

        return loss


    def loss_pinn(self, verbose=True):
        """ Loss function. """
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub)
        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
        f_pred = self.net_f(self.x_f, self.t_f)

        if self.loss_style == 'mean':
            loss_u = torch.mean((self.u - u_pred) ** 2)
            loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            if self.nu != 0:
                loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.mean(f_pred ** 2)
        elif self.loss_style == 'sum':
            loss_u = torch.sum((self.u - u_pred) ** 2)
            loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
            if self.nu != 0:
                loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.sum(f_pred ** 2)

        loss = loss_u + loss_b + self.L*loss_f

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
        if self.train_method == 'pinns':
            self.optimizer.step(self.loss_pinn)
        elif self.train_method == 'regression':
            self.optimizer.step(self.loss_regression)
        else:
            assert False


    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()

        return u

    def predict_f(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        f = self.net_f(x, t)
        f = f.detach().cpu().numpy() ** 2

        return f
