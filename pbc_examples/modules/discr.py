import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ToyDiscriminator(nn.Module):
    def __init__(self, ndim, ndf, use_spectral_norm=True):
        super().__init__()

        if not use_spectral_norm:
            self.mod = nn.Sequential(
                nn.Linear(ndim, ndf),
                nn.LeakyReLU(0.2),
                nn.Linear(ndf, ndf),
                nn.LeakyReLU(0.2),
                nn.Linear(ndf, 1),
                nn.Sigmoid()
            )
        else:
            self.mod = nn.Sequential(
                spectral_norm(nn.Linear(ndim, ndf)),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Linear(ndf, ndf)),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Linear(ndf, 1)),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.mod(x)

class ToyDiscriminator(nn.Module):
    def __init__(self, ndim, ndf, use_spectral_norm=True):
        super().__init__()

        if not use_spectral_norm:
            self.mod = nn.Sequential(
                nn.Conv1d(ndim, ndf, kernel_size=7, stride=2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(ndf, ndf, kernel_size=7, stride=2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(ndf, ndf, kernel_size=7, stride=2),
                nn.LeakyReLU(0.2),
                nn.Sigmoid()
            )
        else:
            self.mod = nn.Sequential(
                spectral_norm(nn.Conv1d(ndim, ndf, kernel_size=7, stride=2)),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv1d(ndf, ndf, kernel_size=7, stride=2)),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv1d(ndf, ndf, kernel_size=7, stride=2)),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv1d(ndf, 1, kernel_size=7, stride=2)),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = x.unsqueeze(-2)
        out =  self.mod(x)
        out = out.squeeze(-2)
        return out



# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.real_label = torch.tensor(target_real_label)
        self.fake_label = torch.tensor(target_fake_label)
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real, is_disc=False):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            labels = (self.real_label if target_is_real else self.fake_label).expand_as(prediction).type_as(
                prediction).to(prediction)
            loss = self.loss(prediction, labels)
        elif self.gan_mode in ['hinge', 'wgangp']:
            if is_disc:
                if target_is_real:
                    prediction = -prediction
                if self.gan_mode == 'hinge':
                    loss = self.loss(1 + prediction).mean()
                elif self.gan_mode == 'wgangp':
                    loss = prediction.mean()
            else:
                loss = -prediction.mean()
        return loss


def loss_dis(self, dis, xd, xt, zd, zt, **bin):
    t = xt if self.lmbda_gan_x > 0 else zt
    d = xd if self.lmbda_gan_x > 0 else zd
    loss_real = self.criterion_gan(dis(d), True)
    loss_fake = self.criterion_gan(dis(t.detach()), False)
    loss = 0.5 * (loss_fake + loss_real)
    return loss

def optimize_dis(self, A, B, **bin):
    self.optim_dis.zero_grad()
    losses_A, losses_B = {}, {}
    loss_A = self.loss_dis(self.dis_B, **A)
    losses_A['loss_dis'] = loss_A
    loss = loss_A
    if self.optimize_on_B:
        loss_B = self.loss_dis(self.dis_A, **B)
        losses_B['loss_dis'] = loss_B
        loss = loss_A + loss_B
    loss.backward()
    self.optim_dis.step()
    return losses_A, losses_B


def optimize(self, x_A, x_B, **bin):
    set_requires_grad([self.dis_A, self.dis_B], False)
    A, B = self.infer(x_A, x_B)
    losses_phi_A, losses_phi_B = self.optimize_phi(A, B)
    losses_enc_dec_A, losses_enc_dec_B = self.optimize_enc_dec(A, B)

    set_requires_grad([self.dis_A, self.dis_B], True)
    losses_dis_A, losses_dis_B = self.optimize_dis(A, B)
    losses_A = {**losses_phi_A, **losses_enc_dec_A, **losses_dis_A}
    losses_B = {**losses_phi_B, **losses_enc_dec_B, **losses_dis_B}
    return losses_A, losses_B


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad