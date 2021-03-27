import torch
import torch.nn.functional as F

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(torch.nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(torch.nn.BatchNorm2d(c_out))
    return torch.nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(torch.nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(torch.nn.BatchNorm2d(c_out))
    return torch.nn.Sequential(*layers)

class G(torch.nn.Module):
    """Generator for translating domains"""
    def __init__(self, conv_dim=64, in_out_dim=1):
        super().__init__()
        # encoding blocks
        self.conv1 = conv(in_out_dim, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        
        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        
        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, in_out_dim, 4, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)
        
        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )
        
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = torch.tanh(self.deconv2(out))              # (?, c, 32, 32)
        return out

class D(torch.nn.Module):
    """Discriminator for both domains."""
    def __init__(self, conv_dim=64, in_dim=1):
        super().__init__()
        self.conv1 = conv(in_dim, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.fc = conv(conv_dim*4, 1, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out

import pytorch_lightning as pl
class TranslationModel(pl.LightningModule):
    def __init__(self, exp, lr, betas, loss='l1'):
        super().__init__()
        if exp == 'mnist2svhn':
            num_channels = 1
        else:
            num_channels = 3
        self.forw_g = G(in_out_dim=num_channels)
        self.back_g = G(in_out_dim=num_channels)
        self.A_d = D(in_dim=num_channels)
        self.B_d = D(in_dim=num_channels)

        if loss=='l2':
            self.rec_loss = torch.nn.MSELoss()
        elif loss == 'l1':
            self.rec_loss = torch.nn.L1Loss()
        self.gan_loss = torch.nn.MSELoss()

        self.lr = lr
        self.betas = betas

    @staticmethod
    def set_requires_grad(nets, requires_grad):
        """
        Set requies_grad for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def training_step(self, batch, batch_idx, optimizer_idx):
        A, B = batch

        if optimizer_idx == 0:
            # Train the generators
            self.set_requires_grad([self.A_d, self.B_d], False)
            B_hat = self.forw_g(A)
            A_hat = self.back_g(B)

            B_hat_hat = self.forw_g(A_hat)
            A_hat_hat = self.back_g(B_hat)
            loss_cycle_A = self.rec_loss(A, A_hat_hat)
            loss_cycle_B = self.rec_loss(B, B_hat_hat)

            gen_A = self.A_d(A_hat)
            gen_B = self.B_d(B_hat)
            loss_gen_A = self.gan_loss(gen_A, torch.ones_like(gen_A))
            loss_gen_B = self.gan_loss(gen_B, torch.ones_like(gen_B))

            return 10*(loss_cycle_A + loss_cycle_B) + 0.5*(loss_gen_A + loss_gen_B)

        elif optimizer_idx == 1:
            # Discriminator for domain A
            self.set_requires_grad(self.A_d, True)

            real = self.A_d(A)
            loss_real = self.gan_loss(real, torch.ones_like(real))

            A_hat = self.back_g(B)
            fake = self.A_d(A_hat)
            loss_fake = self.gan_loss(fake, torch.zeros_like(fake))

            return 0.5*(loss_real + loss_fake)

        elif optimizer_idx == 2:
            # Discriminator for domain B
            self.set_requires_grad(self.B_d, True)

            real = self.B_d(B)
            loss_real = self.gan_loss(real, torch.ones_like(real))

            B_hat = self.back_g(A)
            fake = self.B_d(B_hat)
            loss_fake = self.gan_loss(fake, torch.zeros_like(fake))

            return 0.5*(loss_real + loss_fake)

    def validation_step(self, batch, batch_idx):
        A, B = batch
        loss = {}

        B_hat = self.forw_g(A)
        A_hat = self.back_g(B)

        B_hat_hat = self.forw_g(A_hat)
        A_hat_hat = self.back_g(B_hat)
        loss['cycle_A'] = self.rec_loss(A, A_hat_hat)
        loss['cycle_B'] = self.rec_loss(B, B_hat_hat)

        real_A = self.A_d(A)
        real_B = self.B_d(B)
        fake_A = self.A_d(A_hat)
        fake_B = self.B_d(B_hat)
        loss['real_A'] = self.gan_loss(real_A, torch.ones_like(real_A))
        loss['real_B'] = self.gan_loss(real_B, torch.ones_like(real_B))
        loss['fake_A'] = self.gan_loss(fake_A, torch.zeros_like(fake_A))
        loss['fake_B'] = self.gan_loss(fake_B, torch.zeros_like(fake_B))

        self.logger.log_metrics(loss)
        return torch.stack([loss[l] for l in loss]).sum()

    def forward(self, x):
        A, B = x
        B_hat = self.forw_g(A)
        A_hat = self.back_g(B)

        return B_hat, A_hat


    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.forw_g.parameters()) + list(self.back_g.parameters()),
                                 self.lr, self.betas)
        d_A_opt = torch.optim.Adam(self.A_d.parameters(), self.lr, self.betas)
        d_B_opt = torch.optim.Adam(self.B_d.parameters(), self.lr, self.betas)
        
        return [g_opt, d_A_opt, d_B_opt]
