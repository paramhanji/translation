import torch, pytorch_lightning as pl
from networks.gan import define_G, define_D

class CycleGAN(pl.LightningModule):
    def __init__(self, exp, lr, betas, epoch_decay, loss='l1'):
        super().__init__()
        if exp == 'mnist2svhn':
            num_channels = 1
        else:
            num_channels = 3
        self.forw_g = define_G(num_channels)
        self.back_g = define_G(num_channels)
        self.A_d = define_D(num_channels)
        self.B_d = define_D(num_channels)

        if loss=='l2':
            self.rec_loss = torch.nn.MSELoss()
        elif loss == 'l1':
            self.rec_loss = torch.nn.L1Loss()
        self.gan_loss = torch.nn.MSELoss()

        self.lr = lr
        self.betas = betas
        self.epoch_decay = epoch_decay

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

            return 10*(loss_cycle_A + loss_cycle_B) + loss_gen_A + loss_gen_B

        elif optimizer_idx == 1:
            # Discriminator for domain A
            self.set_requires_grad(self.A_d, True)

            real = self.A_d(A)
            loss_real = self.gan_loss(real, torch.ones_like(real))

            A_hat = self.back_g(B)
            fake = self.A_d(A_hat)
            loss_fake = self.gan_loss(fake, torch.zeros_like(fake))

            return loss_real + loss_fake

        elif optimizer_idx == 2:
            # Discriminator for domain B
            self.set_requires_grad(self.B_d, True)

            real = self.B_d(B)
            loss_real = self.gan_loss(real, torch.ones_like(real))

            B_hat = self.back_g(A)
            fake = self.B_d(B_hat)
            loss_fake = self.gan_loss(fake, torch.zeros_like(fake))

            return loss_real + loss_fake

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

    def lr_lambda(self, epoch):
        fraction = (epoch - self.epoch_decay) / self.epoch_decay
        return 1 if epoch < self.epoch_decay else 1 - fraction

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.forw_g.parameters()) + list(self.back_g.parameters()),
                                 self.lr, self.betas)
        d_A_opt = torch.optim.Adam(self.A_d.parameters(), self.lr, self.betas)
        d_B_opt = torch.optim.Adam(self.B_d.parameters(), self.lr, self.betas)

        # define the lr_schedulers
        g_sch   = torch.optim.lr_scheduler.LambdaLR(g_opt  , lr_lambda = self.lr_lambda)
        d_A_sch = torch.optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda = self.lr_lambda)
        d_B_sch = torch.optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda = self.lr_lambda)

        return [g_opt, d_A_opt, d_B_opt], [g_sch, d_A_sch, d_B_sch]
