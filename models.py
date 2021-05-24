import torch, pytorch_lightning as pl

from networks.gan import define_G, define_D, define_F
from networks.flow import define_flow
from networks.freia_survae import base_flow, survae_flow

from survae.utils import iwbo_nats

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


class CycleGAN(pl.LightningModule):
    def __init__(self, args, loss='l1'):
        super().__init__()
        self.forw_g = define_G(args.num_channels)
        self.back_g = define_G(args.num_channels)
        self.A_d = define_D(args.num_channels)
        self.B_d = define_D(args.num_channels)

        if loss=='l2':
            self.rec_loss = torch.nn.MSELoss()
        elif loss == 'l1':
            self.rec_loss = torch.nn.L1Loss()
        self.gan_loss = torch.nn.MSELoss()

        self.args = args

    def training_step(self, batch, batch_idx, optimizer_idx):
        A, B = batch

        if optimizer_idx == 0:
            # Train the generators
            set_requires_grad([self.A_d, self.B_d], False)

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
            set_requires_grad(self.A_d, True)

            real = self.A_d(A)
            loss_real = self.gan_loss(real, torch.ones_like(real))

            A_hat = self.back_g(B)
            fake = self.A_d(A_hat)
            loss_fake = self.gan_loss(fake, torch.zeros_like(fake))

            return loss_real + loss_fake

        elif optimizer_idx == 2:
            # Discriminator for domain B
            set_requires_grad(self.B_d, True)

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
        fraction = (epoch - self.args.epoch_decay) / self.args.epoch_decay
        return 1 if epoch < self.args.epoch_decay else 1 - fraction

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(list(self.forw_g.parameters()) + list(self.back_g.parameters()),
                                 self.args.lr, self.args.betas)
        d_A_opt = torch.optim.Adam(self.A_d.parameters(), self.args.lr, self.args.betas)
        d_B_opt = torch.optim.Adam(self.B_d.parameters(), self.args.lr, self.args.betas)

        # define the lr_schedulers
        g_sch   = torch.optim.lr_scheduler.LambdaLR(g_opt  , lr_lambda=self.lr_lambda)
        d_A_sch = torch.optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda=self.lr_lambda)
        d_B_sch = torch.optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda=self.lr_lambda)

        return [g_opt, d_A_opt, d_B_opt], [g_sch, d_A_sch, d_B_sch]


class CUTGAN(pl.LightningModule):
    def __init__(self, args, loss='l1', num_patches=256):
        super().__init__()
        self.f = define_F(args.num_channels)
        self.forw_g = define_G(args.num_channels)
        self.d = define_D(args.num_channels)

        if loss=='l2':
            self.rec_loss = torch.nn.MSELoss()
        elif loss == 'l1':
            self.rec_loss = torch.nn.L1Loss()
        self.gan_loss = torch.nn.MSELoss()
        self.num_patches = num_patches

        # NCE loss
        from loss import PatchNCELoss
        self.nce_layers = [int(i) for i in args.nce_layers.split(',')]
        self.criterionNCE = [PatchNCELoss(args.batch) for l in self.nce_layers]

        self.args = args

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.forw_g(tgt, self.nce_layers, encode_only=True)

        feat_k = self.forw_g(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.f(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.f(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def training_step(self, batch, batch_idx, optimizer_idx):
        A, B = batch

        if optimizer_idx == 0:
            # Train the generators
            set_requires_grad(self.d, False)
            B_hat = self.forw_g(A)

            gen_B = self.d(B_hat)
            loss_gen_B = self.gan_loss(gen_B, torch.ones_like(gen_B))
            loss_nce = self.calculate_NCE_loss(A, B_hat)

            return loss_gen_B + loss_nce

        elif optimizer_idx == 1:
            # Discriminator for domain A
            set_requires_grad(self.d, True)

            real = self.d(B)
            loss_real = self.gan_loss(real, torch.ones_like(real))

            B_hat = self.forw_g(A)
            fake = self.d(B_hat)
            loss_fake = self.gan_loss(fake, torch.zeros_like(fake))

            return 0.5*(loss_real + loss_fake)

    def validation_step(self, batch, batch_idx):
        A, B = batch
        loss = {}

        B_hat = self.forw_g(A)
        loss['nce'] = self.calculate_NCE_loss(A, B_hat)

        real = self.d(B)
        fake = self.d(B_hat)
        loss['real'] = self.gan_loss(real, torch.ones_like(real))
        loss['fake'] = self.gan_loss(fake, torch.zeros_like(fake))

        self.logger.log_metrics(loss)
        return torch.stack([loss[l] for l in loss]).sum()

    def forward(self, x):
        A, B = x
        B_hat = self.forw_g(A)

        return [B_hat]

    def lr_lambda(self, epoch):
        fraction = (epoch - self.args.epoch_decay) / self.args.epoch_decay
        return 1 if epoch < self.args.epoch_decay else 1 - fraction

    def configure_optimizers(self):
        g_f_opt = torch.optim.Adam(list(self.forw_g.parameters()) + list(self.f.parameters()),
                                 self.args.lr, self.args.betas)
        d_opt = torch.optim.Adam(self.d.parameters(), self.args.lr, self.args.betas)

        # define the lr_schedulers
        g_f_sch   = torch.optim.lr_scheduler.LambdaLR(g_f_opt , lr_lambda=self.lr_lambda)
        d_sch = torch.optim.lr_scheduler.LambdaLR(d_opt, lr_lambda=self.lr_lambda)

        return [g_f_opt, d_opt], [g_f_sch, d_sch]


class CycleFlow(pl.LightningModule):
    def __init__(self, args, loss='l1'):
        super().__init__()
        self.flow = define_flow(args.num_channels, num_blocks=6)
        self.A_d = define_D(args.num_channels)
        self.B_d = define_D(args.num_channels)

        if loss=='l2':
            self.rec_loss = torch.nn.MSELoss()
        elif loss == 'l1':
            self.rec_loss = torch.nn.L1Loss()
        self.gan_loss = torch.nn.MSELoss()

        from loss import JacobianClampingLoss
        self.jc_loss = JacobianClampingLoss(1, 20)

        self.args = args

    def training_step(self, batch, batch_idx, optimizer_idx):
        A, B = batch
        B_hat, A_hat = self.forward(batch)

        if optimizer_idx == 0:
            # Train the generators
            set_requires_grad([self.A_d, self.B_d], False)

            gen_A = self.A_d(A_hat)
            gen_B = self.B_d(B_hat)
            loss_gen_A = self.gan_loss(gen_A, torch.ones_like(gen_A))
            loss_gen_B = self.gan_loss(gen_B, torch.ones_like(gen_B))

            return loss_gen_A + loss_gen_B

        elif optimizer_idx == 1:
            # Discriminator for domain A
            set_requires_grad(self.A_d, True)

            real = self.A_d(A)
            loss_real = self.gan_loss(real, torch.ones_like(real))

            fake = self.A_d(A_hat)
            loss_fake = self.gan_loss(fake, torch.zeros_like(fake))

            return loss_real + loss_fake

        elif optimizer_idx == 2:
            # Discriminator for domain B
            set_requires_grad(self.B_d, True)

            real = self.B_d(B)
            loss_real = self.gan_loss(real, torch.ones_like(real))

            fake = self.B_d(B_hat)
            loss_fake = self.gan_loss(fake, torch.zeros_like(fake))

            return loss_real + loss_fake

    def validation_step(self, batch, batch_idx):
        A, B = batch
        loss = {}

        B_hat, A_hat = self.forward(batch)

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
        B_hat = torch.tanh(self.flow(A)[0])
        A_hat = torch.tanh(self.flow(B, reverse=True)[0])

        return B_hat, A_hat

    def lr_lambda(self, epoch):
        fraction = (epoch - self.args.epoch_decay) / self.args.epoch_decay
        return 1 if epoch < self.args.epoch_decay else 1 - fraction

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.flow.parameters(), self.args.lr, self.args.betas)
        d_A_opt = torch.optim.Adam(self.A_d.parameters(), self.args.lr, self.args.betas)
        d_B_opt = torch.optim.Adam(self.B_d.parameters(), self.args.lr, self.args.betas)

        # define the lr_schedulers
        g_sch   = torch.optim.lr_scheduler.LambdaLR(g_opt  , lr_lambda=self.lr_lambda)
        d_A_sch = torch.optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda=self.lr_lambda)
        d_B_sch = torch.optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda=self.lr_lambda)

        return [g_opt, d_A_opt, d_B_opt], [g_sch, d_A_sch, d_B_sch]


class Noise2AFlow(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.flow = survae_flow(args)
        self.args = args

    def training_step(self, batch, batch_idx):
        A, _ = batch
        nll = -self.flow.log_prob(A).mean()
        return nll

    def validation_step(self, batch, batch_idx):
        loss = {}
        A, _ = batch
        nll = -self.flow.log_prob(A).mean()
        loss['nll'] = nll
        loss['nats'] = iwbo_nats(self.flow, A, k=10)

        self.logger.log_metrics(loss)
        return loss['nll']

    # Do both forward and reverse passes
    def forward(self, x):
        A, _ = x
        num_samples = A.shape[0]
        A_hat = self.flow.sample(num_samples)

        return None, A_hat

    def lr_lambda(self, epoch):
        fraction = (epoch - self.args.epoch_decay) / self.args.epoch_decay
        return 1 if epoch < self.args.epoch_decay else 1 - fraction

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.flow.parameters(), self.args.lr, self.args.betas)
        # sch   = torch.optim.lr_scheduler.LambdaLR(opt  , lr_lambda=self.lr_lambda)

        return [opt] #, [sch]
