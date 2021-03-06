import argparse, wandb, os
import torch, numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import torchvision
import matplotlib.pyplot as plt

from models import models, Noise2AFlow
from data import LitData

class Visualizer(Callback):
    def __init__(self, loader, exp):
        self.loader = loader
        self.src_imgs = next(iter(loader))
        self.exp = exp

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.exp == 'crescent2cubed':
            fig = pl_module.plot(self.loader)
            wandb_imgs = {'plot': [wandb.Image(fig)]}
            plt.close(fig)
        else:
            src_imgs = [i.to(device=pl_module.device) for i in self.src_imgs]
            trans_imgs = pl_module(src_imgs)
            wandb_imgs = {'A_hat': [wandb.Image(trans_imgs[1].float())]}

        if type(trainer.logger).__name__ == 'WandbLogger':
            trainer.logger.experiment.log(wandb_imgs)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'test'])
    parser.add_argument('model', type=str,
                        choices=['cyclegan', 'cutgan', 'cycleflow', 'noise2Aflow',
                                 'A2Bflow'])
    parser.add_argument('--exp', type=str, default='summer2winter',
                        choices=['mnist2usps', 'mnist2svhn', 'cifar',
                                 'crescent2cubed', 'summer2winter'])
    parser.add_argument('--pretrained-flow', type=str)

    # Data
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--train-domain', default=None, choices=['A', 'B'])

    # Hyper-parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch-decay', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nce-layers', type=str, default='0,4,8,12,16')

    # Misc
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--log-iter', type=int , default=10)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    return args

if __name__ == '__main__':
    args = get_args()
    if args.model == 'noise2Aflow':
        assert args.train_domain is not None, 'Choose domain to learn first'
    elif args.model == 'A2Bflow':
        assert os.path.exists(args.pretrained_flow)
        assert args.train_domain is None
        checkpoint = torch.load(args.pretrained_flow)
        flow = Noise2AFlow(args)
        flow.load_state_dict(checkpoint['state_dict'])
        flow.eval()
        for p in flow.parameters():
            p.requires_grad = False
        args.pretrained_flow = flow.flow
    model = (models[args.model])(args)
    data = LitData(args.exp, args.num_channels, args.size, args.batch, args.train_domain)

    if args.mode == 'train':
        logger = False
        if args.wandb:
            logger = WandbLogger(project='translation', name=f'{args.model}_{args.exp}')
            logger.log_hyperparams(args)
        trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, num_sanity_val_steps=1,
                             deterministic=True, resume_from_checkpoint=args.resume,
                             logger=logger, check_val_every_n_epoch=args.log_iter,
                             callbacks=[Visualizer(data.val_dataloader(), args.exp)])
        trainer.fit(model, data)

    elif args.mode == 'test':
        assert args.resume is not None
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        if args.exp == 'crescent2cubed':
            fig = model.plot(data.test_dataloader())
            plt.show()
            plt.close(fig)
        else:
            test_sample = next(iter(data.test_dataloader()))
            grid_A_real = torchvision.utils.make_grid(test_sample[0], nrow=8).permute(1, 2, 0)
            grid_B_real = torchvision.utils.make_grid(test_sample[1], nrow=8).permute(1, 2, 0)
            nll_A_real, imgs = model(test_sample)
            nll_A_gen, _ = model((imgs, None))
            nll_B_real, _ = model((test_sample[1], None))
            grid_A_gen = torchvision.utils.make_grid(imgs, nrow=8).permute(1, 2, 0)
            grid = np.concatenate((grid_A_real, grid_A_gen, grid_B_real), axis=1).astype(np.float32)
            print(f'Real A: {nll_A_real.mean()}, gen A: {nll_A_gen.mean()}, real B: {nll_B_real.mean()}')
            plt.imshow(grid)
            plt.show()
            plt.close()
