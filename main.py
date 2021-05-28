import argparse, wandb
import torch, numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import torchvision
import matplotlib.pyplot as plt

from models import *
from datasets.data import LitData

class Visualizer(Callback):
    def __init__(self, src_imgs, exp):
        self.src_imgs = src_imgs
        self.bijective = exp != 'cutgan'

    def on_validation_epoch_end(self, trainer, pl_module):
        src_imgs = [i.to(device=pl_module.device) for i in self.src_imgs]
        trans_imgs = pl_module(src_imgs)
        # combined_AtoB = torch.cat((src_imgs[0], trans_imgs[0]), dim=-1)
        if self.bijective:
            combined_BtoA = torch.cat((src_imgs[1], trans_imgs[1]), dim=-1)
        try:
            # trainer.logger.experiment.log({'A->B':[wandb.Image(combined_AtoB)]})
            # trainer.logger.experiment.log({'A':[wandb.Image(src_imgs[0])]})
            # trainer.logger.experiment.log({'Noise_hat':[wandb.Image(trans_imgs[0])]})
            if self.bijective:
                # trainer.logger.experiment.log({'B->A':[wandb.Image(combined_BtoA)]})
                # trainer.logger.experiment.log({'Noise':[wandb.Image(src_imgs[1])]})
                trainer.logger.experiment.log({'A_hat':[wandb.Image(trans_imgs[1].float())]})
        except AttributeError:
            pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'test'])
    parser.add_argument('model', type=str,
                        choices=['cyclegan', 'cutgan', 'cycleflow', 'noise2Aflow'])
    parser.add_argument('--exp', type=str, choices=['mnist2svhn', 'summer2winter'],
                        default='summer2winter')

    # Data
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=3)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    # Hyper-parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--betas', type=float, nargs=2, default=[0.5, 0.999])
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
    if args.exp == 'mnist2svhn':
        assert args.num_channels == 1, \
               'Mismatch between experiment and number of channels'

    return args

if __name__ == '__main__':
    args = get_args()
    models = {'cyclegan': CycleGAN, 'cutgan': CUTGAN, 'cycleflow': CycleFlow,
              'noise2Aflow': Noise2AFlow}
    model = (models[args.model])(args)
    data = LitData(args.exp, args.size, args.batch, args.num_workers)
    val_sample = next(iter(data.val_dataloader()))

    if args.mode == 'train':
        if args.wandb:
            logger = WandbLogger(project='translation', name=f'{args.model}_{args.exp}')
            logger.log_hyperparams(args)
        else:
            logger = True
    elif args.mode == 'test':
        logger = False
        assert args.resume is not None

    if args.mode == 'train':
        trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, num_sanity_val_steps=1,
                             deterministic=True, resume_from_checkpoint=args.resume,
                             logger=logger, check_val_every_n_epoch=args.log_iter,
                             callbacks=[Visualizer(val_sample, args.exp)])
        trainer.fit(model, data)

    elif args.mode == 'test':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
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
