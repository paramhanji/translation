import argparse, wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

from models import CycleGAN
from datasets.data import LitData

class Visualizer(Callback):
    def __init__(self, src_imgs):
        self.src_imgs = src_imgs

    def on_validation_epoch_end(self, trainer, pl_module):
        src_imgs = [i.to(device=pl_module.device) for i in self.src_imgs]
        trans_imgs = pl_module(src_imgs)
        combined_AtoB = torch.cat((src_imgs[0], trans_imgs[0]), dim=-1)
        combined_BtoA = torch.cat((src_imgs[1], trans_imgs[1]), dim=-1)
        try:
            trainer.logger.experiment.log({'A->B':[wandb.Image(combined_AtoB)]})
            trainer.logger.experiment.log({'B->A':[wandb.Image(combined_BtoA)]})
        except AttributeError:
            pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'test'])
    parser.add_argument('model', type=str, choices=['cyclegan'])
    parser.add_argument('--exp', type=str, choices=['mnist2svhn', 'summer2winter'],
                        default='summer2winter')

    # Data
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=3)

    # Hyper-parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epoch_decay', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    
    # Misc
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--log_iter', type=int , default=10)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    return args

if __name__ == '__main__':
    args = get_args()
    model = CycleGAN(args.exp, args.lr, (args.beta1, args.beta2),
                     epoch_decay=args.epoch_decay)
    data = LitData(args.exp, args.batch, args.num_workers)
    val_sample = next(iter(data.val_dataloader()))
    
    if args.wandb:
        logger = WandbLogger(project='translation', name=f'{args.model}_{args.exp}')
        logger.log_hyperparams(args)
    else:
        logger = True

    if args.mode == 'train':
        trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, num_sanity_val_steps=1,
                             deterministic=True, resume_from_checkpoint=args.resume,
                             logger=logger, check_val_every_n_epoch=args.log_iter,
                             callbacks=[Visualizer(val_sample)])
        trainer.fit(model, data)
