import argparse
import torch
import pytorch_lightning as pl

from networks import cycle_gan
from datasets.data import LitData

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'test'])
    parser.add_argument('--exp', type=str, default='mnist2svhn')

    # Data
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=3)

    # Hyper-parameters
    parser.add_argument('--epochs', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=42)
    
    # Misc
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    return args

if __name__ == '__main__':
    args = get_args()
    model = cycle_gan.TranslationModel(args.lr, (args.beta1, args.beta2))
    data = LitData(args.exp, args.batch, args.num_workers)

    if args.mode == 'train':
        trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, num_sanity_val_steps=1, deterministic=True,
                             resume_from_checkpoint=args.resume)
        trainer.fit(model, data)
