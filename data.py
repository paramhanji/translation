import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

from survae.data.transforms import Quantize
from survae.data.loaders.toy import Gaussian, Crescent, CrescentCubed, SineWave, Abs, Sign, FourCircles, Diamond, TwoSpirals, TwoMoons, Checkerboard, Face

class UnpairedDataset(Dataset):
	def __init__(self, exp, channels, size, split, train_domain=None):
		self.exp = exp
		transform = [transforms.Resize(size, Image.BICUBIC),
					 transforms.ToTensor(),
					 Quantize(8)]

		if exp.startswith('mnist'):
			dataA = datasets.MNIST(root='datasets/mnist', train=(split == 'train'),
								   transform=transforms.Compose(transform))
		elif exp.startswith('cifar'):
			assert channels == 3
			dataA = datasets.CIFAR10(root='datasets/cifar', train=(split == 'train'),
								   transform=transforms.Compose(transform))
			dataB = datasets.CIFAR10(root='datasets/cifar', train=(split == 'train'),
								  transform=transforms.Compose(transform))
		elif exp.startswith('crescent'):
			data = Crescent(train_samples=128*1000, test_samples=10000)
			dataA = data.train if split == 'train' else data.test

		if exp.endswith('usps'):
			assert channels == 1
			dataB = datasets.USPS(root='datasets/usps', train=(split == 'train'),
								  transform=transforms.Compose(transform))
		elif exp.endswith('svhn'):
			if channels == 1:
				transform = [transforms.Grayscale(1)] + transform
			dataB = datasets.SVHN(root='datasets/svhn', split=split,
								  transform=transforms.Compose(transform))
		elif exp.endswith('cubed'):
			data = CrescentCubed(train_samples=128*1000, test_samples=128*1000)
			dataB = data.train if split == 'train' else data.test

		self.datasets = (dataA, dataB)

		if train_domain == 'A':
			self.length = len(dataA)
		elif train_domain == 'B':
			self.length = len(dataB)
		else:
			self.length = min(len(d) for d in self.datasets)

	def __getitem__(self, i):
		if self.exp.startswith(('mnist', 'cifar')):
			# Ignore labels, use only images
			return [d[i%min(len(d), self.length)][0] for d in self.datasets]
		elif self.exp.startswith('crescent'):
			return self.datasets[0][i], self.datasets[1][i]

	def __len__(self):
		return self.length

class LitData(pl.LightningDataModule):
	def __init__(self, exp, channels, size, batch, train_domain):
		super().__init__()
		self.train = UnpairedDataset(exp, channels, size, 'train', train_domain)
		self.test = UnpairedDataset(exp, channels, size, 'test')
		self.batch = batch

	def train_dataloader(self):
		return DataLoader(self.train, batch_size=self.batch, shuffle=True,
						  pin_memory=True, num_workers=4)

	def val_dataloader(self):
		return DataLoader(self.test, batch_size=self.batch, shuffle=False,
						  pin_memory=True, num_workers=4)

	def test_dataloader(self):
		return DataLoader(self.test, batch_size=self.batch, shuffle=False,
						  pin_memory=True, num_workers=4)
