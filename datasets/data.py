import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import pytorch_lightning as pl

class UnpairedDataset(Dataset):
	def __init__(self, exp, size, train):
		self.exp = exp
		transform = transforms.Compose([transforms.Resize(size),
										transforms.ToTensor(),
										transforms.Normalize((0.5), (0.5))])
		if train:
			split='train'
		else:
			split='test'
		if exp == 'mnist2svhn':
			data1 = datasets.MNIST(root='datasets/mnist', train=train,# download=True,
								   transform=transform)
			data2 = datasets.SVHN(root='datasets/svhn', split=split,# download=True,
								  transform=transform)

		self.datasets = (data1, data2)

	def __getitem__(self, i):
		if self.exp == 'mnist2svhn':
			# Use only green channel for SVHN
			return self.datasets[0][i][0], self.datasets[1][i][0][1].unsqueeze(0)

		return self.datasets[0][i], self.datasets[1][i]

	def __len__(self):
		return min(len(d) for d in self.datasets)

class LitData(pl.LightningDataModule):
	def __init__(self, exp, batch, workers):
		super().__init__()
		if exp == 'mnist2svhn':
			size=32
		self.train = UnpairedDataset(exp, size, True)
		self.test = UnpairedDataset(exp, size, False)
		self.batch = batch
		self.workers = workers

	def train_dataloader(self):
		return DataLoader(self.train, batch_size=self.batch, shuffle=True,
						  num_workers=self.workers, pin_memory=True)

	def val_dataloader(self):
		return DataLoader(self.test, batch_size=self.batch, shuffle=False,
						  num_workers=self.workers, pin_memory=True)

	def test_dataloader(self):
		return DataLoader(self.test, batch_size=self.batch, shuffle=False,
						  num_workers=self.workers, pin_memory=True)
