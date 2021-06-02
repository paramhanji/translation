import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

IMG_EXTENSIONS = (
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
	'.tif', '.TIF', '.tiff', '.TIFF',
)

def make_dataset(dir, max_dataset_size=float("inf")):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if fname.endswith(IMG_EXTENSIONS):
				path = os.path.join(root, fname)
				images.append(path)
	return images[:min(max_dataset_size, len(images))]


from survae.data.transforms import DynamicBinarize, Quantize
class UnpairedDataset(Dataset):
	def __init__(self, exp, channels, size, split):
		self.exp = exp
		if exp == 'mnist2svhn':
			transform = [transforms.Resize(size, Image.BICUBIC),
						 transforms.ToTensor(),
						 Quantize(8)]
						 # DynamicBinarize()])
			data1 = datasets.MNIST(root='datasets/mnist', train=(split == 'train'),
								   transform=transforms.Compose(transform))
			if channels == 1:
				transform = [transforms.Grayscale(1)] + transform
			data2 = datasets.SVHN(root='datasets/svhn', split=split,
								  transform=transforms.Compose(transform))
			self.datasets = (data1, data2)
		else:
			A_paths = sorted(make_dataset(f'datasets/{exp}/{split}A'))
			B_paths = sorted(make_dataset(f'datasets/{exp}/{split}B'))
			self.datasets = (A_paths, B_paths)
			self.transform = transforms.Compose([transforms.Resize(size, Image.BICUBIC),
												 transforms.ToTensor(),
												 Quantize(8)])
												 # transforms.Normalize(*[[0.5]*3]*2)])

	def __getitem__(self, i):
		if self.exp == 'mnist2svhn':
			# Ignore labels, use only images
			return self.datasets[0][i][0], self.datasets[1][i][0]

		A_path = self.datasets[0][i]
		B_path = self.datasets[1][i]
		A_img = self.transform(Image.open(A_path).convert('RGB'))
		B_img = self.transform(Image.open(B_path).convert('RGB'))
		return A_img, B_img

	def __len__(self):
		return min(len(d) for d in self.datasets)

class LitData(pl.LightningDataModule):
	def __init__(self, exp, channels, size, batch, workers):
		super().__init__()
		self.train = UnpairedDataset(exp, channels, size, 'train')
		self.test = UnpairedDataset(exp, channels, size, 'test')
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
