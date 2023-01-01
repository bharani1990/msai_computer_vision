import os
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

BATCH_SIZE = 256
NUM_WORKERS = 2
MEANS = np.array((0.4914, 0.4822, 0.4465))
STDS = np.array((0.2023, 0.1994, 0.2010))

base_transforms = [transforms.ToTensor(), transforms.Normalize(MEANS, STDS)]
augmented_transforms = [
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(hue=0.01, brightness=0.3, contrast=0.3, saturation=0.3),
]
augmented_transforms += base_transforms
transform_basic = transforms.Compose(base_transforms)
transform_augment = transforms.Compose(augmented_transforms)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path='./'):
        super().__init__()
        self.data_path = data_path
        
    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path,
                         download=True)

        self.train_transform = transform_augment
        self.test_transform = transform_basic

    def setup(self, stage=None):
        train = datasets.CIFAR10(root=self.data_path, 
                                 train=True, 
                                 transform=self.train_transform,
                                 download=False)

        self.test = datasets.CIFAR10(root=self.data_path, 
                                     train=False, 
                                     transform=self.test_transform,
                                     download=False)

        self.train, self.valid = random_split(train, lengths=[40000, 10000])

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train, 
                                  batch_size=BATCH_SIZE, 
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(dataset=self.valid, 
                                  batch_size=BATCH_SIZE, 
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test, 
                                 batch_size=BATCH_SIZE, 
                                 drop_last=False,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS)
        return test_loader
