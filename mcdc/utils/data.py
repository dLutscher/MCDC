import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, SVHN
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from mcdc.utils.utils import create_dir, swapaxes

__all__ = ['get_dataset', 'get_data_loaders']


def get_dataset(cfg, fine_tune):
    data_transform = Compose([Resize((32, 32)), ToTensor()])
    mnist_transform = Compose([Resize((32, 32)), ToTensor(),
                               Lambda(lambda x: swapaxes(x, 1, -1))])
    vade_transform = Compose([ToTensor()])

    if cfg.DATA.DATASET == 'mnist':
        transform = vade_transform if 'vade' in cfg.DIRS.CHKP_PREFIX \
            else mnist_transform

        training_set = MNIST(download=True, root=cfg.DIRS.DATA,
                             transform=transform, train=True)
        val_set = MNIST(download=False, root=cfg.DIRS.DATA,
                        transform=transform, train=False)
        plot_set = copy.deepcopy(val_set)

    elif cfg.DATA.DATASET == 'svhn':
        training_set = SVHN(download=True,
                            root=create_dir(cfg.DIRS.DATA, 'SVHN'),
                            transform=data_transform,
                            split='train')
        val_set = SVHN(download=True,
                       root=create_dir(cfg.DIRS.DATA, 'SVHN'),
                       transform=data_transform,
                       split='test')
        plot_set = copy.deepcopy(val_set)

    elif cfg.DATA.DATASET == 'cifar':
        training_set = CIFAR10(download=True,
                               root=create_dir(cfg.DIRS.DATA, 'CIFAR'),
                               transform=data_transform,
                               train=True)
        val_set = CIFAR10(download=True,
                          root=create_dir(cfg.DIRS.DATA, 'CIFAR'),
                          transform=data_transform,
                          train=False)
        plot_set = copy.deepcopy(val_set)

    if 'idec' in cfg.DIRS.CHKP_PREFIX and fine_tune:
        training_set = IdecDataset(training_set)
        val_set = IdecDataset(val_set)
        plot_set = IdecDataset(plot_set)

    return training_set, val_set, plot_set


def get_data_loaders(cfg, drop_last=True, fine_tune=False):

    training_set, val_set, plot_set = get_dataset(cfg, fine_tune=fine_tune)

    train_loader = DataLoader(training_set,
                              batch_size=cfg.DATA.BATCH_SIZE,
                              shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_set,
                            batch_size=cfg.DATA.BATCH_SIZE,
                            shuffle=True, drop_last=drop_last)

    plot_loader = DataLoader(val_set,
                             batch_size=cfg.DATA.BATCH_SIZE,
                             shuffle=False, drop_last=drop_last)

    return train_loader, val_loader, plot_loader


class IdecDataset(Dataset):
    def __init__(self, dataset):
        super(IdecDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        img, target = self.dataset.__getitem__(index)
        return img, target, index

    def __len__(self):
        return self.dataset.data.shape[0]
