import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder

import numpy as np

import random

class FeatDataset(DatasetFolder):
    EXTENSIONS = ['.npy']

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None):
        if loader is None:
            loader = self.__feat_loader

        super(FeatDataset, self).__init__(root, loader, self.EXTENSIONS,
                                         transform=transform,
                                         target_transform=target_transform)

    @staticmethod
    def __feat_loader(filename):
        npy = np.load(filename)
        npy = torch.tensor(npy)

        # crit = np.random.sample()

        # if(crit > 0.5):
        #     npy = torch.flip(npy,[1,2])

        return npy