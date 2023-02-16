import torchvision
import torchvision.transforms as TF
import os
import torchvision.datasets
import numpy as np

from fls.datasets.DatasetSet import DatasetSet
from fls.datasets.SamplesDataset import SamplesDataset


class CifarDatasetSet(DatasetSet):
    def __init__(self):
        super().__init__()
        self.name = "CIFAR10"
        self.sample_datasets = []

        self.train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        self.train_dataset.name = "cifar10_train"

        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        self.train_dataset.name = "cifar10_train"
