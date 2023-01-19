from torch.utils.data import Dataset
import torch
import os
import glob
import torchvision.transforms as transforms
from PIL import Image


class RandomDataset(Dataset):

    def __init__(self):
        self.name = "random"
    
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), 0