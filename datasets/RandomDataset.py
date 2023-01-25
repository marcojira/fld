from torch.utils.data import Dataset
import torch
import os
import glob
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class RandomDataset(Dataset):
    def __init__(self):
        self.name = "random"
        self.transform = None

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        img_array = np.random.randint(0, 255, (32, 32, 3))
        img = Image.fromarray(img_array.astype("uint8"), "RGB")

        if self.transform:
            img = self.transform(img)

        return img, 0
