from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class NPYDataset(Dataset):
    """Creates torch Dataset from a .npy of images and a .npy labels"""

    def __init__(self, name, img_npy_path, label_npy_path, transform=None):
        self.name = name
        self.transform = transform
        self.files = np.load(img_npy_path, mmap_mode="c")
        self.labels = np.load(label_npy_path, mmap_mode="c")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.fromarray(self.files[idx])
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]
