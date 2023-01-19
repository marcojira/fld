from torch.utils.data import Dataset
import torch
import os
import glob
import torchvision.transforms as transforms
from PIL import Image


class SamplesDataset(Dataset):

    def __init__(self, name, path=False, subfolders=False, batch_mode=False, transform=None):
        self.name = name
        if path:
            self.path = path
        else:
            self.path = os.path.join("/home/mila/m/marco.jiralerspong/scratch/ills/sample_generation/image_samples/", name)
        self.batch_mode = batch_mode
        
        self.transform = transform
        self.subfolders = subfolders
        self.files = []
        
        
        if self.subfolders:
            glob_pattern = os.path.join(self.path, "*/*.png")
        else:
            glob_pattern = os.path.join(self.path, "*.png")
        
        self.files = glob.glob(glob_pattern)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        
        with Image.open(img_path) as img:
            if self.transform:
                img = self.transform(img)

            return img, 0 # Assuming unconditional for now