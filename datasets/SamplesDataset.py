from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path


class SamplesDataset(Dataset):
    """ 
    Creates torch Dataset from directory of images.
    Must be structured as dir/<class>/<img_name>.<extension>
    """
    def __init__(self, name, path=False, extension="png", transform=None):
        self.name = name
        self.path = path
        self.transform = transform
        self.files = []

        for curr_path in Path(path).rglob(f"*.{extension}"):
            self.files.append((curr_path, curr_path.parent.name))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, class_id = self.files[idx]
        with Image.open(img_path) as img:
            img = np.array(img)
            if self.transform:
                img = self.transform(img)

            return img, int(class_id)
