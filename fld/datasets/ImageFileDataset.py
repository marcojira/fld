from torch.utils.data import Dataset
from PIL import Image


class ImageFileDataset(Dataset):
    """
    Dataset of single image (to easily get features of single image)
    """

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        with Image.open(self.path).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)
            return img, 0
