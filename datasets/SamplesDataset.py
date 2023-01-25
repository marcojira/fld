from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


class SamplesDataset(Dataset):
    def __init__(self, name, path=False, extension="png", transform=None):
        self.name = name
        self.path = path
        self.transform = transform
        self.files = []

        for curr_path in Path(path).rglob(f"*.{extension}"):
            self.files.append(str(curr_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        with Image.open(img_path).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)

            return img, 0  # Assuming unconditional for now
