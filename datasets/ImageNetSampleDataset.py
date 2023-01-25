from torch.utils.data import Dataset
import os
from PIL import Image
import pickle


class ImageNetSampleDataset(Dataset):
    """
    Dataset of ImageNet samples
    """

    def __init__(self, name, split="train"):
        self.name = name
        self.transform = None

        self.path = os.path.join(
            "/home/mila/m/marco.jiralerspong/scratch/sample_imagenet", split
        )
        self.files = os.listdir(self.path)
        self.files = {
            int(file.split(".")[0]): file for file in self.files if ".png" in file
        }

        with open(os.path.join(self.path, "classes.pkl"), "rb") as f:
            self.classes = pickle.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        with Image.open(os.path.join(self.path, img_path)).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)

            return img, self.classes[idx]

    def get_class_idxs(self, class_idx):
        idxs = []

        for k, v in self.classes.items():
            if int(v) == class_idx:
                idxs.append(k)

        return idxs
