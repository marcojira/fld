from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class SamplesDataset(Dataset):
    """
    Creates torch Dataset from directory of images.
    Must be structured as dir/<class>/<img_name>.<extension> for `conditional=True
    """

    def __init__(
        self, name, path=False, extension="png", transform=None, conditional=False
    ):
        self.name = name
        self.path = path
        self.extension = extension

        self.conditional = conditional
        self.transform = transform
        self.files = []
        self.files_loaded = False

    def load_files(self):
        for curr_path in Path(self.path).rglob(f"*.{self.extension}"):
            if self.conditional:
                self.files.append((curr_path, curr_path.parent.name))
            else:
                self.files.append((curr_path, 0))
        self.files_loaded = True

    def __len__(self):
        if not self.files_loaded:
            self.load_files()

        return len(self.files)

    def __getitem__(self, idx):
        if not self.files_loaded:
            self.load_files()

        img_path, class_id = self.files[idx]
        with Image.open(img_path).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)
            return img, int(class_id)
