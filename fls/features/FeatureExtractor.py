from __future__ import annotations

import pickle
import torch
import os
import math
import numpy as np
import torchvision

from pathlib import Path
from tqdm import tqdm
from PIL import Image

from fls.datasets.SamplesDataset import SamplesDataset
from pathlib import Path

NUM_WORKERS = 4
BATCH_SIZE = 256
CHUNK_SIZE = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_img(img_array):
    """From tensor in [-1, 1] of shape [B, C, W, H] to numpy int images"""
    return (
        (img_array * 127.5 + 128)
        .clip(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


class TransformedDataset(torch.utils.data.Dataset):
    """Wrapper class for dataset to add preprocess transform"""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, labels = self.dataset[idx]
        img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.dataset)


class FeatureExtractor:
    def __init__(self, save_path: str | None):
        if save_path is None:
            current_file_path = Path(__file__).absolute()
            save_path = os.path.join(current_file_path.parent, "cached_features")

        self.save_path = os.path.join(save_path, self.name)
        os.makedirs(self.save_path, exist_ok=True)

        # TO BE IMPLEMENTED BY EACH MODULE
        self.features_size = None
        self.preprocess = None

    def get_feature_batch(self, img_batch: torch.Tensor):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_features(
        self, imgs: torch.utils.data.Dataset | torch.Tensor, name=None, recompute=False
    ):
        """
        Gets the features from imgs (either a Dataset) or a tensor of images.
        - cache: Whether to load/save features to a cache
        - name: Unique name of set of images for caching purposes
        """
        if name:
            file_path = os.path.join(self.save_path, f"{name}.pt")

        if name and not recompute:
            if os.path.exists(file_path):
                return torch.load(file_path)

        if isinstance(imgs, torch.utils.data.Dataset):
            features = self.get_dataset_features(imgs)
        elif isinstance(imgs, torch.Tensor):
            features = self.get_tensor_features(imgs)
        else:
            raise NotImplementedError(f"Cannot get features from {type(imgs)}")

        if name:
            torch.save(features, file_path)

        return features

    def get_dataset_features(self, dataset: torch.utils.data.Dataset):
        size = len(dataset)
        features = torch.zeros(size, self.features_size)

        # Add preprocessing to dataset transforms
        dataset = TransformedDataset(dataset, self.preprocess)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            drop_last=False,
            num_workers=NUM_WORKERS,
            shuffle=False,
        )

        start_idx = 0
        for img_batch, _ in tqdm(dataloader, leave=False):
            feature = self.get_feature_batch(img_batch.to(DEVICE))

            # If going to overflow, just get required amount and break
            if size and start_idx + feature.shape[0] > size:
                features[start_idx:] = feature[: size - start_idx]
                break

            features[start_idx : start_idx + feature.shape[0]] = feature
            start_idx = start_idx + feature.shape[0]

        return features

    def get_tensor_features(self, img_tensor: torch.Tensor):
        num_samples = img_tensor.shape[0]
        num_batches = math.ceil(num_samples / BATCH_SIZE)
        features = torch.zeros(num_samples, self.features_size)

        def transform_img(img: torch.tensor):
            img = torchvision.transforms.ToPILImage()(img)
            img = self.preprocess(img)
            img = img.unsqueeze(0)
            return img

        for i in tqdm(range(num_batches), leave=False):
            idx_beg = BATCH_SIZE * i
            idx_end = BATCH_SIZE * (i + 1)

            img_batch = img_tensor[idx_beg:idx_end]
            img_batch = [transform_img(img) for img in img_batch]
            img_batch = torch.cat(img_batch, dim=0).to(DEVICE)
            features[idx_beg:idx_end] = self.get_feature_batch(img_batch)

        return features

    def get_dir_features(self, dir: str, name=None):
        dataset = SamplesDataset(dir)
        feat = self.get_features(dataset, name)
        return feat

    def get_model_features(self, gen_fn: function, num_samples: int):
        img_tensor = gen_fn()
        batch_size = img_tensor.shape[0]

        num_batches = math.ceil(num_samples / batch_size)
        features = torch.zeros(num_batches * batch_size, self.features_size)

        for i in tqdm(range(num_batches)):
            img_tensor = gen_fn()
            curr_features = self.get_features(img_tensor)
            features[i * batch_size : (i + 1) * batch_size] = curr_features

        return features[:num_samples]
