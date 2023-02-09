import pickle
import torch
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms as TF


class TransformedDataset(torch.utils.data.Dataset):
    """Wrapper class for dataset to add transform when there is none"""

    def __init__(self, dataset, transform):
        self.dataset = dataset

        # Ensures None transform isn't added
        if hasattr(dataset, "transform") and dataset.transform:
            self.transform = TF.Compose([dataset.transform, transform])
        else:
            self.transform = transform

    def __getitem__(self, idx):
        img, labels = self.dataset[idx]
        return self.transform(img), labels

    def __len__(self):
        return len(self.dataset)


class FeatureExtractor:
    def __init__(self, recompute=False, save=True):
        self.recompute = recompute
        self.save = save
        self.path = (
            f"/home/mila/m/marco.jiralerspong/scratch/ills/activations/{self.name}/"
        )
        os.makedirs(self.path, exist_ok=True)
        self.features_size = None  # TO BE IMPLEMENTED BY EACH MODULE

    def preprocess_batch(self, img_batch):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_feature_batch(self, img_batch):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_features(self, dataset, size=None, get_indices=False):
        """Gets shuffled features from dataset of size `size` or all if size is None"""
        file_path = os.path.join(self.path, f"{dataset.name}.pkl")

        # Get features
        if not self.recompute and os.path.exists(file_path):
            features = self.load_features(file_path)
        else:
            features = self.compute_features(dataset)
            if self.save:
                self.save_features(features, file_path)

        # Return shuffled
        size = min(size, len(features)) if size else len(features)
        random_sample = np.random.choice(len(features), size=size, replace=False)

        if get_indices:
            return features[random_sample], random_sample

        return features[random_sample]

    def compute_features(self, dataset):
        """Return tensor of feature values for data points in dataset"""
        dataset = TransformedDataset(dataset, self.preprocess_batch)

        size = len(dataset)
        features = torch.zeros(size, self.features_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=256, drop_last=False, num_workers=1, shuffle=False
        )

        start_idx = 0
        for img_batch, _ in tqdm(dataloader, leave=False):
            feature = self.get_feature_batch(img_batch.cuda())

            # If going to overflow, just get required amount and break
            if size and start_idx + feature.shape[0] > size:
                features[start_idx:] = feature[: size - start_idx]
                break

            features[start_idx : start_idx + feature.shape[0]] = feature
            start_idx = start_idx + feature.shape[0]

        return features

    def save_features(self, features, file_path):
        with open(file_path, "wb") as f:
            pickle.dump((features), f)
            return features

    def load_features(self, path):
        with open(path, "rb") as f:
            features = pickle.load(f)

        if type(features) is tuple:
            return features[0]

        return features
