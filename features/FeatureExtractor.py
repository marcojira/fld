import pickle
import torch
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms as TF


class FeatureExtractor:
    def __init__(self, recompute=False, save=True):
        self.recompute = recompute
        self.save = save
        self.path = (
            f"/home/mila/m/marco.jiralerspong/scratch/ills/activations/{self.name}/"
        )
        os.makedirs(self.path, exist_ok=True)

    def preprocess_batch(self, img_batch):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_feature_batch(self, img_batch):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_features(self, dataset, size=None, get_indices=False, verbose=False):
        file_path = os.path.join(self.path, f"{dataset.name}.pkl")

        if not self.recompute and os.path.exists(file_path):
            if verbose:
                print(f"Loading {file_path} from cache")
            features, idxs = self.load_features(file_path)
        else:
            features, idxs = self.compute_features(dataset, size)
            if self.save:
                self.save_features(features, idxs, file_path)

        if size and size < len(features):
            random_sample = np.random.choice(len(features), size=size, replace=False)
        else:
            random_sample = range(len(features))

        if get_indices:
            return features[random_sample], idxs[random_sample]
        else:
            return features[random_sample]

    def compute_features(self, dataset, size=None):
        if not size:
            size = len(dataset)

        # Add preprocessing transforms to dataset
        if hasattr(dataset, "transform") and dataset.transform:
            dataset.transform = TF.Compose(
                [dataset.transform, self.preprocess_batch, TF.ToTensor()]
            )
        else:
            # Ensures None transform isn't added
            dataset.transform = TF.Compose([self.preprocess_batch, TF.ToTensor()])

        features = torch.zeros(size, self.features_size)
        indices = np.random.choice(len(dataset), size=size, replace=False)

        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(
            subset, batch_size=256, drop_last=False, num_workers=4
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

        return features, indices

    def get_path(self, dataset):
        path = f"data/activations/{self.name}/"
        file_path = os.path.join(path, f"{dataset.name}.pkl")
        return file_path

    def save_features(self, features, indices, file_path):
        with open(file_path, "wb") as f:
            pickle.dump((features, indices), f)
            return features

    def load_features(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
