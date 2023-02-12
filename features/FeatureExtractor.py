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
    def __init__(self, recompute=False):
        self.recompute=recompute
        self.features_size = None  # TO BE IMPLEMENTED BY EACH MODULE

    def preprocess_batch(self, img_batch):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_feature_batch(self, img_batch):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_features(self, data_source, size=10000, save_path=None, get_indices=False):
        """Gets shuffled features from data_source (either generator function or torch.utils.data.Dataset) of size `size` or all if size is None"""
        # Get features
        if save_path and os.path.exists(save_path) and not self.recompute:
            features = self.load_features(save_path)
        else:
            if callable(data_source):
                features = self.compute_gen_features(data_source, size)
            else:
                features = self.compute_dataset_features(data_source)

            if save_path:
                self.save_features(features, save_path)

        # Return shuffled
        size = min(size, len(features)) if size else len(features)
        random_sample = np.random.choice(len(features), size=size, replace=False)

        if get_indices:
            return features[random_sample], random_sample

        return features[random_sample]

    def compute_gen_features(self, generator, size=10000):
        """Return tensor of feature values for size samples from generator"""
        features = torch.zeros(size, self.features_size)

        start_idx = 0
        while True:
            gen_samples = generator()

            processed_imgs = []
            for img in gen_samples:
                img = TF.ToPILImage()(img)
                img = self.preprocess_batch(img)
                processed_imgs.append(img)

            gen_samples = torch.stack(processed_imgs)
            feature = self.get_feature_batch(gen_samples.cuda())

            # If going to overflow, just get required amount and break
            if size and start_idx + feature.shape[0] > size:
                features[start_idx:] = feature[: size - start_idx]
                break

            features[start_idx : start_idx + feature.shape[0]] = feature
            start_idx = start_idx + feature.shape[0]

        return features

    def compute_dataset_features(self, dataset):
        """Return tensor of feature values for data points in dataset"""
        size = len(dataset)
        features = torch.zeros(size, self.features_size)
        
        # For dataset, use dataloader
        dataset = TransformedDataset(dataset, self.preprocess_batch)

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

    def load_features(self, file_path):
        with open(file_path, "rb") as f:
            features = pickle.load(f)

        if type(features) is tuple:
            return features[0]

        return features
