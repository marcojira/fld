import pickle
import torch
import os
import math
from tqdm import tqdm
import numpy as np
import yaml

import torchvision
from PIL import Image

CHUNK_SIZE = 100000


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
    """Wrapper class for dataset to add transform when there is none"""

    def __init__(self, dataset, transform, to_pil=False):
        self.dataset = dataset
        self.transform = transform

        self.to_pil = to_pil
        self.pil_transform = torchvision.transforms.ToPILImage()

    def __getitem__(self, idx):
        img, labels = self.dataset[idx]
        if self.to_pil:
            img = self.pil_transform(img)
        img = self.transform(img)
        return img, labels

    def __len__(self):
        return len(self.dataset)


class FeatureExtractor:
    def __init__(self, recompute=False, save_path="config"):
        self.recompute = recompute

        if save_path == "config":
            config_file = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
            with open(config_file, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.save_path = os.path.join(config["feature_save_path"], self.name)
        elif save_path is not None:
            self.save_path = os.path.join(save_path, self.name)
        else:
            self.save_path = False

        os.makedirs(self.save_path, exist_ok=True)
        self.features_size = None  # TO BE IMPLEMENTED BY EACH MODULE

    def preprocess_batch(self, img_batch):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_feature_batch(self, img_batch):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass

    def get_curr_features (self, img_array, idx_beg, idx_end, batchsize):
        img_sub_array = to_img(img_array[idx_beg:idx_end])
        img_batch = torch.stack(
            [
                self.preprocess_batch(Image.fromarray(img_sub_array[i]))
                for i in range(len(img_sub_array))
            ]
        )
        curr_features = self.get_feature_batch(img_batch.cuda())
        return curr_features

    def get_gen_features_from_tensor(self, img_array, batchsize):

        n_gen_samples = img_array.shape[0]
        num_batches = n_gen_samples // batchsize
        features = torch.zeros(n_gen_samples, self.features_size)

        for i in tqdm(range(num_batches), desc='Features extraction loop'):
            idx_beg = batchsize * i
            idx_end = batchsize * (i+1)
            features[idx_beg:idx_end] = self.get_curr_features(
                img_array, idx_beg, idx_end, batchsize)

        idx_beg = batchsize * num_batches
        idx_end = -1
        features[idx_beg:idx_end] = self.get_curr_features(
            img_array, idx_beg, idx_end, batchsize)

        return features

    def get_gen_features(self, f, size=10000):
        try:
            img_array = f()
            batch_size = img_array.shape[0]
        except:
            all_img_array = f
            batch_size = all_img_array.shape[0]
            # all_img_array = to_img(f)

        num_batches = math.ceil(size / batch_size)

        features = torch.zeros(num_batches * batch_size, self.features_size)

        for i in tqdm(range(num_batches)):
            try:
                img_array = to_img(f())
            except:
                img_array = to_img(all_img_array[i * batch_size:(i+1)*batch_size, :])
                # import ipdb; ipdb.set_trace()
            img_batch = torch.stack(
                [
                    self.preprocess_batch(Image.fromarray(img_array[i]))
                    for i in range(batch_size)
                ]
            )
            curr_features = self.get_feature_batch(img_batch.cuda())

            features[i * batch_size : (i + 1) * batch_size] = curr_features

        return features[:size]

    def get_all_features(self, dataset):
        """Returns combined features for all chunks"""
        features = []
        for _, feature in self.get_features(dataset):
            features.append(feature)

        return torch.cat(features)

    def get_features(self, dataset):
        """
        Generator over features of dataset, split into chunks of CHUNK_SIZE
        Yields chunk (full dataset indices of the features), features (features of that chunk)
        """
        num_imgs = len(dataset)
        num_chunks = num_imgs // CHUNK_SIZE + 1

        for chunk_idx in range(num_chunks):
            dir_path = os.path.join(self.save_path, f"{dataset.name}")
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, f"{chunk_idx}.pkl")
            begin, end = chunk_idx * CHUNK_SIZE, min(
                (chunk_idx + 1) * CHUNK_SIZE, len(dataset)
            )
            chunk = list(range(begin, end))

            # Get features
            if not self.recompute and os.path.exists(file_path):
                # If the features are cached and not to be recomputed, simply load them
                features = self.load_features(file_path)
            else:
                # Create Subset just for the chunk and get features of that Subset
                subset = torch.utils.data.Subset(dataset, chunk)
                features = self.compute_features(subset)
                if self.save_path:
                    self.save_features(features, file_path)

            yield chunk, features

    def compute_features(self, dataset):
        """Compute features of dataset by looping over dataset"""

        # Deal with datasets that return tensors
        if torch.is_tensor(dataset[0][0]):
            dataset = TransformedDataset(dataset, self.preprocess_batch, to_pil=True)
        else:
            dataset = TransformedDataset(dataset, self.preprocess_batch, to_pil=False)

        size = len(dataset)
        features = torch.zeros(size, self.features_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=256, drop_last=False, num_workers=4, shuffle=False
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
            pickle.dump(features, f)
            return features

    def load_features(self, path):
        with open(path, "rb") as f:
            features = pickle.load(f)

        return features

    def shuffle_features(self, features, size):
        random_sample = np.random.choice(len(features), size=size, replace=False)
        return random_sample, features[random_sample]
