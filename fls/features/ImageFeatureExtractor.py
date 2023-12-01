from __future__ import annotations

import torch
import math
import torchvision

from tqdm import tqdm

from fls.features.FeatureExtractor import FeatureExtractor
from fls.datasets.ImageFilesDataset import ImageFilesDataset
from fls.datasets.ImageTensorDataset import ImageTensorDataset

BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageFeatureExtractor(FeatureExtractor):
    def get_tensor_features(self, img_tensor: torch.Tensor, name=None):
        dataset = ImageTensorDataset(img_tensor)
        return self.get_features(dataset, name)

    def get_dir_features(self, dir: str, extension="png", name=None):
        dataset = ImageFilesDataset(dir, extension=extension)
        return self.get_features(dataset, name)

    def get_model_features(self, gen_fn: function, num_samples: int):
        img_tensor = gen_fn()
        batch_size = img_tensor.shape[0]

        num_batches = math.ceil(num_samples / batch_size)
        features = torch.zeros(num_batches * batch_size, self.features_size)

        for i in tqdm(range(num_batches)):
            img_tensor = gen_fn()
            curr_features = self.get_tensor_features(img_tensor)
            features[i * batch_size : (i + 1) * batch_size] = curr_features

        return features[:num_samples]
