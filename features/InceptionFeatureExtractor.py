from pytorch_fid.inception import InceptionV3

import torchvision.transforms as TF
import torch

from fls.features.FeatureExtractor import FeatureExtractor


class InceptionFeatureExtractor(FeatureExtractor):
    def __init__(self, recompute=False, save_path=False):
        self.name = "inception"

        super().__init__(recompute=recompute, save_path=save_path)

        self.features_size = 2048
        self.preprocess = TF.ToTensor()

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3(
            [block_idx], resize_input=True, normalize_input=True
        ).cuda()
        self.model.eval()
        return

    def preprocess_batch(self, img_batch):
        return self.preprocess(img_batch)

    def get_feature_batch(self, img_batch):
        assert img_batch.max() <= 1
        assert img_batch.min() >= 0

        with torch.no_grad():
            features = self.model(img_batch)[0].squeeze()

        return features
