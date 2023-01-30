from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import clip
from fls.features.FeatureExtractor import FeatureExtractor


class CLIPFeatureExtractor(FeatureExtractor):
    def __init__(self, recompute=False, save=True):
        self.name = "clip"
        self.features_size = 512

        super().__init__(recompute=recompute, save=save)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
            ]
        )
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        self.model.eval()

    def preprocess_batch(self, img_batch):
        img_batch = self.preprocess(img_batch)
        return img_batch

    def get_feature_batch(self, img_batch):
        T_norm = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        img_batch = T_norm(img_batch)

        with torch.no_grad():
            features = self.model.encode_image(img_batch)
        return features
