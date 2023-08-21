import torch
import torchvision.transforms as transforms
import clip
from fls.features.FeatureExtractor import FeatureExtractor


class CLIPFeatureExtractor(FeatureExtractor):
    def __init__(self, recompute=False, save_path=None):
        self.name = "clip"

        super().__init__(recompute=recompute, save_path=save_path)

        self.features_size = 512
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        self.model.eval()

    def preprocess_batch(self, img_batch):
        return self.preprocess(img_batch)

    def get_feature_batch(self, img_batch):
        with torch.no_grad():
            features = self.model.encode_image(img_batch)
        return features
