from fls.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fls.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from fls.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor

ALL_FEATURE_EXTRACTORS = [
    InceptionFeatureExtractor(),
    CLIPFeatureExtractor(),
    DINOv2FeatureExtractor(),
]

MAIN_FEATURE_EXTRACTORS = [
    InceptionFeatureExtractor(),
    DINOv2FeatureExtractor(),
]
