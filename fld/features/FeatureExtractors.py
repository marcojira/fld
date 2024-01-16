from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor

ALL_FEATURE_EXTRACTORS = [
    InceptionFeatureExtractor(),
    CLIPFeatureExtractor(),
    DINOv2FeatureExtractor(),
]

MAIN_FEATURE_EXTRACTORS = [
    InceptionFeatureExtractor(),
    DINOv2FeatureExtractor(),
]
