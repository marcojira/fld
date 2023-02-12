import argparse
import torch
import os

from fls.metrics.AuthPct import AuthPct
from fls.metrics.FLS import FLS
from fls.metrics.CTScore import CTScore
from fls.metrics.FID import FID
from fls.datasets.SamplesDataset import SamplesDataset
from fls.features.InceptionFeatureExtractor import InceptionFeatureExtractor


def get_metrics(metrics):
    all_metrics = [FLS(), AuthPct(), CTScore(), FID(), FLS("overfit")]
    all_metrics = {metric.name: metric for metric in all_metrics}

    for metric in metrics:
        if metric not in all_metrics:
            raise ValueError(
                f"Metric {metric} not supported. Supported metrics are: {all_metrics.keys()}"
            )

    metrics = [all_metrics[metric] for metric in metrics]
    return metrics


def get_features(arg, save_arg, feature_extractor):
    # Get cached features if exists
    if save_arg and os.path.exists(save_arg):
        print(f"Loading features from {save_arg}")
        return torch.load(save_arg)

    # Check if function
    if callable(arg):
        features = feature_extractor.get_features(arg, size=10000, save_path=save_arg)
    # Check if appropriate path
    elif type(arg) is str:
        if arg.endswith(".pkl"):
            print(f"Loading features from {arg}")
            features = torch.load(arg)
        elif os.path.isdir(arg):
            dataset = SamplesDataset(arg, path=arg)
            features = feature_extractor.get_features(dataset, save_path=save_arg)
        else:
            raise ValueError(f"Argument {arg} is not a valid path")
    else:
        raise ValueError(f"Argument {arg} is not a valid path or function.")

    # Save features if save_arg is provided
    if save_arg:
        print(f"Saving features to {save_arg}")
        torch.save(features, save_arg)

    return features


def compute_metrics(
    train,
    test,
    gen,
    metrics=["FLS"],
    feature_extractor=InceptionFeatureExtractor(),
    train_save=None,
    test_save=None,
    gen_save=None,
):
    metrics = get_metrics(metrics)

    # Randomly split train into train and baseline
    train_feat = get_features(train, train_save, feature_extractor)
    perm = torch.randperm(len(train_feat))
    mid_split = len(train_feat) // 2

    train_feat, baseline_feat = (
        train_feat[perm[:mid_split]],
        train_feat[perm[mid_split:]],
    )

    test_feat = get_features(test, test_save, feature_extractor)
    gen_feat = get_features(gen, gen_save, feature_extractor)

    metric_vals = {}
    for metric in metrics:
        metric_vals[metric.name] = metric.compute_metric(
            train_feat, baseline_feat, test_feat, gen_feat
        )
    return metric_vals
