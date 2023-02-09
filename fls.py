import argparse
import torch
import os

from metrics.AuthPct import AuthPct
from metrics.FLS import FLS
from metrics.CTScore import CTScore
from metrics.FID import FID
from features.InceptionFeatureExtractor import InceptionFeatureExtractor


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
        return
    # Check if appropriate path
    elif type(arg) is str:
        if arg.endswith(".pt"):
            features = torch.load(arg)
        elif os.path.isdir(arg):
            return
        else:
            raise ValueError(f"Argument {arg} is not a valid path")
    else:
        raise ValueError(f"Argument {arg} is not a valid path or function.")

    # Save features if save_arg is provided
    if save_arg:
        print(f"Saving features to {save_arg}")
        torch.save(features, save_arg)

    return features


def compute_score(
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

    train_feat = get_features(train, train_save, feature_extractor)
    test_feat = get_features(test, test_save, feature_extractor)
    gen_feat = get_features(gen, gen_save, feature_extractor)

    metric_vals = {}
    for metric in metrics:
        metric_vals[metric.name] = metric.compute_metric(
            train_feat, test_feat, gen_feat
        )
    return metric_vals


if __name__ == "__main__":
    """Construct argument parser"""
    parser = argparse.ArgumentParser(description="Compute FLS and other metrics.")

    for dataset in ["train", "test", "gen"]:
        parser.add_argument(
            dataset,
            type=str,
            help=f"Path to folder containing {dataset} data or path to .pt file with pre-computed features",
            required=True,
        )
        parser.add_argument(
            f"{dataset}_save",
            type=str,
            help=f"Destination path to save features obtained from {dataset} data (by default, features are not saved)",
            default=None,
        )

    parser.add_argument(
        "metrics",
        type=str,
        nargs="+",
        help="Path to folder containing test data or path to .pt file with pre-computed features",
    )

    args = parser.parse_args()
