import argparse

from fls.utils import compute_metrics

if __name__ == "__main__":
    """Construct argument parser"""
    parser = argparse.ArgumentParser(description="Compute FLS and other metrics.")

    for dataset in ["train", "test", "gen"]:
        parser.add_argument(
            f"--{dataset}",
            type=str,
            help=f"Path to folder containing {dataset} data or path to .pkl file with pre-computed features",
            required=True,
        )
        parser.add_argument(
            f"--{dataset}_save",
            type=str,
            help=f"Destination path to save features obtained from {dataset} data (by default, features are not saved)",
            default=None,
        )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        help="Path to folder containing test data or path to .pkl file with pre-computed features",
        default=["FLS"],
    )

    args = parser.parse_args()

    metrics = compute_metrics(
        args.train,
        args.test,
        args.gen,
        metrics=args.metrics,
        train_save=args.train_save,
        test_save=args.test_save,
        gen_save=args.gen_save,
    )

    for metric, val in metrics.items():
        print(f"{metric}: {val:.3f}")
