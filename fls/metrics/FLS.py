import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fls.metrics.Metric import Metric
from fls.MoG import preprocess_feat, MoG


class FLS(Metric):
    def __init__(self, eval_feat="test", baseline_LL=0):
        super().__init__()

        # One of ("train", "test")
        # Corresponds to the set whose likelihood is evaluated with the MoG
        self.eval_feat = eval_feat
        self.name = f"{eval_feat.title()} FLS"

        # Corresponds to the likelihood of the test set under a MoG centered at half of the train set fit to the other half of the train set
        self.baseline_LL = baseline_LL

    def compute_metric(self, train_feat, test_feat, gen_feat):
        """Preprocess"""
        train_feat, test_feat, gen_feat = preprocess_feat(
            train_feat, test_feat, gen_feat
        )

        # Fit MoGs
        mog_gen = MoG(gen_feat)
        mog_gen.fit(train_feat)

        # Default eval_feat, fits a MoG centered at generated samples to train samples, then gets LL of test samples
        if self.eval_feat == "test":
            nll = mog_gen.get_dim_adjusted_nlls(test_feat).mean()
        # As above but evalutes the LL of train samples
        elif self.eval_feat == "train":
            nll = mog_gen.get_dim_adjusted_nlls(train_feat).mean()
        else:
            raise Exception(f"Invalid mode for FLS metric: {self.eval_feat}")

        return nll.mean().item()
