import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from fls.utils import shuffle
from fls.metrics.Metric import Metric
from fls.MoG import preprocess_feat, MoG


GEN_SIZE = 10000

curr_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(curr_path, "baseline_nlls.json"), "rb") as f:
    baseline_nlls = json.load(f)


class FLS(Metric):
    def __init__(self, eval_feat="test", baseline_nll=None):
        super().__init__()

        # One of ("train", "test")
        # Corresponds to the set whose likelihood is evaluated with the MoG
        self.eval_feat = eval_feat
        self.name = f"FLS {eval_feat.title()}"

        # Corresponds to the likelihood of the test set under a MoG centered at half of the train set fit to the other half of the train set
        self.baseline_nll = baseline_nll

    def get_nll_diff(self, nll):
        return (nll - self.baseline_nll) * 100

    def compute_metric(self, train_feat, test_feat, gen_feat):
        if not self.baseline_nll:
            self.get_baseline_nll(train_feat, test_feat)

        if len(gen_feat) > GEN_SIZE:
            gen_feat = shuffle(gen_feat, min(len(gen_feat), GEN_SIZE))

        """Preprocess"""
        train_feat, test_feat, gen_feat = preprocess_feat(
            train_feat, test_feat, gen_feat
        )

        # Fit MoGs
        mog_gen = MoG(gen_feat)
        mog_gen.fit(train_feat)

        # Default eval_feat, fits a MoG centered at generated samples to train samples, then gets LL of test samples
        if self.eval_feat == "test":
            nlls = mog_gen.get_dim_adjusted_nlls(test_feat)
        # As above but evalutes the LL of train samples
        elif self.eval_feat == "train":
            nlls = mog_gen.get_dim_adjusted_nlls(train_feat)
        else:
            raise Exception(f"Invalid mode for FLS metric: {self.eval_feat}")

        nll = nlls.mean().item()
        return self.get_nll_diff(nll)

    def get_baseline_nll(self, train_feat, test_feat):
        """Preprocess"""
        train_feat, test_feat, _ = preprocess_feat(train_feat, test_feat, train_feat)

        train_feat = shuffle(train_feat)

        # Fit MoGs
        split_size = min(GEN_SIZE, len(train_feat) // 2)
        mog = MoG(train_feat[:split_size])
        mog.fit(train_feat[split_size:])

        self.baseline_nll = mog.get_dim_adjusted_nlls(test_feat).mean().item()
        return self.baseline_nll
