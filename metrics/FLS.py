import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fls.metrics.Metric import Metric


def preprocess_fls(train_feat, test_feat, gen_feat, normalize=True):
    # Assert correct device
    train_feat = train_feat.cuda()
    test_feat = test_feat.cuda()
    gen_feat = gen_feat.cuda()

    # Normalize features to 0 mean, unit variance
    mean_vals = test_feat.mean(dim=0)
    std_vals = test_feat.std(dim=0)

    def normalize(feat):
        feat = (feat - mean_vals) / (std_vals)
        return feat

    if normalize:
        train_feat = normalize(train_feat)
        test_feat = normalize(test_feat)
        gen_feat = normalize(gen_feat)

    return train_feat, test_feat, gen_feat


def compute_dists(x_data, x_kernel):
    """Returns the dists tensor of all L2^2 distances between samples from x_data and x_kernel"""
    # More accurate but slower
    # return (
    #     torch.cdist(x_data, x_kernel, compute_mode="donot_use_mm_for_euclid_dist")
    # ) ** 2
    dists = torch.cdist(x_data, x_kernel) ** 2
    return dists.detach()


class MoG:
    def __init__(self, mus, log_sigmas=None, lr=0.5, num_steps=75):
        reg_point = torch.zeros((1, mus.shape[1])).cuda()

        self.mus = torch.cat([reg_point, mus], dim=0)
        self.n_gaussians = mus.shape[0]
        self.dim = mus.shape[1]

        if log_sigmas is None:
            self.log_sigmas = torch.zeros(
                self.n_gaussians + 1, requires_grad=True, device="cuda"
            )
        else:
            self.log_sigmas = log_sigmas

        # Optimization hyperparameters
        self.lr = lr
        self.num_steps = num_steps

    def ll(self, dists, fit=False, lambd=0.5):
        """Computes the MoG LL using the matrix of distances"""
        exponent_term = (-0.5 * dists) / (torch.exp(self.log_sigmas))

        # Here we use that dividing by x is equivalent to multiplying by e^{-ln(x)}
        # allows for use of logsumexp
        exponent_term -= (self.dim / 2) * self.log_sigmas
        exponent_term -= (self.dim / 2) * np.log(2 * np.pi)
        exponent_term -= np.log(self.n_gaussians)

        if fit:
            exponent_term[:, 1:] += np.log(1 - lambd)
            exponent_term[:, 0] += np.log(self.n_gaussians)
            exponent_term[:, 0] += np.log(lambd)
            # exponent_term[:, 0] = -2500
            # print(exponent_term[:, 0])
            # print(self.log_sigmas[0])
            inner_term = torch.logsumexp(exponent_term, dim=1)
        else:
            inner_term = torch.logsumexp(exponent_term[:, 1:], dim=1)
            print(inner_term.shape)
            sns.kdeplot(inner_term.detach().cpu().numpy())
            plt.show()
            sns.kdeplot(torch.logsumexp(exponent_term, dim=1).detach().cpu().numpy())
            plt.show()

        return inner_term

    def get_pairwise_ll(self, x):
        dists = compute_dists(x, self.mus)
        exponent_term = (-0.5 * dists) / torch.exp(self.log_sigmas)
        exponent_term -= (self.dim / 2) * self.log_sigmas
        exponent_term -= (self.dim / 2) * np.log(2 * np.pi)
        return exponent_term

    def fit(self, x):
        """Fit log_sigmas to minimize NLL of x under MoG"""
        # Tracking
        losses = []

        optim = torch.optim.Adam([self.log_sigmas], lr=self.lr)
        dists = compute_dists(x, self.mus)

        for step in range(self.num_steps):
            optim.zero_grad()
            loss = -(self.ll(dists, fit=True)).mean()

            loss.backward()
            optim.step()

            # Here we clamp log_sigmas to stop values exploding for identical samples
            with torch.no_grad():
                self.log_sigmas.data = self.log_sigmas.clamp(-30, 20).data

            losses.append(loss.item())

        self.log_sigmas = self.log_sigmas.detach()
        sns.kdeplot(self.log_sigmas.cpu().numpy())
        plt.show()
        return self.log_sigmas, losses

    def evaluate(self, x):
        """Evaluate LL of x under MoG"""
        dists = compute_dists(x, self.mus)
        self.log_sigmas[1:] = torch.full(
            (self.n_gaussians,), -20.0, requires_grad=True, device="cuda"
        )
        return self.ll(dists)


class FLS(Metric):
    def __init__(self, mode="", baseline_LL=0):
        super().__init__()
        self.mode = mode  # One of ("", "train", "% overfit samples")
        self.name = f"{mode} FLS"
        self.baseline_LL = (
            baseline_LL  # Obtained by getting the FLS of a subset of the train set
        )

    def get_adjusted_nll(self, ll, dim):
        dim_adjusted_nll = -ll.cpu().item() / dim
        return (dim_adjusted_nll - self.baseline_LL) * 100

    def compute_metric(self, train_feat, test_feat, gen_feat):
        """Preprocess"""
        train_feat, test_feat, gen_feat = preprocess_fls(
            train_feat, test_feat, gen_feat
        )

        dim = train_feat.shape[1]

        # Default mode, fits a MoG centered at generated samples to train samples, then gets LL of test samples
        if self.mode == "":
            # Fit MoGs
            mog_gen = MoG(gen_feat)
            mog_gen.fit(train_feat)
            gen_ll = mog_gen.evaluate(test_feat).mean()

            return self.get_adjusted_nll(gen_ll, dim)

        # As above but evalutes the LL of train samples
        elif self.mode == "train":
            # Fit MoGs
            mog_gen = MoG(gen_feat)
            mog_gen.fit(train_feat)
            gen_ll = mog_gen.evaluate(train_feat).mean()

            return self.get_adjusted_nll(gen_ll, dim)

        # Difference between the above two
        elif self.mode == "train_vs_test":
            mog_gen = MoG(gen_feat)
            mog_gen.fit(train_feat)

            train_ll = mog_gen.evaluate(train_feat).mean()
            test_ll = mog_gen.evaluate(test_feat).mean()

            return self.get_adjusted_nll(train_ll, dim) - self.get_adjusted_nll(
                test_ll, dim
            )

        # Percent of samples closer to nearest test than nearest train
        elif self.mode == "AuthPctTest":
            train_dists = compute_dists(gen_feat, train_feat)
            min_train_dists = train_dists.min(dim=1).values

            test_dists = compute_dists(gen_feat, test_feat)
            min_test_dists = test_dists.min(dim=1).values

            pct_closer = (
                100
                * (min_test_dists < min_train_dists).sum().item()
                / gen_feat.shape[0]
            )
            return pct_closer - 50

        # Percent of samples closer to nearest test than nearest train (adjusted for dataset size)
        elif self.mode == "AuthPctAdjusted":
            train_dists = compute_dists(gen_feat, train_feat)
            min_train_dists = train_dists.min(dim=1).values
            min_train_dists = min_train_dists * np.sqrt(2 * np.log(train_feat.shape[0]))

            test_dists = compute_dists(gen_feat, test_feat)
            min_test_dists = test_dists.min(dim=1).values
            min_test_dists = min_test_dists * np.sqrt(2 * np.log(test_feat.shape[0]))

            pct_closer = (
                100
                * (min_test_dists < min_train_dists).sum().item()
                / gen_feat.shape[0]
            )
            return pct_closer - 50

        # Percent of samples that have lower sigma when fit to train than test
        elif self.mode == "Pct lower sigma":
            mog_gen = MoG(gen_feat)
            train_log_sigmas, _ = mog_gen.fit(train_feat)

            mog_gen = MoG(gen_feat)
            test_log_sigmas, _ = mog_gen.fit(test_feat)

            pct_smaller = (
                100
                * (train_log_sigmas > test_log_sigmas).sum().item()
                / gen_feat.shape[0]
            )
            return pct_smaller - 50

        elif self.mode == "POG NN":
            mog_gen = MoG(gen_feat)
            mog_gen.fit(train_feat)

            train_lls = mog_gen.get_pairwise_ll(train_feat)
            nn_train_lls = train_lls.min(dim=0).values

            test_lls = mog_gen.get_pairwise_ll(test_feat)
            nn_test_lls = train_lls.min(dim=0).values

            pct_smaller = (
                100 * (nn_train_lls < nn_test_lls).sum().item() / gen_feat.shape[0]
            )
            return pct_smaller - 50

        # Percentage of Gaussians that assign higher LL to test than train (-50)
        elif self.mode == "POG":
            mog_gen = MoG(gen_feat)
            mog_gen.fit(train_feat)
            train_lls = mog_gen.get_pairwise_ll(train_feat)
            test_lls = mog_gen.get_pairwise_ll(test_feat)

            ll_diff = train_lls.mean(axis=0) - test_lls.mean(axis=0)
            score = ((ll_diff < 0).sum().item() / ll_diff.shape[0]) * 100
            return score - 50

        else:
            raise Exception(f"Invalid mode for FLS metric: {self.mode}")

    def get_nns(self, gen_feat, train_feat, idxs, k=2):
        """
        Get indices of k nearest neighbors in feature space for generated samples at `idxs`
        """
        dists = compute_dists(gen_feat[idxs], train_feat)
        return dists.topk(k, largest=False).indices

    def get_overfit_samples(self, train_feat, test_feat, gen_feat, percentiles, mode):
        """
        After sorting generated samples by O_n, return idxs of those at `percentiles`
        e.g. percentiles = [99] will return the idx of the sample with higher O_n than 99% of other generated samples and be deemed highly overfit
        """
        overfit_idxs = torch.zeros(len(percentiles), dtype=torch.int64)

        train_feat, test_feat, gen_feat = preprocess_fls(
            train_feat, test_feat, gen_feat
        )

        def ll_diff():
            # Fit MoG
            mog_gen = MoG(gen_feat)
            log_sigmas, _ = mog_gen.fit(train_feat)

            # Get highest LL differences
            train_lls = mog_gen.get_pairwise_ll(train_feat)
            test_lls = mog_gen.get_pairwise_ll(test_feat)

            ll_diff = train_lls.mean(axis=0) - test_lls.mean(axis=0)
            ll_diff, idxs = ll_diff.sort(descending=False)
            return idxs

        def nn_dist_diff():
            train_dists = torch.cdist(gen_feat, train_feat) ** 2
            min_train_idxs, min_train_dists = (
                train_dists.min(dim=1).indices,
                train_dists.min(dim=1).values,
            )
            test_dists = torch.cdist(gen_feat, test_feat) ** 2
            min_test_dists = test_dists.min(dim=1).values

            diff = min_train_dists / min_test_dists
            sorted_diff = diff.sort(descending=True)
            sns.kdeplot(diff.cpu().numpy())
            return sorted_diff.indices, sorted_diff.values

        def nn_dist():
            train_dists = torch.cdist(gen_feat, train_feat) ** 2
            min_train_idxs, min_train_dists = (
                train_dists.min(dim=1).indices,
                train_dists.min(dim=1).values,
            )

            sorted_dists = min_train_dists.sort(descending=True)
            sns.kdeplot(min_train_dists.cpu().numpy())
            return sorted_dists.indices, sorted_dists.values

        def nn_dist_diff_adjusted():
            train_dists = torch.cdist(gen_feat, train_feat) ** 2
            min_train_idxs, min_train_dists = (
                train_dists.min(dim=1).indices,
                train_dists.min(dim=1).values,
            )
            min_train_dists = min_train_dists * np.sqrt(2 * np.log(train_feat.shape[0]))

            test_dists = torch.cdist(gen_feat, test_feat) ** 2
            min_test_dists = test_dists.min(dim=1).values
            min_test_dists = min_test_dists * np.sqrt(2 * np.log(test_feat.shape[0]))

            diff = min_train_dists - min_test_dists
            sorted_diff = diff.sort(descending=True)
            sns.kdeplot(diff.cpu().numpy())
            return sorted_diff.indices, sorted_diff.values

        if mode == "nn_dist_diff":
            idxs, values = nn_dist_diff()
        elif mode == "nn_dist":
            idxs, values = nn_dist()
        elif mode == "nn_dist_diff_adjusted":
            idxs, values = nn_dist_diff_adjusted()

        elif mode == "min_sigma":
            # Fit MoG
            mog_gen = MoG(gen_feat)
            log_sigmas, _ = mog_gen.fit(train_feat)

            idxs, values = (
                log_sigmas.sort(descending=True).indices,
                log_sigmas.sort(descending=True).values,
            )

        elif mode == "min_sigma_train-min_sigma_test":
            # Fit MoG
            mog_gen = MoG(gen_feat)
            log_sigmas_train, _ = mog_gen.fit(train_feat)

            mog_gen = MoG(gen_feat)
            log_sigmas_test, _ = mog_gen.fit(test_feat)

            diff = log_sigmas_train.exp() / log_sigmas_test.exp()
            idxs, values = (
                diff.sort(descending=True).indices,
                diff.sort(descending=True).values,
            )

        elif mode == "highest_ll":
            # Fit MoG
            mog_gen = MoG(gen_feat)
            log_sigmas_train, _ = mog_gen.fit(train_feat)

            train_lls = mog_gen.get_pairwise_ll(train_feat).max(dim=0).values

            diff = train_lls
            idxs, values = (
                diff.sort(descending=False).indices,
                diff.sort(descending=False).values,
            )

        elif mode == "highest_ll_diff":
            # Fit MoG
            mog_gen = MoG(gen_feat)
            log_sigmas_train, _ = mog_gen.fit(train_feat)

            train_lls = mog_gen.get_pairwise_ll(train_feat).max(dim=0).values
            test_lls = mog_gen.get_pairwise_ll(test_feat).max(dim=0).values

            # diff = torch.cat([train_lls.unsqueeze(1), -test_lls.unsqueeze(1)], dim=1).logsumexp(dim=1)
            diff = train_lls - test_lls
            idxs, values = (
                diff.sort(descending=False).indices,
                diff.sort(descending=False).values,
            )

        elif mode == "precision":
            # Fit MoG
            mog_train = MoG(train_feat)
            log_sigmas_train, _ = mog_train.fit(gen_feat)

            gen_lls = mog_train.get_pairwise_ll(gen_feat).max(dim=0).values

            diff = gen_lls
            idxs, values = (
                diff.sort(descending=False).indices,
                diff.sort(descending=False).values,
            )
            for i, quantile in enumerate(percentiles):
                pos = int(len(train_feat) * quantile / 100)
                print(values[pos])
                overfit_idxs[i] = idxs[pos]

            return overfit_idxs

        sns.kdeplot(values.cpu().numpy())

        for i, quantile in enumerate(percentiles):
            pos = int(len(gen_feat) * quantile / 100)
            print(values[pos])
            overfit_idxs[i] = idxs[pos]

        return overfit_idxs

    def get_best_samples(self, train_feat, test_feat, gen_feat, best=True, k=10):
        """Get indices of k highest LL gen samples for MoG centered at test samples"""
        train_feat, test_feat, gen_feat = preprocess_fls(
            train_feat, test_feat, gen_feat
        )

        mog_test = MoG(test_feat)
        mog_test.fit(train_feat)
        lls = mog_test.evaluate(gen_feat)
        return lls.topk(k, largest=best).indices
