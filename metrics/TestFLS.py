import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fls.metrics.Metric import Metric


def plot_sigmas(sigmas):
    sns.kdeplot(sigmas.cpu().numpy())
    plt.show()


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
        plot_sigmas(self.log_sigmas)
        return self.log_sigmas, losses

    def evaluate(self, x):
        """Evaluate LL of x under MoG"""
        dists = compute_dists(x, self.mus)
        # self.log_sigmas[1:] = torch.full(
        #     (self.n_gaussians,), -20.0, requires_grad=True, device="cuda"
        # )
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
