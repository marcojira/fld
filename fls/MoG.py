import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from fls.utils import shuffle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_feat(train_feat, test_feat, gen_feat, normalize=True):
    # Assert correct device
    train_feat = train_feat.to(DEVICE)
    test_feat = test_feat.to(DEVICE)
    gen_feat = gen_feat.to(DEVICE)

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


def compute_dists(x_data, x_gaussians):
    """Returns the dists tensor of all L2^2 distances between samples from x_data and x_gaussians"""
    # More accurate but slower
    # return (
    #     torch.cdist(x_data, x_gaussians, compute_mode="donot_use_mm_for_euclid_dist")
    # ) ** 2
    dists = torch.cdist(x_data, x_gaussians) ** 2
    return dists.detach()


class MoG:
    def __init__(self, mus, log_sigmas=None, lr=0.3, num_epochs=50):
        self.mus = mus
        self.num_gaussians = mus.shape[0]
        self.dim = mus.shape[1]

        # Log sigmas are used for a stable optimization process
        if log_sigmas is None:
            self.log_sigmas = torch.full(
                (self.num_gaussians,), -0.1, requires_grad=True, device=DEVICE
            )
        else:
            self.log_sigmas = log_sigmas

        self.origin_log_sigma = torch.tensor(0.0, device=DEVICE, requires_grad=True)

        # Optimization hyperparameters
        self.lr = lr
        self.num_epochs = num_epochs

    def dists(self, x):
        return compute_dists(x, self.mus)

    def pairwise_lls_from_dists(self, dists, log_sigmas):
        """Returns an n x m matrix (where n is the number of samples and m the number of gaussians)

        An entry a_ij of this matrix is such that a_ij = log[N(x_i|x_j, sigma_j)]
        """
        exponent_term = (-0.5 * dists) / (torch.exp(log_sigmas))

        # Here we use that dividing by x is equivalent to multiplying by e^{-ln(x)}
        # allows for use of logsumexp for numerical stability
        exponent_term -= (self.dim / 2) * log_sigmas
        exponent_term -= (self.dim / 2) * np.log(2 * np.pi)
        return exponent_term

    def lls_from_pairwise(self, pairwise_lls, origin_lls=None, lambd=0.1):
        if origin_lls is None:
            all_lls = pairwise_lls - np.log(self.num_gaussians)
        else:
            pairwise_lls -= 1 / lambd * np.log(self.num_gaussians)
            origin_lls -= 1 / (1 - lambd)
            all_lls = torch.cat([pairwise_lls, origin_lls], dim=1)

        lls = torch.logsumexp(all_lls, dim=1)
        return lls

    def lls(self, x):
        dists = self.dists(x)
        pairwise_lls = self.pairwise_lls_from_dists(dists, self.log_sigmas)
        lls = self.lls_from_pairwise(pairwise_lls)

        return lls

    def fit(self, x, batch_size=10000):
        """Fit log_sigmas to minimize NLL of x under MoG"""
        # Tracking
        losses = []

        optim = torch.optim.Adam([self.log_sigmas, self.origin_log_sigma], lr=self.lr)

        x = shuffle(x)
        for _ in tqdm(range(self.num_epochs), leave=False):
            for batch in x.split(batch_size):
                optim.zero_grad()

                dists = self.dists(batch)
                pairwise_lls = self.pairwise_lls_from_dists(dists, self.log_sigmas)
                # print(pairwise_lls)

                # LL of points relative to origin point (ensures each point has some minimum LL)
                origin_dists = (batch * batch).sum(dim=1, keepdim=True)
                origin_lls = self.pairwise_lls_from_dists(
                    origin_dists, self.origin_log_sigma
                )

                lls = self.lls_from_pairwise(pairwise_lls, origin_lls)
                loss = -lls.mean()
                loss /= self.dim
                loss.backward()
                optim.step()

                # We clamp log_sigmas to stop NANs for identical samples
                with torch.no_grad():
                    self.log_sigmas.data = self.log_sigmas.clamp(-30, 20).data

                losses.append(loss.item())

        self.log_sigmas = self.log_sigmas.detach()

        plt.plot(losses)
        plt.show()

        sns.kdeplot(self.log_sigmas.cpu().numpy())
        plt.show()

        sns.kdeplot(lls.detach().cpu().numpy())
        plt.show()

        return self.log_sigmas, losses

    def get_dim_adjusted_nlls(self, x):
        """Get LL of x under MoG, adjusted for dimension"""
        lls = self.lls(x)
        dim_adjusted_nlls = -lls / self.dim
        return dim_adjusted_nlls
