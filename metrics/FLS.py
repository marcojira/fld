import torch
import numpy as np

from fls.metrics.Metric import Metric

def preprocess_fls(
    train_feat, test_feat, gen_feat, normalize=True
):
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
        self.mus = mus
        self.n_gaussians = mus.shape[0]
        self.dim = mus.shape[1]

        if log_sigmas is None:
            self.log_sigmas = torch.zeros(
                self.n_gaussians, requires_grad=True, device="cuda"
            )
        else:
            self.log_sigmas = log_sigmas

        # Optimization hyperparameters
        self.lr = lr
        self.num_steps = num_steps

    def ll(self, dists):
        """Computes the MoG LL using the matrix of distances"""
        exponent_term = (-0.5 * dists) / (torch.exp(self.log_sigmas))

        # Here we use that dividing by x is equivalent to multiplying by e^{-ln(x)}
        # allows for use of logsumexp
        exponent_term -= (self.dim / 2) * self.log_sigmas
        exponent_term -= (self.dim / 2) * np.log(2 * np.pi)
        exponent_term -= np.log(self.n_gaussians)

        inner_term = torch.logsumexp(exponent_term, dim=1)
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
            loss = -self.ll(dists).mean()
            loss.backward()
            optim.step()

            # Here we clamp log_sigmas to stop values exploding for identical samples
            with torch.no_grad():
                self.log_sigmas.data = self.log_sigmas.clamp(-30, 20).data

            losses.append(loss.item())

        self.log_sigmas = self.log_sigmas.detach()
        return self.log_sigmas, losses

    def evaluate(self, x):
        """Evaluate LL of x under MoG"""
        dists = compute_dists(x, self.mus)
        return self.ll(dists)


class FLS(Metric):
    def __init__(self, mode="", c=0):
        super().__init__()
        self.mode = mode # One of ["", "train", "% overfit samples", ]
        self.name = f"{mode} FLS"
        self.c = c

    def compute_metric(
        self, train_feat, baseline_feat, gen_feat
    ):
        """Preprocess"""
        train_feat, baseline_feat, test_feat, gen_feat = preprocess_fls(
            train_feat, baseline_feat, test_feat, gen_feat, self.pca_dim
        )

        dim = train_feat.shape[1]

        # 
        if self.mode == "":
            # Fit MoGs
            mog_gen = MoG(gen_feat)
            mog_gen.fit(train_feat)
            gen_ll = mog_gen.evaluate(test_feat).mean()

            dim_adjusted_nll = -gen_ll.cpu().item() / dim
            return (dim_adjusted_nll - self.c) * 100

        elif self.mode == "train":
            # Fit MoGs
            mog_gen = MoG(gen_feat)
            mog_gen.fit(train_feat)
            gen_ll = mog_gen.evaluate(train_feat).mean()

            dim_adjusted_nll = -gen_ll.cpu().item() / dim
            return (dim_adjusted_nll - self.c) * 100

        elif self.mode == "%_overfit_gaussians(-50%)":
            # Fit MoGs
            mog_gen = MoG(gen_feat)
            mog_gen.fit(torch.cat([test_feat, train_feat], dim=0))

            test_ll = mog_gen.evaluate(test_feat).mean()/dim
            train_ll = mog_gen.evaluate(train_feat).mean()/dim

            return (test_ll - train_ll).item() * 100

            # mog_gen = MoG(gen_feat)
            # log_sigmas_train, _ = mog_gen.fit(train_feat)
            # mog_gen = MoG(gen_feat)
            # log_sigmas_test, _ = mog_gen.fit(test_feat)

            # df = pd.DataFrame(
            #     {
            #         "train": log_sigmas_train.flatten().cpu().numpy(),
            #         "test": log_sigmas_test.flatten().cpu().numpy(),
            #     }
            # )
            # sns.kdeplot(data=df, fill=True, alpha=0.5)
            # plt.show()

            # # return log_sigmas_train.mean() - log_sigmas_test.mean()
            # return (log_sigmas_train < log_sigmas_test).sum()/gen_feat.shape[0]
            return (
                (log_sigmas_train < log_sigmas_test).sum().item()
                * 100
                / gen_feat.shape[0]
            )

            train_lls = mog_gen.get_pairwise_ll(train_feat)
            test_lls = mog_gen.get_pairwise_ll(test_feat)

            ll_diff = (
                train_lls.sum(axis=0) / train_feat.shape[0]
                - test_lls.sum(axis=0) / test_feat.shape[0]
            )
            score = ((ll_diff > 0).sum().item() / ll_diff.shape[0]) * 100
            return score - 50
        else:
            raise Exception("Invalid mode for FLS metric")


    def get_nns(self, gen_feat, train_feat, idxs, k=2):
        dists = compute_dists(gen_feat[idxs], train_feat)
        return dists.topk(k, largest=False).indices

    def get_overfit_samples(self, train_feat, test_feat, gen_feat, quantiles, k=2):
        overfit_idxs = torch.zeros(len(quantiles), dtype=torch.int64)

        train_feat, baseline_feat, test_feat, gen_feat = preprocess_fls(
            train_feat, test_feat, test_feat, gen_feat, self.pca_dim
        )

        mog_gen = MoG(gen_feat)
        log_sigmas, _ = mog_gen.fit(train_feat)

        # sigmas, sigma_idxs = mog_gen.log_sigmas.sort(descending=False)
        # print(sigmas)

        # curr_delta = 0
        # for i in range(len(sigmas)):
        #     if sigmas[i] > deltas[curr_delta]:
        #         print(f"Found {sigmas[i]} at {i}")
        #         overfit_idxs[curr_delta] = sigma_idxs[i]
        #         curr_delta += 1

        #     if curr_delta == len(deltas):
        #         break

        train_lls = mog_gen.get_pairwise_ll(train_feat)
        test_lls = mog_gen.get_pairwise_ll(test_feat)

        ll_diff = log_sigmas
        # ll_diff = train_lls.logsumexp(axis=0) - test_lls.logsumexp(axis=0)
        ll_diff, idxs = ll_diff.sort(descending=True)
        print(ll_diff)
        # print(idxs)

        for i, quantile in enumerate(quantiles):
            pos = int(len(gen_feat) * quantile / 100)
            print(pos)
            overfit_idxs[i] = idxs[pos]

        return (
            overfit_idxs,
            self.get_nns(gen_feat, train_feat, overfit_idxs, k=1),
            self.get_nns(gen_feat, test_feat, overfit_idxs, k=1),
        )

    def get_best_samples(self, train_feat, test_feat, gen_feat, best=True, k=10):
        train_feat, baseline_feat, test_feat, gen_feat = preprocess_fls(
            train_feat, test_feat, test_feat, gen_feat, self.pca_dim
        )

        mog_test = MoG(test_feat)
        mog_test.fit(train_feat)
        lls = mog_test.evaluate(gen_feat)
        return lls.topk(k, largest=best).indices

    # def get_nns(self, dataset_1, dataset_2, feature_extractor):
    #     """
    #     For 3 points in dataset_1 closest to dataset_2 (i.e. with minimal nearest neighbors),
    #     return the 3 closest points in dataset_2
    #     """
    #     feat_1, idxs_1 = feature_extractor.get_features(
    #         dataset_1, size=min(len(dataset_1), 30000), get_indices=True
    #     )
    #     feat_2, idxs_2 = feature_extractor.get_features(
    #         dataset_2, size=min(len(dataset_2), 10000), get_indices=True
    #     )
    #     nns = []

    #     # Get closest points
    #     dists = compute_dists(feat_1, feat_2)
    #     dists.fill_diagonal_(float("inf"))  # Ignore self as closest neighbor

    #     min_dists = dists.min(dim=1).values
    #     dataset_1_closest = min_dists.topk(3, largest=False).indices

    #     # Get their nearest neighbors
    #     for idx_1 in dataset_1_closest:
    #         nn = {
    #             "overfit_sample": dataset_1[idxs_1[idx_1]][0],
    #             "nearest_neighbors": [],
    #         }
    #         for idx_2 in dists[idx_1].topk(3, largest=False).indices:
    #             nn["nearest_neighbors"].append(dataset_2[idxs_2[idx_2]][0])
    #         nns.append(nn)
    #     return nns

    # def get_overfit_samples(
    #     self, train_dataset, test_dataset, gen_dataset, feature_extractor, largest=True
    # ):
    #     # Get features and indices from datasets
    #     train_feat, train_idxs = feature_extractor.get_features(
    #         train_dataset, get_indices=True, size=min(len(train_dataset), 30000)
    #     )
    #     test_feat, test_idxs = feature_extractor.get_features(
    #         test_dataset, get_indices=True, size=min(len(test_dataset), 10000)
    #     )
    #     gen_feat, gen_idxs = feature_extractor.get_features(
    #         gen_dataset, get_indices=True, size=min(len(gen_dataset), 10000)
    #     )
    #     train_feat = train_feat.cuda()
    #     test_feat = test_feat.cuda()
    #     gen_feat = gen_feat.cuda()

    #     # Fit MoG to train_feat
    #     gen_log_sigmas, train_losses = optimize_sigmas(
    #         train_feat, gen_feat, init_val=0, verbose=False
    #     )

    #     # Get likelihood of train set/test set for each Gaussian and look at
    #     # gaussians with biggest discrepancies
    #     train_lls = get_pairwise_likelihood(train_feat, gen_feat, gen_log_sigmas)
    #     test_lls = get_pairwise_likelihood(test_feat, gen_feat, gen_log_sigmas)
    #     ll_diff = train_lls.logsumexp(axis=0) - test_lls.logsumexp(axis=0)

    #     top_diffs = ll_diff.topk(3, largest=largest)

    #     res = []  # Store the overfitting samples and highest ll train/test samples
    #     for i, diff_idx in enumerate(top_diffs.indices):
    #         curr_res = {
    #             "overfit_sample": gen_dataset[gen_idxs[diff_idx]][0],
    #             "ll_diff": top_diffs.values[i].item(),
    #             "highest_train": [],
    #             "highest_test": [],
    #         }

    #         # Add 2 highest likelihood train samples to grid as next 3 images
    #         top_train_sample_lls = train_lls[:, diff_idx].topk(10, largest=True).indices
    #         for train_idx in top_train_sample_lls:
    #             curr_res["highest_train"].append(
    #                 train_dataset[train_idxs[train_idx]][0]
    #             )

    #         # Add 2 highest likelihood test samples to grid as next 3 images (as comparison)
    #         top_test_sample_lls = test_lls[:, diff_idx].topk(3, largest=True).indices
    #         for test_idx in top_test_sample_lls:
    #             curr_res["highest_test"].append(test_dataset[test_idxs[test_idx]][0])

    #         res.append(curr_res)
    #     return res
