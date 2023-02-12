import math
import torch
from fls.metrics.Metric import Metric


def preprocess_fls(train_feat, baseline_feat, test_feat, gen_feat):
    # Assert correct device
    train_feat = train_feat.cuda()
    baseline_feat = baseline_feat.cuda()
    test_feat = test_feat.cuda()
    gen_feat = gen_feat.cuda()

    # Normalize features to 0 mean, unit variance
    all_features = torch.cat((train_feat, baseline_feat, test_feat, gen_feat), dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)

    def normalize(feat):
        return (feat - mean) / std

    train_feat = normalize(train_feat)
    baseline_feat = normalize(baseline_feat)
    test_feat = normalize(test_feat)
    gen_feat = normalize(gen_feat)
    return train_feat, baseline_feat, test_feat, gen_feat


def tensor_to_numpy(tensor):
    """Shortcut to get a np.array corresponding to tensor"""
    return tensor.detach().cpu().numpy()


def compute_dists(x_data, x_kernel):
    """Returns the dists tensor of all L2^2 distances between samples from x_data and x_kernel"""
    # More accurate but slower
    # return (
    #     torch.cdist(x_data, x_kernel, compute_mode="donot_use_mm_for_euclid_dist")
    # ) ** 2

    dists = (torch.cdist(x_data, x_kernel)) ** 2
    return dists.detach()


def get_pairwise_likelihood(x_data, x_kernel, log_sigmas):
    dists = compute_dists(x_data, x_kernel)
    exponent_term = (-0.5 * dists) / torch.exp(log_sigmas)
    exponent_term -= (x_kernel.shape[1] / 2) * log_sigmas
    exponent_term += torch.log(torch.tensor(1 / dists.shape[1]))
    return exponent_term


def nll(dists, log_sigmas, dim, lambd=0, detailed=False):
    """Computes the negative KDE log-likelihood using the distances between x_data and x_kernel

    Args:
    - dists: N x M tensor where the i,j entry is the squared L2 distance between the i-th row of x_data and the j-th row of x_kernel
        - x_data is N x dim and x_kernel is M x dim (where x_kernel are the points of the KDE)
        - dists is passed as an argument so that it can be computed once and cached (as it is O(N x M x dim))
    - log_sigmas: Tensor of size M where the i-th entry is the log of the bandwidth for the i-th kernel point
    - dim: Dimension of the data (passed as argument since it cannot be inferred from the dists)
    - lambd: Optional regularization parameter for the sigmas (default 0)
    - detailed: If True, returns the NLL of each datapoint as well as the mean

    Returns: The NLL of the above
    """
    exponent_term = (-0.5 * dists) / torch.exp(log_sigmas)

    # Here we use that dividing by x is equivalent to multiplying by e^{-ln(x)}
    # allows for use of logsumexp
    exponent_term -= (dim / 2) * log_sigmas
    exponent_term += torch.log(torch.tensor(1 / dists.shape[1]))
    inner_term = torch.logsumexp(exponent_term, dim=1)

    total_nll = torch.mean(-inner_term)

    reg_term = lambd / 2 * torch.norm(torch.exp(-log_sigmas)) ** 2
    final_nll = total_nll + reg_term

    if detailed:
        return final_nll, -inner_term

    return final_nll


def optimize_sigmas(x_data, x_kernel, init_val=1, verbose=False):
    """Find the sigmas that minimize the NLL of x_data under a kernel given by x_kernel

    Args:
    - x_data: N x dim tensor we are evaluating the NLL of
    - x_kernel: M x dim tensor of points to use as kernels for KDE
    - init_val: Initial value of tensor of log_sigmas
    - verbose: Whether to print optimization progress

    Returns: (log_sigmas, losses)
    - log_sigmas: Tensor of log of sigmas
    - losses: List of losses at each step of optimization
    """
    # Tracking
    losses = []

    log_sigmas = torch.ones(x_kernel.shape[0], requires_grad=True, device="cuda")
    log_sigmas.data = init_val * log_sigmas

    optim = torch.optim.Adam([log_sigmas], lr=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[50], gamma=0.1)

    dim = x_data.shape[1]

    # Precompute dists
    dists = compute_dists(x_data, x_kernel)

    for i in range(100):
        loss = nll(dists, log_sigmas, dim)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        # Here we clamp log_sigmas to stop values exploding for identical samples
        with torch.no_grad():
            log_sigmas.data = log_sigmas.clamp(-100, 20).data

        if verbose and i % 25 == 0:
            print(
                f"Loss: {loss:.2f} | Sigmas: min({torch.min(log_sigmas):.4f}), mean({torch.mean(log_sigmas):.4f}), max({torch.max(log_sigmas):.2f})"
            )

        losses.append(tensor_to_numpy(loss))

    return log_sigmas, losses


def evaluate_set(evaluated_set, x_kernel, log_sigmas):
    """Gets the NLL of the test set using given kernel/bandwidths"""
    dists = compute_dists(evaluated_set, x_kernel)
    nlls = nll(dists, log_sigmas, x_kernel.shape[1], detailed=True)
    return tensor_to_numpy(nlls[0]), nlls[1]


class FLS(Metric):
    def __init__(self, mode=""):
        super().__init__()
        self.mode = mode
        self.name = f"FLS{mode}"

    def compute_metric(
        self, train_feat, baseline_feat, test_feat, gen_feat, plot=False
    ):
        """Preprocess"""
        train_feat, baseline_feat, test_feat, gen_feat = preprocess_fls(
            train_feat, baseline_feat, test_feat, gen_feat
        )

        """ Fit MOG to train_feat """
        gen_log_sigmas, gen_losses = optimize_sigmas(train_feat, gen_feat, init_val=0)

        """Compute score"""
        if self.mode == "":
            # Get gen_nll
            gen_nlls = evaluate_set(test_feat, gen_feat, gen_log_sigmas)
            gen_nll = gen_nlls[0].item()

            # Get baseline_nll
            base_log_sigmas, base_losses = optimize_sigmas(
                train_feat, baseline_feat, init_val=0
            )
            baseline_nlls = evaluate_set(test_feat, baseline_feat, base_log_sigmas)
            baseline_nll = baseline_nlls[0].item()

            diff = 2 * (gen_nll - baseline_nll) / train_feat.shape[1]

            score = math.e ** (-diff) * 100
            return score

        elif self.mode == "overfit":
            # Ensure both sets have the same amount of data points
            size = min(test_feat.shape[0], train_feat.shape[0])

            train_lls = get_pairwise_likelihood(
                train_feat[:size], gen_feat, gen_log_sigmas
            )
            test_lls = get_pairwise_likelihood(
                test_feat[:size], gen_feat, gen_log_sigmas
            )

            ll_diff = train_lls.logsumexp(axis=0) - test_lls.logsumexp(axis=0)
            score = ((ll_diff > 0).sum().item() / ll_diff.shape[0]) * 100
            return score - 50

        else:
            raise Exception("Invalid mode for FLS metric")

    def get_nns(self, dataset_1, dataset_2, feature_extractor):
        """
        For 3 points in dataset_1 closest to dataset_2 (i.e. with minimal nearest neighbors),
        return the 3 closest points in dataset_2
        """
        feat_1, idxs_1 = feature_extractor.get_features(
            dataset_1, size=min(len(dataset_1), 30000), get_indices=True
        )
        feat_2, idxs_2 = feature_extractor.get_features(
            dataset_2, size=min(len(dataset_2), 10000), get_indices=True
        )
        nns = []

        # Get closest points
        dists = compute_dists(feat_1, feat_2)
        dists.fill_diagonal_(float("inf"))  # Ignore self as closest neighbor

        min_dists = dists.min(dim=1).values
        dataset_1_closest = min_dists.topk(3, largest=False).indices

        # Get their nearest neighbors
        for idx_1 in dataset_1_closest:
            nn = {
                "overfit_sample": dataset_1[idxs_1[idx_1]][0],
                "nearest_neighbors": [],
            }
            for idx_2 in dists[idx_1].topk(3, largest=False).indices:
                nn["nearest_neighbors"].append(dataset_2[idxs_2[idx_2]][0])
            nns.append(nn)
        return nns

    def get_overfit_samples(
        self, train_dataset, test_dataset, gen_dataset, feature_extractor, largest=True
    ):
        # Get features and indices from datasets
        train_feat, train_idxs = feature_extractor.get_features(
            train_dataset, get_indices=True, size=min(len(train_dataset), 30000)
        )
        test_feat, test_idxs = feature_extractor.get_features(
            test_dataset, get_indices=True, size=min(len(test_dataset), 10000)
        )
        gen_feat, gen_idxs = feature_extractor.get_features(
            gen_dataset, get_indices=True, size=min(len(gen_dataset), 10000)
        )
        train_feat = train_feat.cuda()
        test_feat = test_feat.cuda()
        gen_feat = gen_feat.cuda()

        # Fit MoG to train_feat
        gen_log_sigmas, train_losses = optimize_sigmas(
            train_feat, gen_feat, init_val=0, verbose=False
        )

        # Get likelihood of train set/test set for each Gaussian and look at
        # gaussians with biggest discrepancies
        train_lls = get_pairwise_likelihood(train_feat, gen_feat, gen_log_sigmas)
        test_lls = get_pairwise_likelihood(test_feat, gen_feat, gen_log_sigmas)
        ll_diff = train_lls.logsumexp(axis=0) - test_lls.logsumexp(axis=0)

        top_diffs = ll_diff.topk(3, largest=largest)

        res = []  # Store the overfitting samples and highest ll train/test samples
        for i, diff_idx in enumerate(top_diffs.indices):
            curr_res = {
                "overfit_sample": gen_dataset[gen_idxs[diff_idx]][0],
                "ll_diff": top_diffs.values[i].item(),
                "highest_train": [],
                "highest_test": [],
            }

            # Add 2 highest likelihood train samples to grid as next 3 images
            top_train_sample_lls = train_lls[:, diff_idx].topk(10, largest=True).indices
            for train_idx in top_train_sample_lls:
                curr_res["highest_train"].append(
                    train_dataset[train_idxs[train_idx]][0]
                )

            # Add 2 highest likelihood test samples to grid as next 3 images (as comparison)
            top_test_sample_lls = test_lls[:, diff_idx].topk(3, largest=True).indices
            for test_idx in top_test_sample_lls:
                curr_res["highest_test"].append(test_dataset[test_idxs[test_idx]][0])

            res.append(curr_res)
        return res
