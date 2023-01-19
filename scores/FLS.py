import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from fls.scores.Score import Score

def tensor_to_numpy(tensor):
    """ Shortcut to get a np.array corresponding to tensor """
    return tensor.detach().cpu().numpy()

def compute_dists(x_data, x_kernel):
    """ Returns the dists tensor of all L2^2 distances between samples from x_data and x_kernel """
    return torch.cdist(x_data, x_kernel) ** 2
    # return torch.cdist(x_data, x_kernel, compute_mode='donot_use_mm_for_euclid_dist') ** 2

def nll(dists, log_sigmas, dim, detailed=False):
    """ Computes the negative KDE log-likelihood using the distances between x_data and x_kernel
    
    Args:
    - dists: N x M tensor where the i,j entry is the squared L2 distance between the i-th row of x_data and the j-th row of x_kernel
        - x_data is N x dim and x_kernel is M x dim (where x_kernel are the points of the KDE)
        - dists is passed as an argument so that it can be computed once and cached (as it is O(N x M x dim))
    - log_sigmas: Tensor of size M where the i-th entry is the log of the bandwidth for the i-th kernel point
    - dim: Dimension of the data (passed as argument since it cannot be inferred from the dists)
    
    Returns: The NLL of the above
    """
    exponent_term = (-.5 * dists)/torch.exp(log_sigmas)

    # Here we use that dividing by x is equivalent to multiplying by e^{-ln(x)}
    # allows for use of logsumexp
    exponent_term -= (dim/2) * log_sigmas
    inner_term = torch.logsumexp(exponent_term, axis=1)
    
    if detailed:
        return -torch.sum(inner_term)/dists.shape[0], -inner_term

    return -torch.sum(inner_term)/dists.shape[0]

def optimize_sigmas(x_data, x_kernel, init_val=1, verbose=False):
    """ Find the sigmas that minimize the NLL of x_data under a kernel given by x_kernel
    
    Args:
    - x_data: N x dim tensor we are evaluating the NLL of
    - x_kernel: M x dim tensor of points to use as kernels for KDE
    - init_val: Initial value of tensor of log_sigmas
    - optim_iter: Number of iterations of SGD to perform
    - verbose: Whether to print optimization progress
    - plot: Whether to plot loss progression and sigmas histogram
    
    Returns: (log_sigmas, losses)
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
        
        # Here we clamp log_sigmas or the values explode
        with torch.no_grad():
            log_sigmas.data = log_sigmas.clamp(-100, 20).data
            
        if verbose and i % 25 == 0:
            print(f"Loss: {loss:.2f} | Sigmas: min({torch.min(log_sigmas):.4f}), mean({torch.mean(log_sigmas):.4f}), max({torch.max(log_sigmas):.2f})")
              
        losses.append(tensor_to_numpy(loss))
        
    return log_sigmas, losses

def evaluate_set(evaluated_set, x_kernel, log_sigmas):
    """ Gets the NLL of the test set using given kernel/bandwidths """
    dists = compute_dists(evaluated_set, x_kernel)
    nlls = nll(dists, log_sigmas, x_kernel.shape[1], detailed=True)
    return tensor_to_numpy(nlls[0]), nlls[1]

def diff_over_mean(gen_nll, baseline_nll):
    """ Returns a scaled difference score """
    score = (gen_nll - baseline_nll)/(abs(gen_nll) + abs(baseline_nll))
    score = score * 100
    return score

def percentage(gen_nll, baseline_nll):
    score = 1 - (gen_nll - baseline_nll)/abs(gen_nll)
    # score = gen_nll/baseline_nll
    score = score * 100
    return score

def plot_dist(name, dist, ax):
    # ax.set_title(f"{name}: Min: {dist.min()} | Mean: {dist.mean()} | Max: {dist.max()}")
    ax.hist(dist, bins=50, alpha=0.5, label=name)

def get_optimized_nll(x_data, x_kernel, x_test):
    log_sigmas, losses = optimize_sigmas(x_data, x_kernel, init_val=0)
    nlls = evaluate_set(x_test, x_kernel, log_sigmas)
    return nlls, min(losses)


class FLS(Score):
    def __init__(self, mode="general_recall"):
        super().__init__()
        self.name = f"FLS_{mode}"
        self.mode = mode

    def compute_score(self, train_feat, baseline_feat, test_feat, gen_feat, plot=False):
        # Assert correct device
        train_feat = train_feat.cuda()
        baseline_feat = baseline_feat.cuda()
        test_feat = test_feat.cuda()
        gen_feat = gen_feat.cuda()

        all_features = torch.cat((train_feat, baseline_feat, test_feat, gen_feat), dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)

        def normalize(feat):
            return (feat - mean)/std

        train_feat = normalize(train_feat)
        baseline_feat = normalize(baseline_feat)
        test_feat = normalize(test_feat)
        gen_feat = normalize(gen_feat)

        # # Assert appropriate sizes for both kernels (if different would yield incomparable LLs)
        # if gen_feat.shape[0] != kernel_size:
        #     raise ValueError(f"Incorrect gen size: {gen_feat.shape[0]} != {kernel_size}")
        # if baseline_feat.shape[0] != kernel_size:
        #     raise ValueError(f"Incorrect baseline size: {baseline_feat.shape[0]} != {kernel_size}")
    
        if self.mode == "overfit_precision":
            gen_log_sigmas, train_losses = optimize_sigmas(gen_feat, train_feat, init_val=0)
            gen_log_sigmas, test_losses = optimize_sigmas(gen_feat, test_feat, init_val=0)
            return diff_over_mean(min(test_losses), min(train_losses))

        if self.mode == "overfit_recall":
            gen_log_sigmas, gen_losses = optimize_sigmas(train_feat, gen_feat, init_val=0)
            test_nll = evaluate_set(test_feat, gen_feat, gen_log_sigmas)

            diff = diff_over_mean(test_nll[0].item(), min(gen_losses))
            return 2 ** (-(diff)) * 100

        if self.mode == "general_recall":
            gen_log_sigmas, gen_losses = optimize_sigmas(train_feat, gen_feat, init_val=0)
            gen_nlls = evaluate_set(test_feat, gen_feat, gen_log_sigmas)
            gen_nll = gen_nlls[0].item()
        
            # Get baseline_nll
            baseline_nlls, baseline_min_nll = get_optimized_nll(train_feat, baseline_feat, test_feat)
            baseline_nll = baseline_nlls[0].item()
        
            return percentage(gen_nll, baseline_nll)
        
        if self.mode == "general_precision":
            gen_log_sigmas, gen_losses = optimize_sigmas(gen_feat, test_feat, init_val=0)

            # Get baseline_nll
            baseline_log_sigmas, baseline_losses = optimize_sigmas(baseline_feat, test_feat, init_val=0)
            return diff_over_mean(min(gen_losses), min(baseline_losses))


    def get_overfit_samples(self, train_feat, test_feat, gen_feat, train_idxs, gen_idxs, train_dataset, gen_dataset):
        train_feat = train_feat.cuda()
        test_feat = test_feat.cuda()
        gen_feat = gen_feat.cuda()

        gen_log_sigmas, train_losses = optimize_sigmas(train_feat, gen_feat, init_val=0, verbose=True)

        def get_exponent_term(x_data, x_kernel, sigmas):
            dists = compute_dists(x_data, x_kernel)
            exponent_term = (-.5 * dists)/torch.exp(sigmas)
            exponent_term -= (x_kernel.shape[1]/2) * sigmas
            return exponent_term

        train_lls = get_exponent_term(train_feat, gen_feat, gen_log_sigmas)
        test_lls = get_exponent_term(test_feat, gen_feat, gen_log_sigmas)

        ll_diff = train_lls.logsumexp(axis=0) - test_lls.logsumexp(axis=0)
        top_diffs = ll_diff.topk(10, largest=True).indices

        overfit_grids = []
        for diff in top_diffs:
            gen_idx = diff.item()
            top_train_sample_lls = train_lls[:, gen_idx].topk(3, largest=True).indices

            overfit_grid = [gen_dataset[gen_idxs[gen_idx]][0]]
            for train_idx in top_train_sample_lls:
                overfit_grid.append(train_dataset[train_idxs[train_idx.item()]][0])

            overfit_grids.append(overfit_grid)

        return overfit_grids