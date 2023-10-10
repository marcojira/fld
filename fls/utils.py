import torch
import numpy as np


def shuffle(tensor: torch.Tensor, size=None):
    """Gets randomly ordered subset of `tensor` of `size`"""
    idx = np.random.choice(len(tensor), size if size else len(tensor))
    return tensor[idx]
