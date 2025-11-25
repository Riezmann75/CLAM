from typing import Optional
import torch
import numpy as np


def mixup_dataset(
    features: torch.Tensor,
    labels: tuple,
    alpha: float = 1.0,
    device: Optional[torch.device] = torch.device("cpu"),
):
    # Generate random mixup pairs from batch
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = features.shape[0]
    permutation = torch.randperm(batch_size).to(device)
    mixed_features = lam * features + (1 - lam) * features[permutation, :]
    permutated_labels = labels[permutation, :]
    # shape (batch_size, num_stacked_tensors, num_features)
    concatenated_labels = torch.stack((labels, permutated_labels), dim=1)
    return mixed_features, concatenated_labels, lam
