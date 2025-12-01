import torch
import numpy as np
from typing import List
from ot import emd2, emd2_1d, dist


def r2_accuracy(
        pred:   torch.Tensor,
        target: torch.Tensor
    ) -> float:
    """Compute the coefficient of determination (a.k.a index of correlation) between `pred` and `target` [https://en.wikipedia.org/wiki/Coefficient_of_determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).
    
    Args:
        pred   (torch.Tensor): Predicted values
        target (torch.Tensor): Target values. Dimension must match `pred`.

    Returns:
        torch.Tensor: Coefficient of determination.    
    """

    if (pred.dim()==1) or (pred.dim()==2):
        # Remove elements that cause division by 0
        mask = (target!=target.mean().item())
        res = ((target[mask]-pred[mask])**2).sum().item()
        tot = ((target[mask]-target.mean().item())**2).sum().item()
        return 1 - res / tot
    else:
        raise RuntimeError('Invalid dimensions')
    

def w2_distance_nd(
        pred,
        target
    ) -> float:
    """Compute the N(num_nodes)-dimensional Wasserstein-2 distance between two distributions.

    Args:
        pred   (torch.Tensor): Predicted values (num_nodes, num pred fields) or (num_nodes, num pred samples, num pred fields)
        target (torch.Tensor): Target values    (num_nodes, num gt fields)   or (num_nodes, num gt samples,   num gt fields)

    Returns:
        float: N-dimensional Wasserstein-2 distance.

    """

    assert pred.shape[0] == target.shape[0], f"`pred` and `target` must have the same number of nodes: {pred.shape[0]} != {target.shape[0]}"
    assert target.ndim == pred.ndim, f"`target` must have the same number of dimensions as `pred`: {target.ndim} != {pred.ndim}"
    if pred.ndim == 2:
        pass
    elif pred.ndim == 3:
        pred   = pred  .transpose(1, 2).reshape(-1, pred.shape[1])
        target = target.transpose(1, 2).reshape(-1, target.shape[1])
    pred   = pred  .cpu().numpy().T
    target = target.cpu().numpy().T
    # Graph-wise Wasserstein-2 distance
    cost_matric = dist(target, pred, metric='sqeuclidean')
    d = np.sqrt(emd2([], [], cost_matric))
    return d


def w2_distance_1d(
        pred,
        target
    ) -> float:
    """Compute the 1-dimensional Wasserstein-2 distance between two distributions.

    Args:
        pred   (torch.Tensor): Predicted values (num_nodes, num pred fields) or (num_nodes, num pred samples, num pred fields)
        target (torch.Tensor): Target values    (num_nodes, num gt fields)   or (num_nodes, num gt samples,   num gt fields)

    Returns:
        float: 1-dimensional Wasserstein-2 distance.

    """

    assert pred.shape[0] == target.shape[0], f"`pred` and `target` must have the same number of nodes: {pred.shape[0]} != {target.shape[0]}"
    assert pred.ndim == target.ndim, f"`pred` and `target` must have the same number of dimensions: {pred.ndim} != {target.ndim}"
    pred   = pred  .cpu().numpy() 
    target = target.cpu().numpy()
    # Node-wise Wasserstein-2 distance
    if pred.ndim == 2:
        res = [np.sqrt(emd2_1d(target[i], pred[i])) for i in range(target.shape[0])]
    elif pred.ndim == 3:
        res = [np.sqrt(emd2_1d(
            target[i].flatten(),
            pred[i]  .flatten()
        )) for i in range(target.shape[0])]
    else:
        raise RuntimeError('Invalid dimensions')
    return np.mean(res)


def samples_r2_accuracy(
    pred:   torch.Tensor, # (num_nodes, num_samples_pred, num_fields)
    target: torch.Tensor, # (num_nodes, num_samples_gt,   num_fields)
) -> List[float]:
    """Compute the accuracy of each sample in the predicted distribution by comparing it with the closest sample in the target distribution.

    Args:
        pred   (torch.Tensor): Predicted values (num_nodes, num_samples_pred, num_fields)
        target (torch.Tensor): Target values    (num_nodes, num_samples_gt,   num_fields)

    Returns:
        List[float]: Accuracy of each sample in the predicted distribution

    """    

    targets = target.split(1, dim=1)
    acc_list = []
    for p in pred.split(1, dim=1):
        acc = [r2_accuracy(p.squeeze(1), t.squeeze(1)) for t in targets]
        acc = np.max(acc)
        acc_list.append(acc)
    return acc_list