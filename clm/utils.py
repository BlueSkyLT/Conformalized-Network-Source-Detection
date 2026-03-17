#!/usr/bin/env python3
"""
Utility functions for CPNet-CLM.

This module provides pure NumPy/PyTorch implementations of scoring functions
used in conformal prediction, without TensorFlow dependencies.

These functions are adapted from utils/score_convert.py and utils/functions.py
to avoid TensorFlow import dependencies.
"""

import numpy as np
from typing import List, Union, Optional

# Optional PyTorch support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def cpquantile(z: np.ndarray, tau: float) -> float:
    """
    Custom implementation of inf quantile for conformal prediction.
    
    Args:
        z: Array of conformity scores
        tau: Quantile level (0 < tau < 1)
    
    Returns:
        The tau-th quantile of z
    """
    z_sorted = np.sort(z)
    n = len(z)
    rank = tau * n
    idx = int(np.ceil(rank) - 1)  # smallest index where F_z(v) >= tau
    return z_sorted[idx] if idx < n else z_sorted[-1]


def set_truncate(set_onehot: np.ndarray, prob: np.ndarray, pow: float) -> np.ndarray:
    """
    Truncate the input set to contain only the largest 'pow' proportion of probabilities.
    
    Args:
        set_onehot: One-hot vector representing the original set
        prob: Predicted probability for each entry
        pow: Proportion of elements to retain in the truncated set
    
    Returns:
        One-hot vector for the truncated set
    """
    set_origin = np.nonzero(np.array(set_onehot))[0]
    prob_quantile = -cpquantile(-prob[set_origin], pow)
    
    set_truncated = np.array(set_onehot)
    set_truncated[np.where(prob < prob_quantile)[0]] = 0
    
    return set_truncated


def recall_score(pred_prob: np.ndarray, ground_truth: np.ndarray, 
                 prop_model: str, infected_nodes: np.ndarray) -> float:
    """
    Compute the recall-based conformity score for a single sample.
    
    Args:
        pred_prob: Predicted probabilities for each node
        ground_truth: One-hot vector of ground truth sources
        prop_model: Propagation model ('SI' or 'SIR') - kept for API compatibility
                    with utils.score_convert, currently uses indicator masking
        infected_nodes: Indices of infected nodes at earliest time step
    
    Returns:
        Recall score (lower is better prediction)
    """
    n_nodes = len(pred_prob)
    
    # Adjust pred_prob according to propagation model
    pi_hat = np.array(pred_prob)
    
    indicator_vec = np.zeros(n_nodes)
    indicator_vec[infected_nodes] = 1
    
    pi_hat = pi_hat * indicator_vec
    
    # Find the smallest pi_hat inside the ground truth
    gt_indices = np.where(ground_truth > 0)[0]
    min_pi_hat = np.min(pi_hat[gt_indices])
    
    # Find all nodes with pi_hat >= min_pi_hat
    indices = np.where(pi_hat >= min_pi_hat)[0]
    
    # Compute the sum of probability inside the indices
    total_prob = np.sum(pi_hat)
    if total_prob == 0:
        return 1.0  # Edge case
    pred_score = np.sum(pi_hat[indices]) / total_prob
    
    return pred_score


def recall_score_gtunknown(pred_prob: np.ndarray, prop_model: str, 
                           infected_nodes: np.ndarray) -> np.ndarray:
    """
    Compute recall-based conformity scores for all possible labels (ground truth unknown).
    
    This is used on the test set where we need to compute scores for all possible labels.
    
    Args:
        pred_prob: Predicted probabilities for each node
        prop_model: Propagation model ('SI' or 'SIR') - kept for API compatibility
                    with utils.score_convert, currently uses indicator masking
        infected_nodes: Indices of infected nodes at earliest time step
    
    Returns:
        Array of conformity scores for each node
    """
    n_nodes = len(pred_prob)
    
    # Adjust pred_prob according to propagation model
    pi_hat = np.array(pred_prob)
    
    indicator_vec = np.zeros(n_nodes)
    indicator_vec[infected_nodes] = 1
    
    pi_hat = pi_hat * indicator_vec
    
    # Sort all probabilities in descending order
    sorted_indices = np.argsort(-pi_hat)
    sorted_pi_hat = pi_hat[sorted_indices]
    
    # Compute scores
    pred_scores = np.zeros(n_nodes)
    probability_sum = 0
    total_probability = np.sum(pi_hat)
    
    if total_probability == 0:
        return pred_scores  # All zeros
    
    for sort_idx in range(n_nodes):
        node = sorted_indices[sort_idx]
        probability_sum += sorted_pi_hat[sort_idx]
        pred_scores[node] = probability_sum / total_probability
    
    return pred_scores


# PyTorch versions of the functions (for future GPU acceleration)
if HAS_TORCH:
    def cpquantile_torch(z: torch.Tensor, tau: float) -> torch.Tensor:
        """PyTorch version of cpquantile."""
        z_sorted, _ = torch.sort(z)
        n = len(z)
        rank = tau * n
        idx = int(np.ceil(rank) - 1)
        return z_sorted[idx] if idx < n else z_sorted[-1]
    
    def recall_score_torch(pred_prob: torch.Tensor, ground_truth: torch.Tensor,
                           infected_nodes: torch.Tensor) -> torch.Tensor:
        """PyTorch version of recall_score."""
        n_nodes = len(pred_prob)
        
        # Create indicator vector for infected nodes
        indicator_vec = torch.zeros(n_nodes, device=pred_prob.device)
        indicator_vec[infected_nodes] = 1
        
        pi_hat = pred_prob * indicator_vec
        
        # Find ground truth indices
        gt_indices = torch.nonzero(ground_truth > 0, as_tuple=True)[0]
        min_pi_hat = torch.min(pi_hat[gt_indices])
        
        # Find indices with pi_hat >= min_pi_hat
        mask = pi_hat >= min_pi_hat
        
        total_prob = torch.sum(pi_hat)
        if total_prob == 0:
            return torch.tensor(1.0, device=pred_prob.device)
        
        pred_score = torch.sum(pi_hat[mask]) / total_prob
        return pred_score
    
    def recall_score_gtunknown_torch(pred_prob: torch.Tensor,
                                      infected_nodes: torch.Tensor) -> torch.Tensor:
        """PyTorch version of recall_score_gtunknown."""
        n_nodes = len(pred_prob)
        
        indicator_vec = torch.zeros(n_nodes, device=pred_prob.device)
        indicator_vec[infected_nodes] = 1
        
        pi_hat = pred_prob * indicator_vec
        
        # Sort in descending order
        sorted_pi_hat, sorted_indices = torch.sort(pi_hat, descending=True)
        
        pred_scores = torch.zeros(n_nodes, device=pred_prob.device)
        total_probability = torch.sum(pi_hat)
        
        if total_probability == 0:
            return pred_scores
        
        cumsum = torch.cumsum(sorted_pi_hat, dim=0)
        
        # Map back to original indices using vectorized indexing
        pred_scores[sorted_indices] = cumsum / total_probability
        
        return pred_scores


def avg_score(pred_prob: np.ndarray, ground_truth: np.ndarray,
              prop_model: str, infected_nodes: np.ndarray) -> float:
    """
    Compute the average-based conformity score for a single sample.
    
    Args:
        pred_prob: Predicted probabilities for each node
        ground_truth: One-hot vector of ground truth sources
        prop_model: Propagation model ('SI' or 'SIR') - kept for API compatibility
                    with utils.score_convert, currently uses indicator masking
        infected_nodes: Indices of infected nodes at earliest time step
    
    Returns:
        Average score (more negative is better prediction)
    """
    n_nodes = len(pred_prob)
    
    # Adjust pred_prob according to propagation model
    pi_hat = np.array(pred_prob)
    
    indicator_vec = np.zeros(n_nodes)
    indicator_vec[infected_nodes] = 1
    
    pi_hat = pi_hat * indicator_vec
    
    # Find the smallest pi_hat inside the ground truth
    gtsource = np.nonzero(ground_truth)[0]
    min_pi_hat = np.min(pi_hat[gtsource])
    
    # Find all nodes with pi_hat >= min_pi_hat
    indices = np.where(pi_hat >= min_pi_hat)[0]
    
    # Compute the average score inside the indices
    pred_score = np.mean(pred_prob[indices])
    pred_score = -pred_score  # Small score implies good prediction
    
    return pred_score


def avg_score_gtunknown(pred_prob: np.ndarray, prop_model: str,
                        infected_nodes: np.ndarray) -> np.ndarray:
    """
    Compute average-based conformity scores for all possible labels.
    
    Args:
        pred_prob: Predicted probabilities for each node
        prop_model: Propagation model ('SI' or 'SIR') - kept for API compatibility
                    with utils.score_convert, currently uses indicator masking
        infected_nodes: Indices of infected nodes at earliest time step
    
    Returns:
        Array of conformity scores for each node
    """
    n_nodes = len(pred_prob)
    
    # Adjust pred_prob according to propagation model
    pi_hat = np.array(pred_prob)
    
    indicator_vec = np.zeros(n_nodes)
    indicator_vec[infected_nodes] = 1
    
    pi_hat = pi_hat * indicator_vec
    
    # Sort all probabilities in descending order
    sorted_indices = np.argsort(-pi_hat)
    sorted_pi_hat = pi_hat[sorted_indices]
    
    # Compute scores
    pred_scores = np.zeros(n_nodes)
    probability_sum = 0
    
    for sort_idx in range(n_nodes):
        node = sorted_indices[sort_idx]
        probability_sum += sorted_pi_hat[sort_idx]
        pred_scores[node] = probability_sum / (sort_idx + 1)
    
    pred_scores = -pred_scores  # Small score implies good prediction
    
    return pred_scores
