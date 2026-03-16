#!/usr/bin/env python3
"""
Learn Then Test (LTT) Framework for CPNet-CLM

This module implements the LTT framework with Pareto Testing for finding valid
lambda configurations that provide statistical guarantees on recall.

The LTT approach treats lambda selection as a multiple hypothesis testing problem:
1. Define a candidate grid Λ of lambda configurations
2. Split calibration data into optimization (D_opt) and validation (D_val) sets
3. Use Pareto testing to efficiently search for optimal configurations
4. Perform fixed sequence testing with FWER control

Reference: https://doi.org/10.48550/arXiv.2306.10193
"""

import os
import sys
import numpy as np
from scipy.stats import binom
from typing import List, Tuple, Dict, Set, Optional
import networkx as nx
from dataclasses import dataclass

# Handle imports for both direct execution and module import
try:
    from lambda_rules import LambdaRules, evaluate_lambda_config, generate_lambda_grid
except ImportError:
    from clm.lambda_rules import LambdaRules, evaluate_lambda_config, generate_lambda_grid


@dataclass
class LambdaCandidate:
    """Data class for a lambda configuration candidate."""
    lambda1: float
    lambda2: float
    lambda3: float
    risk: float = 0.0  # 1 - recall
    efficiency: float = 0.0  # 1 / set_size or negative set_size
    p_value: float = 1.0
    
    @property
    def config(self) -> Tuple[float, float, float]:
        return (self.lambda1, self.lambda2, self.lambda3)


def compute_empirical_risk(lambda_config: Tuple[float, float, float],
                           samples: List[Dict], G: nx.Graph,
                           gamma: float = 1.0) -> float:
    """
    Compute empirical risk (1 - recall) for a lambda configuration.
    
    Args:
        lambda_config: (λ₁, λ₂, λ₃) tuple
        samples: List of sample dictionaries with keys:
                 'probs', 'ground_truths', 'cp_set', 'infected_nodes'
        G: NetworkX graph
        gamma: Distance decay parameter
    
    Returns:
        Empirical risk = 1 - average recall
    """
    total_recall = 0.0
    n_samples = len(samples)
    
    for sample in samples:
        recall, _, _ = evaluate_lambda_config(
            lambda_config,
            sample['probs'],
            sample['ground_truths'],
            sample['cp_set'],
            G,
            sample.get('infected_nodes'),
            gamma
        )
        total_recall += recall
    
    avg_recall = total_recall / n_samples if n_samples > 0 else 0.0
    return 1.0 - avg_recall


def compute_efficiency(lambda_config: Tuple[float, float, float],
                       samples: List[Dict], G: nx.Graph,
                       gamma: float = 1.0) -> float:
    """
    Compute efficiency metric (negative average set size).
    
    Lower set size = higher efficiency (more negative).
    
    Args:
        lambda_config: (λ₁, λ₂, λ₃) tuple
        samples: List of sample dictionaries
        G: NetworkX graph
        gamma: Distance decay parameter
    
    Returns:
        Negative average set size
    """
    total_size = 0.0
    n_samples = len(samples)
    
    for sample in samples:
        _, _, size = evaluate_lambda_config(
            lambda_config,
            sample['probs'],
            sample['ground_truths'],
            sample['cp_set'],
            G,
            sample.get('infected_nodes'),
            gamma
        )
        total_size += size
    
    avg_size = total_size / n_samples if n_samples > 0 else 0.0
    return -avg_size  # Negative for efficiency (smaller is better)


def is_pareto_dominated(candidate: LambdaCandidate, 
                        others: List[LambdaCandidate]) -> bool:
    """
    Check if a candidate is Pareto-dominated by any other candidate.
    
    A candidate is dominated if there exists another candidate that is:
    - At least as good in all objectives (risk, efficiency)
    - Strictly better in at least one objective
    
    Args:
        candidate: Candidate to check
        others: List of other candidates
    
    Returns:
        True if candidate is dominated
    """
    for other in others:
        # Check if other dominates candidate
        # Lower risk is better, higher (less negative) efficiency is worse
        at_least_as_good = (other.risk <= candidate.risk and 
                           other.efficiency >= candidate.efficiency)
        strictly_better = (other.risk < candidate.risk or 
                          other.efficiency > candidate.efficiency)
        
        if at_least_as_good and strictly_better:
            return True
    
    return False


def find_pareto_frontier(candidates: List[LambdaCandidate]) -> List[LambdaCandidate]:
    """
    Find the Pareto frontier among candidates.
    
    The Pareto frontier contains all non-dominated candidates.
    
    Args:
        candidates: List of all candidates with computed risk and efficiency
    
    Returns:
        List of candidates on the Pareto frontier
    """
    pareto_frontier = []
    
    for candidate in candidates:
        if not is_pareto_dominated(candidate, candidates):
            pareto_frontier.append(candidate)
    
    return pareto_frontier


def compute_binomial_pvalue(n_samples: int, empirical_risk: float, 
                            epsilon: float) -> float:
    """
    Compute binomial tail p-value for risk control.
    
    p = P(Binom(n, epsilon) <= n * empirical_risk)
    
    Args:
        n_samples: Number of validation samples
        empirical_risk: Observed empirical risk
        epsilon: Target risk level
    
    Returns:
        P-value
    """
    n_failures = int(np.ceil(n_samples * empirical_risk))
    p_value = binom.cdf(n_failures, n_samples, epsilon)
    return p_value


class LTTCalibrator:
    """
    Learn Then Test Calibrator for finding valid lambda configurations.
    
    Implements Pareto Testing to efficiently search the lambda space while
    controlling the family-wise error rate (FWER).
    
    Attributes:
        alpha: Recall error tolerance (we want recall >= 1 - alpha)
        delta: Statistical failure probability
        epsilon: Target risk level
        gamma: Distance decay parameter for lambda rules
        n_grid_points: Number of points per dimension in lambda grid
    """
    
    def __init__(self, alpha: float = 0.1, delta: float = 0.1, 
                 epsilon: float = 0.1, gamma: float = 1.0,
                 n_grid_points: int = 10):
        """
        Initialize LTT calibrator.
        
        Args:
            alpha: Recall error tolerance
            delta: Statistical failure probability (controls FWER)
            epsilon: Target risk level
            gamma: Distance decay for lambda rules
            n_grid_points: Grid resolution per dimension
        """
        self.alpha = alpha
        self.delta = delta
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_grid_points = n_grid_points
    
    def _prepare_samples(self, probs_list: List[np.ndarray],
                         ground_truths_list: List[np.ndarray],
                         cp_sets_list: List[Set[int]],
                         infected_nodes_list: Optional[List[Set[int]]] = None) -> List[Dict]:
        """
        Prepare sample dictionaries for evaluation.
        
        Args:
            probs_list: List of probability arrays
            ground_truths_list: List of ground truth arrays
            cp_sets_list: List of original CP sets
            infected_nodes_list: Optional list of infected node sets
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        n_samples = len(probs_list)
        
        for i in range(n_samples):
            sample = {
                'probs': probs_list[i],
                'ground_truths': ground_truths_list[i],
                'cp_set': cp_sets_list[i]
            }
            if infected_nodes_list is not None:
                sample['infected_nodes'] = infected_nodes_list[i]
            samples.append(sample)
        
        return samples
    
    def _create_candidates(self, samples: List[Dict], G: nx.Graph) -> List[LambdaCandidate]:
        """
        Create candidate list with computed risk and efficiency.
        
        Args:
            samples: List of sample dictionaries
            G: NetworkX graph
        
        Returns:
            List of LambdaCandidate objects
        """
        grid = generate_lambda_grid(self.n_grid_points)
        candidates = []
        
        for l1, l2, l3 in grid:
            config = (l1, l2, l3)
            risk = compute_empirical_risk(config, samples, G, self.gamma)
            efficiency = compute_efficiency(config, samples, G, self.gamma)
            
            candidate = LambdaCandidate(
                lambda1=l1, lambda2=l2, lambda3=l3,
                risk=risk, efficiency=efficiency
            )
            candidates.append(candidate)
        
        return candidates
    
    def calibrate(self, calibration_data: Dict, G: nx.Graph,
                  opt_ratio: float = 0.5,
                  verbose: bool = True) -> Tuple[Optional[Tuple[float, float, float]], 
                                                  List[Tuple[float, float, float]]]:
        """
        Run LTT calibration to find valid lambda configurations.
        
        Args:
            calibration_data: Dictionary with keys:
                - 'probs': List of probability arrays
                - 'ground_truths': List of ground truth arrays
                - 'cp_sets': List of original CP sets
                - 'infected_nodes': Optional list of infected node sets
            G: NetworkX graph
            opt_ratio: Fraction of data for optimization (rest for validation)
            verbose: Print progress information
        
        Returns:
            Tuple of (best_lambda, valid_lambdas_list)
            best_lambda is None if no valid configuration found
        """
        probs_list = calibration_data['probs']
        ground_truths_list = calibration_data['ground_truths']
        cp_sets_list = calibration_data['cp_sets']
        infected_nodes_list = calibration_data.get('infected_nodes')
        
        n_total = len(probs_list)
        n_opt = int(n_total * opt_ratio)
        n_val = n_total - n_opt
        
        if verbose:
            print(f"LTT Calibration: {n_total} samples")
            print(f"  Optimization set: {n_opt}, Validation set: {n_val}")
        
        # Split data
        opt_samples = self._prepare_samples(
            probs_list[:n_opt], ground_truths_list[:n_opt],
            cp_sets_list[:n_opt], 
            infected_nodes_list[:n_opt] if infected_nodes_list else None
        )
        val_samples = self._prepare_samples(
            probs_list[n_opt:], ground_truths_list[n_opt:],
            cp_sets_list[n_opt:],
            infected_nodes_list[n_opt:] if infected_nodes_list else None
        )
        
        # Stage 1: Find Pareto frontier on optimization set
        if verbose:
            print("Stage 1: Finding Pareto frontier on optimization set...")
        
        candidates = self._create_candidates(opt_samples, G)
        pareto_candidates = find_pareto_frontier(candidates)
        
        if verbose:
            print(f"  Found {len(pareto_candidates)} Pareto-optimal candidates")
        
        # Sort by risk (ascending) for sequential testing
        pareto_candidates.sort(key=lambda c: c.risk)
        
        # Stage 2: Fixed sequence testing on validation set
        if verbose:
            print("Stage 2: Sequential testing on validation set...")
        
        valid_lambdas = []
        
        for candidate in pareto_candidates:
            # Compute empirical risk on validation set
            val_risk = compute_empirical_risk(candidate.config, val_samples, G, self.gamma)
            
            # Compute p-value
            p_value = compute_binomial_pvalue(n_val, val_risk, self.epsilon)
            candidate.p_value = p_value
            
            if verbose:
                print(f"  λ=({candidate.lambda1:.2f}, {candidate.lambda2:.2f}, {candidate.lambda3:.2f}): "
                      f"risk={val_risk:.3f}, p={p_value:.3f}")
            
            # Test: accept if p-value >= delta
            if p_value >= self.delta:
                valid_lambdas.append(candidate)
            else:
                # Fixed sequence testing: stop at first failure (FWER control)
                break
        
        if verbose:
            print(f"Found {len(valid_lambdas)} valid configurations")
        
        # Select best (most efficient) among valid configurations
        if len(valid_lambdas) == 0:
            if verbose:
                print("Warning: No valid lambda found. Using default.")
            return None, []
        
        # Select most efficient (largest negative efficiency = smallest set)
        best_candidate = min(valid_lambdas, key=lambda c: -c.efficiency)
        
        if verbose:
            print(f"Best λ: ({best_candidate.lambda1:.2f}, {best_candidate.lambda2:.2f}, "
                  f"{best_candidate.lambda3:.2f})")
        
        return best_candidate.config, [c.config for c in valid_lambdas]


def calibrate_ltt(calibration_data: Dict, G: nx.Graph,
                  alpha: float = 0.1, delta: float = 0.1, 
                  epsilon: float = 0.1, gamma: float = 1.0,
                  n_grid_points: int = 10,
                  verbose: bool = True) -> Tuple[Optional[Tuple[float, float, float]], 
                                                  List[Tuple[float, float, float]]]:
    """
    Convenience function for LTT calibration.
    
    Args:
        calibration_data: Dictionary with calibration samples
        G: NetworkX graph
        alpha: Recall error tolerance
        delta: Statistical failure probability
        epsilon: Target risk level
        gamma: Distance decay parameter
        n_grid_points: Grid resolution
        verbose: Print progress
    
    Returns:
        Tuple of (best_lambda, valid_lambdas_list)
    """
    calibrator = LTTCalibrator(alpha, delta, epsilon, gamma, n_grid_points)
    return calibrator.calibrate(calibration_data, G, verbose=verbose)


if __name__ == '__main__':
    # Example usage
    print("LTT Framework for CPNet-CLM")
    print("=" * 50)
    
    # Create test graph
    G = nx.karate_club_graph()
    n_nodes = G.number_of_nodes()
    
    # Generate synthetic calibration data
    np.random.seed(42)
    n_samples = 100
    
    calibration_data = {
        'probs': [],
        'ground_truths': [],
        'cp_sets': [],
        'infected_nodes': []
    }
    
    for _ in range(n_samples):
        # Random probabilities
        probs = np.random.rand(n_nodes)
        probs = probs / probs.sum()
        
        # Random ground truth (1-3 sources)
        n_sources = np.random.randint(1, 4)
        gt_sources = np.random.choice(n_nodes, n_sources, replace=False)
        ground_truths = np.zeros(n_nodes)
        ground_truths[gt_sources] = 1
        
        # Random CP set (top 15 nodes)
        top_nodes = np.argsort(-probs)[:15]
        cp_set = set(top_nodes)
        
        # Infected nodes (superset of sources)
        infected = set(gt_sources)
        infected.update(np.random.choice(n_nodes, 10, replace=False))
        
        calibration_data['probs'].append(probs)
        calibration_data['ground_truths'].append(ground_truths)
        calibration_data['cp_sets'].append(cp_set)
        calibration_data['infected_nodes'].append(infected)
    
    # Run LTT calibration
    best_lambda, valid_lambdas = calibrate_ltt(
        calibration_data, G,
        alpha=0.1, delta=0.1, epsilon=0.1,
        n_grid_points=5,  # Small for demo
        verbose=True
    )
    
    print(f"\nResult:")
    print(f"  Best λ: {best_lambda}")
    print(f"  Valid λ count: {len(valid_lambdas)}")
