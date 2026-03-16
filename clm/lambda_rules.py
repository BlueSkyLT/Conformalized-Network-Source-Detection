#!/usr/bin/env python3
"""
Lambda Rules for CPNet-CLM (Conformal Language Modeling applied to Network Source Detection)

This module implements three lambda-based rules inspired by CLM's rejection and stopping rules:
1. λ₁ - Diversity Rejection: Rejects nodes too close to already-selected sources
2. λ₂ - Quality Rejection: Rejects nodes with probability below threshold  
3. λ₃ - Stopping Rule: Stops when cumulative probability exceeds threshold

Reference: https://doi.org/10.48550/arXiv.2306.10193
"""

import numpy as np
import networkx as nx
from typing import Set, List, Tuple, Dict, Optional


def distance_kernel(dist: float, gamma: float = 1.0) -> float:
    """
    Distance decay kernel function K(d) = exp(-gamma * d).
    Closer nodes (smaller distance) get higher weights.
    
    Args:
        dist: Graph distance between two nodes
        gamma: Decay parameter (default: 1.0)
    
    Returns:
        Kernel weight in (0, 1]
    """
    return np.exp(-gamma * dist)


def compute_graph_distances(G: nx.Graph, nodes: List[int]) -> Dict[Tuple[int, int], int]:
    """
    Compute pairwise shortest path distances between nodes in graph G.
    
    Args:
        G: NetworkX graph
        nodes: List of node indices to compute distances for
    
    Returns:
        Dictionary mapping (i, j) -> distance
    """
    distances = {}
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i < j:
                try:
                    dist = nx.shortest_path_length(G, source=node_i, target=node_j)
                except nx.NetworkXNoPath:
                    dist = float('inf')
                distances[(node_i, node_j)] = dist
                distances[(node_j, node_i)] = dist
    return distances


class LambdaRules:
    """
    CLM-inspired lambda rules for conformal prediction set refinement.
    
    The rules work as a post-processing step on the conformal prediction set:
    1. Start with an initial CP set C
    2. Apply λ₁ (diversity rejection) to remove redundant nodes
    3. Apply λ₂ (quality rejection) to filter low-probability nodes
    4. Apply λ₃ (stopping rule) to limit set size
    
    Attributes:
        lambda1: Diversity rejection threshold
        lambda2: Quality (minimum probability) threshold
        lambda3: Stopping rule (cumulative probability) threshold
        gamma: Distance decay parameter for kernel function
    """
    
    def __init__(self, lambda1: float = 0.5, lambda2: float = 0.1, 
                 lambda3: float = 0.9, gamma: float = 1.0):
        """
        Initialize lambda rules with given thresholds.
        
        Args:
            lambda1: Diversity rejection threshold (reject if max(K(d)*π) > λ₁)
            lambda2: Minimum probability threshold (reject if π < λ₂)
            lambda3: Cumulative probability threshold for stopping
            gamma: Distance decay parameter
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.gamma = gamma
    
    def diversity_rejection_score(self, v: int, selected_set: Set[int], 
                                   probs: np.ndarray, G: nx.Graph) -> float:
        """
        Compute the diversity rejection score for node v.
        
        Score = max_{v_j ∈ C_λ} { K(Dist(v, v_j)) * π̂(v) }
        
        If score > λ₁, reject node v (too similar to existing selections).
        
        Args:
            v: Node index to evaluate
            selected_set: Set of already-selected source nodes
            probs: Array of P(source) for each node
            G: NetworkX graph for computing distances
        
        Returns:
            Diversity rejection score
        """
        if len(selected_set) == 0:
            return 0.0
        
        max_score = 0.0
        prob_v = probs[v]
        
        for v_j in selected_set:
            try:
                dist = nx.shortest_path_length(G, source=v, target=v_j)
            except nx.NetworkXNoPath:
                dist = float('inf')
            
            kernel_weight = distance_kernel(dist, self.gamma)
            score = kernel_weight * prob_v
            max_score = max(max_score, score)
        
        return max_score
    
    def apply_diversity_rejection(self, candidate_nodes: List[int], 
                                   probs: np.ndarray, G: nx.Graph) -> List[int]:
        """
        Apply λ₁ diversity rejection rule iteratively.
        
        Process nodes in descending probability order, reject if too close to 
        already-selected nodes.
        
        Args:
            candidate_nodes: List of candidate node indices
            probs: Array of P(source) for each node
            G: NetworkX graph
        
        Returns:
            List of remaining nodes after diversity rejection
        """
        # Sort by probability (descending)
        sorted_nodes = sorted(candidate_nodes, key=lambda x: probs[x], reverse=True)
        
        selected_set = set()
        accepted_nodes = []
        
        for v in sorted_nodes:
            score = self.diversity_rejection_score(v, selected_set, probs, G)
            
            if score <= self.lambda1:
                # Accept node
                selected_set.add(v)
                accepted_nodes.append(v)
        
        return accepted_nodes
    
    def apply_quality_rejection(self, candidate_nodes: List[int], 
                                 probs: np.ndarray) -> List[int]:
        """
        Apply λ₂ quality rejection rule.
        
        Reject node v if π̂(v) < λ₂
        
        Args:
            candidate_nodes: List of candidate node indices
            probs: Array of P(source) for each node
        
        Returns:
            List of remaining nodes with probability >= λ₂
        """
        return [v for v in candidate_nodes if probs[v] >= self.lambda2]
    
    def apply_stopping_rule(self, candidate_nodes: List[int], 
                            probs: np.ndarray) -> List[int]:
        """
        Apply λ₃ stopping rule.
        
        Stop adding nodes when cumulative probability sum exceeds λ₃.
        
        Args:
            candidate_nodes: List of candidate node indices (should be sorted by prob)
            probs: Array of P(source) for each node
        
        Returns:
            List of nodes before stopping threshold
        """
        # Sort by probability (descending) if not already
        sorted_nodes = sorted(candidate_nodes, key=lambda x: probs[x], reverse=True)
        
        cumulative_prob = 0.0
        accepted_nodes = []
        
        for v in sorted_nodes:
            cumulative_prob += probs[v]
            accepted_nodes.append(v)
            
            if cumulative_prob >= self.lambda3:
                break
        
        return accepted_nodes
    
    def refine_cp_set(self, cp_set: Set[int], probs: np.ndarray, 
                      G: nx.Graph, infected_nodes: Optional[Set[int]] = None) -> Set[int]:
        """
        Refine a conformal prediction set using all lambda rules.
        
        Processing order:
        1. Filter to infected nodes only (if provided)
        2. Apply λ₂ quality rejection
        3. Apply λ₁ diversity rejection  
        4. Apply λ₃ stopping rule
        
        Args:
            cp_set: Original conformal prediction set
            probs: Array of P(source) for each node
            G: NetworkX graph
            infected_nodes: Optional set of infected nodes to restrict to
        
        Returns:
            Refined prediction set
        """
        candidates = list(cp_set)
        
        # Optionally restrict to infected nodes
        if infected_nodes is not None:
            candidates = [v for v in candidates if v in infected_nodes]
        
        # Step 1: Quality rejection
        candidates = self.apply_quality_rejection(candidates, probs)
        
        # Step 2: Diversity rejection
        candidates = self.apply_diversity_rejection(candidates, probs, G)
        
        # Step 3: Stopping rule
        candidates = self.apply_stopping_rule(candidates, probs)
        
        return set(candidates)
    
    def compute_rejection_status(self, cp_set: Set[int], refined_set: Set[int]) -> Dict[int, str]:
        """
        Compute rejection status for each node in the original CP set.
        
        Args:
            cp_set: Original conformal prediction set
            refined_set: Refined set after applying lambda rules
        
        Returns:
            Dictionary mapping node -> status ('accepted' or 'rejected')
        """
        status = {}
        for node in cp_set:
            status[node] = 'accepted' if node in refined_set else 'rejected'
        return status


def evaluate_lambda_config(lambda_config: Tuple[float, float, float],
                           probs: np.ndarray, ground_truths: np.ndarray,
                           cp_set: Set[int], G: nx.Graph,
                           infected_nodes: Optional[Set[int]] = None,
                           gamma: float = 1.0) -> Tuple[float, float, int]:
    """
    Evaluate a lambda configuration on a single sample.
    
    Args:
        lambda_config: (λ₁, λ₂, λ₃) tuple
        probs: Array of P(source) for each node
        ground_truths: One-hot array of true sources
        cp_set: Original conformal prediction set
        G: NetworkX graph
        infected_nodes: Optional set of infected nodes
        gamma: Distance decay parameter
    
    Returns:
        Tuple of (recall, precision, set_size)
    """
    lambda1, lambda2, lambda3 = lambda_config
    rules = LambdaRules(lambda1, lambda2, lambda3, gamma)
    
    refined_set = rules.refine_cp_set(cp_set, probs, G, infected_nodes)
    
    # Compute metrics
    true_sources = set(np.nonzero(ground_truths)[0])
    
    if len(true_sources) == 0:
        recall = 1.0
    else:
        recall = len(refined_set & true_sources) / len(true_sources)
    
    if len(refined_set) == 0:
        precision = 0.0
    else:
        precision = len(refined_set & true_sources) / len(refined_set)
    
    set_size = len(refined_set)
    
    return recall, precision, set_size


def generate_lambda_grid(n_points: int = 10) -> List[Tuple[float, float, float]]:
    """
    Generate a grid of lambda configurations for search.
    
    Args:
        n_points: Number of points per dimension
    
    Returns:
        List of (λ₁, λ₂, λ₃) tuples
    """
    lambda1_values = np.linspace(0.1, 1.0, n_points)
    lambda2_values = np.linspace(0.01, 0.5, n_points)
    lambda3_values = np.linspace(0.5, 2.0, n_points)
    
    grid = []
    for l1 in lambda1_values:
        for l2 in lambda2_values:
            for l3 in lambda3_values:
                grid.append((l1, l2, l3))
    
    return grid


if __name__ == '__main__':
    # Example usage
    print("Lambda Rules for CPNet-CLM")
    print("=" * 50)
    
    # Create a simple test graph
    G = nx.karate_club_graph()
    n_nodes = G.number_of_nodes()
    
    # Generate random probabilities
    np.random.seed(42)
    probs = np.random.rand(n_nodes)
    probs = probs / probs.sum()  # Normalize
    
    # Create a sample CP set (top 10 nodes)
    top_nodes = np.argsort(-probs)[:10]
    cp_set = set(top_nodes)
    
    # Ground truth (random 3 sources)
    gt_sources = np.random.choice(n_nodes, 3, replace=False)
    ground_truths = np.zeros(n_nodes)
    ground_truths[gt_sources] = 1
    
    print(f"Original CP set size: {len(cp_set)}")
    print(f"Ground truth sources: {gt_sources}")
    
    # Apply lambda rules
    rules = LambdaRules(lambda1=0.5, lambda2=0.02, lambda3=0.8, gamma=0.5)
    refined_set = rules.refine_cp_set(cp_set, probs, G)
    
    print(f"Refined set size: {len(refined_set)}")
    print(f"Rejected nodes: {cp_set - refined_set}")
    
    # Evaluate
    recall, precision, size = evaluate_lambda_config(
        (0.5, 0.02, 0.8), probs, ground_truths, cp_set, G
    )
    print(f"Recall: {recall:.3f}, Precision: {precision:.3f}, Size: {size}")
