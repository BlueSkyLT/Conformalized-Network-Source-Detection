#!/usr/bin/env python3
"""
Plot histogram of pairwise shortest-path distances between ground truth source nodes.

Given a dataset (res.pickle) and a graph, this script:
1. Loads ground truth source nodes for each sample
2. Computes all pairwise shortest-path distances between sources using networkx
3. Plots a histogram of these distances

Usage:
    python clm/plot_gt_distances.py --graph highSchool --res_path path/to/res.pickle
    python clm/plot_gt_distances.py --graph highSchool --res_path path/to/res.pickle --sample_index 0
    python clm/plot_gt_distances.py --graph highSchool --res_path path/to/res.pickle --aggregate
"""

import os
import sys
import argparse
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


def load_graph(graph_name):
    """Load the graph from edgelist file."""
    # Try multiple possible locations
    paths_to_try = [
        f'SD-STGCN/dataset/{graph_name}/data/graph/{graph_name}.edgelist',
        f'data/{graph_name}/graph/{graph_name}.edgelist',
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            G = nx.read_edgelist(path, nodetype=int)
            print(f"Loaded graph from {path}")
            print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            return G
    
    raise FileNotFoundError(f"Could not find graph for '{graph_name}'. Tried: {paths_to_try}")


def load_res_pickle(res_path):
    """Load and flatten the res.pickle data."""
    with open(res_path, 'rb') as f:
        data = pickle.load(f)
    
    ground_truths = []
    for batch_idx in range(len(data['ground_truth'])):
        batch_gt = data['ground_truth'][batch_idx]
        for sample_idx in range(len(batch_gt)):
            ground_truths.append(batch_gt[sample_idx])
    
    print(f"Loaded {len(ground_truths)} samples from {res_path}")
    return ground_truths


def get_source_nodes(ground_truth):
    """Extract source node indices from one-hot ground truth array."""
    return np.nonzero(ground_truth)[0]


def compute_pairwise_distances(G, source_nodes):
    """
    Compute all pairwise shortest-path distances between source nodes.
    
    Returns a list of distances for all C(n,2) pairs where n = len(source_nodes).
    """
    distances = []
    source_list = list(source_nodes)
    
    for i, j in combinations(source_list, 2):
        try:
            dist = nx.shortest_path_length(G, source=i, target=j)
            distances.append(dist)
        except nx.NetworkXNoPath:
            # If no path exists, use infinity or skip
            distances.append(float('inf'))
    
    return distances


def plot_distance_histogram(distances, title, output_path=None, bins='auto'):
    """Plot histogram of distances."""
    # Filter out infinite distances
    finite_distances = [d for d in distances if d != float('inf')]
    
    if len(finite_distances) == 0:
        print("Warning: No finite distances to plot!")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Determine bins
    if bins == 'auto':
        max_dist = max(finite_distances)
        bins = range(0, int(max_dist) + 2)
    
    plt.hist(finite_distances, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Shortest-Path Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', alpha=0.5)
    
    # Add statistics
    mean_dist = np.mean(finite_distances)
    median_dist = np.median(finite_distances)
    plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f}')
    plt.axvline(median_dist, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_dist:.2f}')
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot histogram of pairwise distances between GT source nodes')
    parser.add_argument('--graph', type=str, required=True, help='Graph name (e.g., highSchool)')
    parser.add_argument('--res_path', type=str, required=True, help='Path to res.pickle file')
    parser.add_argument('--sample_index', type=int, default=None, 
                        help='Specific sample index to analyze (default: first sample with multiple sources)')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate distances across all samples')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (default: display)')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to process when aggregating (default: 100)')
    
    args = parser.parse_args()
    
    # Load graph and data
    G = load_graph(args.graph)
    ground_truths = load_res_pickle(args.res_path)
    
    if args.aggregate:
        # Aggregate distances across multiple samples
        all_distances = []
        n_samples_used = 0
        
        for i, gt in enumerate(ground_truths[:args.max_samples]):
            sources = get_source_nodes(gt)
            if len(sources) >= 2:
                distances = compute_pairwise_distances(G, sources)
                all_distances.extend(distances)
                n_samples_used += 1
        
        if len(all_distances) == 0:
            print("No samples with multiple sources found!")
            return
        
        print(f"\nAggregated {len(all_distances)} pairwise distances from {n_samples_used} samples")
        title = f'Histogram of GT Source Pairwise Distances\n({args.graph}, {n_samples_used} samples)'
        plot_distance_histogram(all_distances, title, args.output)
        
    else:
        # Analyze a single sample
        if args.sample_index is not None:
            idx = args.sample_index
            if idx >= len(ground_truths):
                print(f"Error: sample_index {idx} out of range (max: {len(ground_truths)-1})")
                return
        else:
            # Find first sample with multiple sources
            idx = None
            for i, gt in enumerate(ground_truths):
                sources = get_source_nodes(gt)
                if len(sources) >= 2:
                    idx = i
                    break
            
            if idx is None:
                print("No samples with multiple sources found!")
                return
        
        sources = get_source_nodes(ground_truths[idx])
        n_sources = len(sources)
        n_pairs = n_sources * (n_sources - 1) // 2  # C(n,2) = n!/(2!*(n-2)!)
        
        print(f"\nSample {idx}:")
        print(f"  Number of sources: {n_sources}")
        print(f"  Source nodes: {sources}")
        print(f"  Number of pairs: {n_pairs}")
        
        if n_sources < 2:
            print("  Error: Need at least 2 sources to compute pairwise distances!")
            return
        
        distances = compute_pairwise_distances(G, sources)
        
        print(f"  Distances: {distances}")
        print(f"  Min distance: {min(distances)}")
        print(f"  Max distance: {max(distances)}")
        print(f"  Mean distance: {np.mean(distances):.2f}")
        
        title = f'Histogram of GT Source Pairwise Distances\n({args.graph}, Sample {idx}, {n_sources} sources, {n_pairs} pairs)'
        plot_distance_histogram(distances, title, args.output)


if __name__ == '__main__':
    main()
