#!/usr/bin/env python3
"""
Plot bar chart of SD-STGCN source probabilities for all nodes.

Given a dataset (res.pickle), this script:
1. Loads the model predictions for a specific sample
2. Plots a bar chart with nodes on x-axis and P(source) on y-axis
3. Highlights ground truth source nodes
4. Optionally shows CP set membership and CLM rejection status

Usage:
    # Basic usage
    python clm/plot_sdstgcn_probs.py --res_path path/to/res.pickle --sample_index 0
    
    # Show top 100 nodes with CP set and CLM rejection
    python clm/plot_sdstgcn_probs.py --res_path path/to/res.pickle --sample_index 0 --top_k 100 --clm_results path/to/clm_results.pickle
    
    # Save output
    python clm/plot_sdstgcn_probs.py --res_path path/to/res.pickle --sample_index 0 --output plot.png
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_res_pickle(res_path):
    """Load and flatten the res.pickle data."""
    with open(res_path, 'rb') as f:
        data = pickle.load(f)
    
    predictions = []
    ground_truths = []
    inputs = []
    
    for batch_idx in range(len(data['predictions'])):
        batch_preds = data['predictions'][batch_idx]
        batch_gt = data['ground_truth'][batch_idx]
        batch_inputs = data['inputs'][batch_idx]
        
        for sample_idx in range(len(batch_preds)):
            predictions.append(batch_preds[sample_idx])
            ground_truths.append(batch_gt[sample_idx])
            inputs.append(batch_inputs[sample_idx])
    
    print(f"Loaded {len(predictions)} samples from {res_path}")
    return predictions, ground_truths, inputs


def load_clm_results(clm_path):
    """Load CLM results including CP sets and refined sets."""
    with open(clm_path, 'rb') as f:
        return pickle.load(f)


def load_graph(graph_name):
    """Load graph for CLM analysis."""
    paths = [
        f'SD-STGCN/dataset/{graph_name}/data/graph/{graph_name}.edgelist',
        f'data/{graph_name}/graph/{graph_name}.edgelist',
    ]
    for path in paths:
        if os.path.exists(path):
            return nx.read_edgelist(path, nodetype=int)
    return None


def get_source_probs(prediction):
    """Extract P(source) for each node from prediction array.
    
    prediction shape: (n_nodes, 2) where [:, 1] is P(source)
    """
    return prediction[:, 1]


def get_source_nodes(ground_truth):
    """Extract source node indices from one-hot ground truth array."""
    return np.nonzero(ground_truth)[0]


def compute_cp_set_simple(probs, threshold=0.1):
    """Compute a simple CP set based on probability threshold."""
    return set(np.where(probs >= threshold)[0])


def plot_probability_barchart_enhanced(probs, source_nodes, title, output_path=None, 
                                        top_k=100, sort_by_prob=True,
                                        cp_set=None, refined_set=None):
    """
    Enhanced bar chart showing probabilities with CP set and CLM rejection status.
    
    Color coding:
    - Green: Ground truth source nodes
    - Blue: Non-source nodes in refined set (accepted by CLM)
    - Orange: Nodes in original CP set but rejected by CLM
    - Gray: Nodes not in CP set
    
    Args:
        probs: array of P(source) for each node
        source_nodes: array of ground truth source node indices
        title: plot title
        output_path: if provided, save to this path
        top_k: show top K nodes by probability (default: 100)
        sort_by_prob: if True, sort nodes by probability (descending)
        cp_set: optional set of nodes in original CP prediction set
        refined_set: optional set of nodes after CLM refinement
    """
    n_nodes = len(probs)
    nodes = np.arange(n_nodes)
    
    # Prepare data for plotting
    if sort_by_prob:
        sorted_indices = np.argsort(-probs)  # Descending order
        nodes = sorted_indices
        probs_to_plot = probs[sorted_indices]
    else:
        probs_to_plot = probs
    
    # Limit to top_k if specified
    if top_k is not None:
        nodes = nodes[:top_k]
        probs_to_plot = probs_to_plot[:top_k]
    
    # Create color array based on node status
    source_set = set(source_nodes)
    cp_set = cp_set or set()
    refined_set = refined_set or set()
    
    colors = []
    for n in nodes:
        if n in source_set:
            colors.append('green')  # GT source
        elif n in refined_set:
            colors.append('steelblue')  # In refined set (accepted)
        elif n in cp_set:
            colors.append('orange')  # In CP set but rejected by CLM
        else:
            colors.append('lightgray')  # Not in CP set
    
    # Determine figure size based on number of nodes
    width = max(12, min(24, len(nodes) * 0.15))
    fig, ax = plt.subplots(figsize=(width, 7))
    
    # Plot bars
    x_positions = np.arange(len(nodes))
    print(f"Plotting {len(nodes)} nodes (top_k={top_k}, sort_by_prob={sort_by_prob})")
    
    # Add edge colors for better distinction
    edge_colors = []
    edge_widths = []
    for n in nodes:
        if n in source_set:
            edge_colors.append('darkgreen')
            edge_widths.append(1.5)
        elif n in cp_set:
            edge_colors.append('black')
            edge_widths.append(0.8)
        else:
            edge_colors.append('gray')
            edge_widths.append(0.3)
    
    bars = ax.bar(x_positions, probs_to_plot, color=colors, 
                  edgecolor=edge_colors, linewidth=edge_widths, alpha=0.85)
    
    # Customize x-axis
    if len(nodes) <= 50:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(nodes, rotation=90, fontsize=8)
        ax.set_xlabel('Node ID', fontsize=12)
    else:
        # For many nodes, show fewer tick labels
        step = max(1, len(nodes) // 25)
        tick_positions = x_positions[::step]
        tick_labels = nodes[::step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_xlabel('Node ID (sorted by probability)', fontsize=12)
    
    ax.set_ylabel('P(Source)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', alpha=0.5)
    
    # Add legend with all categories
    legend_elements = [
        Patch(facecolor='green', edgecolor='darkgreen', linewidth=1.5, label='Ground Truth Source'),
    ]
    
    if len(cp_set) > 0 and len(refined_set) > 0:
        legend_elements.extend([
            Patch(facecolor='steelblue', edgecolor='black', label='CLM Accepted (in refined set)'),
            Patch(facecolor='orange', edgecolor='black', label='CLM Rejected (was in CP set)'),
            Patch(facecolor='lightgray', edgecolor='gray', label='Not in CP set'),
        ])
    else:
        legend_elements.append(
            Patch(facecolor='steelblue', edgecolor='black', label='Non-Source Node')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Add statistics text
    gt_probs = probs[source_nodes] if len(source_nodes) > 0 else []
    non_gt_probs = np.delete(probs, source_nodes) if len(source_nodes) > 0 else probs
    
    stats_text = f"GT Sources ({len(source_nodes)}): "
    if len(gt_probs) > 0:
        stats_text += f"mean={np.mean(gt_probs):.3f}, max={np.max(gt_probs):.3f}\n"
    else:
        stats_text += "N/A\n"
    stats_text += f"Non-Sources ({len(non_gt_probs)}): mean={np.mean(non_gt_probs):.3f}, max={np.max(non_gt_probs):.3f}"
    
    # Add CP/CLM stats if available
    if len(cp_set) > 0:
        stats_text += f"\n\nOriginal CP set: {len(cp_set)} nodes"
        if len(refined_set) > 0:
            rejected = cp_set - refined_set
            stats_text += f"\nCLM refined set: {len(refined_set)} nodes"
            stats_text += f"\nCLM rejected: {len(rejected)} nodes"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot bar chart of SD-STGCN source probabilities')
    parser.add_argument('--res_path', type=str, required=True, help='Path to res.pickle file')
    parser.add_argument('--sample_index', type=int, default=0, help='Sample index to visualize (default: 0)')
    parser.add_argument('--output', type=str, default=None, help='Output path for the plot (default: display)')
    parser.add_argument('--top_k', type=int, default=100, 
                        help='Show top K nodes by probability (default: 100)')
    parser.add_argument('--sort', action='store_true', default=True,
                        help='Sort nodes by probability (descending, default: True)')
    parser.add_argument('--no_sort', action='store_true',
                        help='Do not sort nodes (overrides --sort)')
    parser.add_argument('--all_nodes', action='store_true',
                        help='Show all nodes even if there are many')
    parser.add_argument('--clm_results', type=str, default=None,
                        help='Path to CLM results pickle (for CP set and rejection visualization)')
    parser.add_argument('--graph', type=str, default=None,
                        help='Graph name (for computing CP sets without CLM results)')
    
    args = parser.parse_args()
    
    # Load data
    predictions, ground_truths, inputs = load_res_pickle(args.res_path)
    
    if args.sample_index >= len(predictions):
        print(f"Error: sample_index {args.sample_index} out of range (max: {len(predictions)-1})")
        return
    
    # Get data for the specified sample
    probs = get_source_probs(predictions[args.sample_index])
    source_nodes = get_source_nodes(ground_truths[args.sample_index])
    input_state = inputs[args.sample_index]
    infected_nodes = set(np.nonzero(input_state)[0])
    
    n_nodes = len(probs)
    n_sources = len(source_nodes)
    
    print(f"\nSample {args.sample_index}:")
    print(f"  Total nodes: {n_nodes}")
    print(f"  Number of sources: {n_sources}")
    print(f"  Source nodes: {source_nodes}")
    print(f"  Infected nodes: {len(infected_nodes)}")
    
    # Print probability statistics for source nodes
    if n_sources > 0:
        source_probs = probs[source_nodes]
        print(f"\n  Source node probabilities:")
        for node, prob in zip(source_nodes, source_probs):
            print(f"    Node {node}: P(source) = {prob:.4f}")
        print(f"  Mean GT probability: {np.mean(source_probs):.4f}")
        print(f"  Min GT probability: {np.min(source_probs):.4f}")
        print(f"  Max GT probability: {np.max(source_probs):.4f}")
    
    # Print overall statistics
    print(f"\n  Overall probability statistics:")
    print(f"    Mean: {np.mean(probs):.4f}")
    print(f"    Std: {np.std(probs):.4f}")
    print(f"    Max: {np.max(probs):.4f} (node {np.argmax(probs)})")
    print(f"    Min: {np.min(probs):.4f}")
    
    # Load CLM results if provided
    cp_set = set()
    refined_set = set()
    
    if args.clm_results and os.path.exists(args.clm_results):
        print(f"\n  Loading CLM results from {args.clm_results}")
        clm_results = load_clm_results(args.clm_results)
        
        # Find the matching sample in CLM results
        test_indices = clm_results.get('test_indices', [])
        if args.sample_index in test_indices:
            idx_in_test = list(test_indices).index(args.sample_index)
            cp_set = clm_results['original_cp_sets'][idx_in_test]
            refined_set = clm_results['refined_sets'][idx_in_test]
            print(f"    Found sample in CLM results (test index {idx_in_test})")
            print(f"    Original CP set size: {len(cp_set)}")
            print(f"    Refined set size: {len(refined_set)}")
        else:
            print(f"    Warning: Sample {args.sample_index} not in CLM test indices")
    elif args.graph:
        # Compute simple CP set based on probability threshold
        print(f"\n  Computing simple CP set (top probability nodes)")
        threshold = np.percentile(probs, 80)  # Top 20%
        cp_set = set(np.where(probs >= threshold)[0])
        # Restrict to infected nodes
        cp_set = cp_set & infected_nodes
        refined_set = cp_set  # No CLM refinement without full CLM results
        print(f"    CP set size (top 20% infected): {len(cp_set)}")
    
    # Determine top_k
    top_k = args.top_k
    if args.all_nodes:
        top_k = None
    
    # Determine sorting
    sort_by_prob = args.sort and not args.no_sort
    
    # Generate title
    title = f'SD-STGCN Source Probabilities\n(Sample {args.sample_index}, {n_nodes} nodes, {n_sources} GT sources)'
    if top_k:
        title += f'\n(Top {top_k} by probability)'
    
    # Plot with enhanced function
    plot_probability_barchart_enhanced(
        probs, 
        source_nodes, 
        title, 
        output_path=args.output,
        top_k=top_k,
        sort_by_prob=sort_by_prob,
        cp_set=cp_set,
        refined_set=refined_set
    )


if __name__ == '__main__':
    main()
