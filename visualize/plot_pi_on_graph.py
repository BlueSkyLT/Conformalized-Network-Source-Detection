#!/usr/bin/env python3
"""
Plot graph with nodes colored by SD-STGCN predicted source probabilities.

Given a graph and res.pickle file, this script:
1. Loads the graph structure
2. Loads SD-STGCN predictions for a specific sample
3. Visualizes the graph with:
   - Node fill color based on P(source) probability (Reds colormap: white=0 to red=1)
   - Red edge color for ground truth source nodes
   - Black edge color for non-source nodes
   - Uniform node sizes for better visibility

Designed for large graphs (700+ nodes) with adjustable figure size.
Supports two modes: full graph view and zoomed view focused on GT nodes.

Usage:
    # Basic usage - full graph
    python visualize/plot_pi_on_graph.py --graph highSchool \\
        --input SD-STGCN/output/test_res/highSchool/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16/res.pickle

    # Zoomed view focused on GT nodes and neighbors
    python visualize/plot_pi_on_graph.py --graph highSchool \\
        --input path/to/res.pickle --zoom_gt --output zoomed_graph.png

    # With custom figure size for large graphs
    python visualize/plot_pi_on_graph.py --graph highSchool \\
        --input path/to/res.pickle --figsize 30 30
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_graph(graph_name):
    """
    Load graph from edgelist file.
    
    Args:
        graph_name: Name of the graph (e.g., 'highSchool', 'bkFratB')
    
    Returns:
        NetworkX graph object
    """
    paths = [
        f'SD-STGCN/dataset/{graph_name}/data/graph/{graph_name}.edgelist',
        f'data/{graph_name}/graph/{graph_name}.edgelist',
    ]
    
    for path in paths:
        if os.path.exists(path):
            G = nx.read_edgelist(path, nodetype=int)
            print(f"Loaded graph from {path}")
            print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            return G
    
    raise FileNotFoundError(f"Graph not found for '{graph_name}'. Tried: {paths}")


def load_res_pickle(res_path):
    """
    Load and flatten the res.pickle data.
    
    Args:
        res_path: Path to res.pickle file
    
    Returns:
        Tuple of (predictions, ground_truths, inputs) lists
    """
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


def get_source_probs(prediction):
    """
    Extract P(source) for each node from prediction array.
    
    Args:
        prediction: Shape (n_nodes, 2) where [:, 1] is P(source)
    
    Returns:
        Array of probabilities
    """
    return prediction[:, 1]


def get_source_nodes(ground_truth):
    """
    Extract source node indices from one-hot ground truth array.
    
    Args:
        ground_truth: One-hot encoded ground truth array
    
    Returns:
        Array of source node indices
    """
    return np.nonzero(ground_truth)[0]


def plot_graph_with_probabilities(G, probs, source_nodes, title, output_path=None,
                                   figsize=(20, 20), layout='spring', 
                                   node_size=100, cmap='Reds',
                                   edge_alpha=0.6, edge_width=0.8, seed=42,
                                   zoom_gt=False, zoom_hops=2):
    """
    Plot graph with nodes colored by probability and GT sources highlighted.
    
    Args:
        G: NetworkX graph
        probs: Array of P(source) for each node
        source_nodes: Array of ground truth source node indices
        title: Plot title
        output_path: If provided, save to this path
        figsize: Figure size tuple (width, height)
        layout: Graph layout algorithm ('spring', 'kamada_kawai', 'spectral', 'circular')
        node_size: Uniform node size for all nodes
        cmap: Colormap for probability (default: Reds - white=low, red=high)
        edge_alpha: Edge transparency (default: 0.6 for better visibility)
        edge_width: Edge line width (default: 0.8)
        seed: Random seed for reproducible layouts
        zoom_gt: If True, focus view on GT nodes and their neighbors
        zoom_hops: Number of hops from GT nodes to include in zoomed view
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    source_set = set(source_nodes)
    
    # Count nodes with prob > 0
    node_list_all = sorted(G.nodes())
    node_probs_all = []
    for node in node_list_all:
        if node < len(probs):
            node_probs_all.append(probs[node])
        else:
            node_probs_all.append(0.0)
    node_probs_all = np.array(node_probs_all)
    n_positive_prob = np.sum(node_probs_all > 0)
    
    print(f"\nPlotting graph with {n_nodes} nodes and {n_edges} edges...")
    print(f"  Nodes with P(source) > 0: {n_positive_prob}")
    print(f"  Figure size: {figsize}")
    print(f"  Layout: {layout}")
    
    # If zooming to GT nodes, create subgraph
    if zoom_gt and len(source_nodes) > 0:
        print(f"  Zooming to GT nodes and {zoom_hops}-hop neighbors...")
        # Get all nodes within zoom_hops of any GT source
        nodes_to_include = set(source_nodes)
        for src in source_nodes:
            if src in G:
                for neighbor in nx.single_source_shortest_path_length(G, src, cutoff=zoom_hops):
                    nodes_to_include.add(neighbor)
        G = G.subgraph(nodes_to_include).copy()
        print(f"  Zoomed subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute layout with better spreading
    print("  Computing layout...")
    if layout == 'spring':
        # Use larger k for more spacing between nodes
        k = 3.0 / np.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 100 else 1.5
        iterations = 200  # More iterations for better convergence
        pos = nx.spring_layout(G, k=k, iterations=iterations, seed=seed, scale=2.0)
    elif layout == 'kamada_kawai':
        if G.number_of_nodes() > 500:
            print("  Warning: kamada_kawai may be slow for large graphs")
        pos = nx.kamada_kawai_layout(G, scale=2.0)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G, scale=2.0)
    elif layout == 'circular':
        pos = nx.circular_layout(G, scale=2.0)
    else:
        pos = nx.spring_layout(G, seed=seed, scale=2.0)
    
    # Create node list in sorted order for consistent indexing
    node_list = sorted(G.nodes())
    
    # Map probabilities to nodes
    node_probs = []
    for node in node_list:
        if node < len(probs):
            node_probs.append(probs[node])
        else:
            node_probs.append(0.0)  # Default for nodes not in prediction
    node_probs = np.array(node_probs)
    
    # Uniform node sizes
    node_sizes = np.full(len(node_list), node_size)
    
    # Compute edge colors for nodes (red for GT sources, black for others)
    node_edge_colors = []
    node_edge_widths = []
    for node in node_list:
        if node in source_set:
            node_edge_colors.append('red')
            node_edge_widths.append(3.0)  # Thicker edge for GT sources
        else:
            node_edge_colors.append('black')
            node_edge_widths.append(1.0)
    
    # Create colormap for node fill (Reds: white=0, red=1)
    colormap = plt.colormaps.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    node_colors = [colormap(norm(p)) for p in node_probs]
    
    # Draw edges first (so nodes are on top) - more visible now
    print("  Drawing edges...")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha, 
                           width=edge_width, edge_color='black')
    
    # Draw nodes
    print("  Drawing nodes...")
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors=node_edge_colors,
                           linewidths=node_edge_widths)
    
    # Add labels for source nodes only (to avoid clutter)
    if len(source_nodes) <= 20:  # Only label if not too many sources
        source_pos = {node: pos[node] for node in source_nodes if node in pos}
        nx.draw_networkx_labels(G, source_pos, ax=ax, 
                                labels={n: str(n) for n in source_nodes if n in pos},
                                font_size=8, font_weight='bold',
                                font_color='darkred')
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('P(Source)', fontsize=12)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
               markeredgecolor='red', markeredgewidth=3.0, markersize=12,
               label=f'GT Source ({len(source_nodes)} nodes)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='mistyrose', 
               markeredgecolor='black', markeredgewidth=1.0, markersize=10,
               label=f'Non-Source ({n_nodes - len(source_nodes)} nodes)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add statistics text including nodes with prob > 0
    gt_probs = node_probs_all[np.isin(node_list_all, source_nodes)]
    non_gt_mask = ~np.isin(node_list_all, source_nodes)
    non_gt_probs = node_probs_all[non_gt_mask]
    
    stats_text = f"Nodes with P > 0: {n_positive_prob} / {n_nodes}\n"
    stats_text += f"GT Sources: mean P={np.mean(gt_probs):.3f}, max={np.max(gt_probs):.3f}\n"
    stats_text += f"Non-Sources: mean P={np.mean(non_gt_probs):.3f}, max={np.max(non_gt_probs):.3f}"
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        print(f"  Saving to {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"  Saved!")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot graph with nodes colored by SD-STGCN source probabilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with highSchool graph (full graph view)
    python visualize/plot_pi_on_graph.py --graph highSchool \\
        --input SD-STGCN/output/test_res/highSchool/SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16/res.pickle

    # Zoomed view focused on GT nodes and their 2-hop neighbors
    python visualize/plot_pi_on_graph.py --graph highSchool --input path/to/res.pickle \\
        --zoom_gt --output zoomed_graph.png

    # Save both full and zoomed views
    python visualize/plot_pi_on_graph.py --graph highSchool --input path/to/res.pickle \\
        --output full_graph.png
    python visualize/plot_pi_on_graph.py --graph highSchool --input path/to/res.pickle \\
        --zoom_gt --output zoomed_graph.png --figsize 15 15
        """
    )
    
    parser.add_argument('--graph', type=str, required=True,
                        help='Graph name (e.g., highSchool, bkFratB)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to res.pickle file from SD-STGCN')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='Sample index to visualize (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (default: display)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[25, 25],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches (default: 25 25)')
    parser.add_argument('--layout', type=str, default='spring',
                        choices=['spring', 'kamada_kawai', 'spectral', 'circular'],
                        help='Graph layout algorithm (default: spring)')
    parser.add_argument('--node_size', type=int, default=100,
                        help='Uniform node size (default: 100)')
    parser.add_argument('--cmap', type=str, default='Reds',
                        help='Colormap for probabilities (default: Reds)')
    parser.add_argument('--edge_alpha', type=float, default=0.6,
                        help='Edge transparency 0-1 (default: 0.6)')
    parser.add_argument('--edge_width', type=float, default=0.8,
                        help='Edge line width (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for layout (default: 42)')
    parser.add_argument('--zoom_gt', action='store_true',
                        help='Zoom to GT nodes and their neighbors')
    parser.add_argument('--zoom_hops', type=int, default=2,
                        help='Number of hops from GT nodes in zoomed view (default: 2)')
    
    args = parser.parse_args()
    
    # Load graph
    print(f"\nLoading graph: {args.graph}")
    G = load_graph(args.graph)
    
    # Load predictions
    print(f"\nLoading predictions from: {args.input}")
    predictions, ground_truths, inputs = load_res_pickle(args.input)
    
    if args.sample_index >= len(predictions):
        print(f"Error: sample_index {args.sample_index} out of range (max: {len(predictions)-1})")
        return
    
    # Get data for the specified sample
    probs = get_source_probs(predictions[args.sample_index])
    source_nodes = get_source_nodes(ground_truths[args.sample_index])
    
    # Count nodes with prob > 0
    n_positive_prob = np.sum(probs > 0)
    
    print(f"\nSample {args.sample_index}:")
    print(f"  Prediction array shape: {predictions[args.sample_index].shape}")
    print(f"  Number of GT sources: {len(source_nodes)}")
    print(f"  GT source nodes: {source_nodes}")
    print(f"  Nodes with P(source) > 0: {n_positive_prob}")
    
    # Print probability statistics for source nodes
    if len(source_nodes) > 0:
        source_probs = probs[source_nodes]
        print(f"\n  Source node probabilities:")
        for node, prob in sorted(zip(source_nodes, source_probs), key=lambda x: -x[1]):
            print(f"    Node {node}: P(source) = {prob:.4f}")
    
    # Generate title
    view_type = "Zoomed (GT + neighbors)" if args.zoom_gt else "Full Graph"
    title = f'{args.graph} Graph - SD-STGCN Source Probabilities ({view_type})\n'
    title += f'(Sample {args.sample_index}, {G.number_of_nodes()} nodes, '
    title += f'{G.number_of_edges()} edges, {len(source_nodes)} GT sources)'
    
    # Plot
    plot_graph_with_probabilities(
        G=G,
        probs=probs,
        source_nodes=source_nodes,
        title=title,
        output_path=args.output,
        figsize=tuple(args.figsize),
        layout=args.layout,
        node_size=args.node_size,
        cmap=args.cmap,
        edge_alpha=args.edge_alpha,
        edge_width=args.edge_width,
        seed=args.seed,
        zoom_gt=args.zoom_gt,
        zoom_hops=args.zoom_hops
    )


if __name__ == '__main__':
    main()
