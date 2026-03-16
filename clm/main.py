#!/usr/bin/env python3
"""
Main script for CPNet-CLM (Conformal Language Modeling applied to Network Source Detection)

This script runs the full CLM-enhanced conformal prediction process:
1. Load SD-STGCN predictions and graph data
2. Compute original conformal prediction sets
3. Calibrate lambda parameters using LTT framework
4. Apply lambda rules to refine prediction sets
5. Evaluate and save results

Usage:
    python clm/main.py --graph highSchool --exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16
    python clm/main.py --graph highSchool --exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16 --alpha 0.1
"""

import os
import sys
import argparse
import pickle
import numpy as np
import networkx as nx
from typing import List, Set, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Handle imports for both direct execution and module import
try:
    from lambda_rules import LambdaRules, evaluate_lambda_config
    from ltt import LTTCalibrator, calibrate_ltt
except ImportError:
    from clm.lambda_rules import LambdaRules, evaluate_lambda_config
    from clm.ltt import LTTCalibrator, calibrate_ltt

from utils.score_convert import recall_score, recall_score_gtunknown
from utils.functions import cpquantile


def load_graph(graph_name: str) -> nx.Graph:
    """Load graph from edgelist file."""
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
    
    raise FileNotFoundError(f"Graph not found for '{graph_name}'")


def load_res_pickle(res_path: str) -> Tuple[List, List, List, List]:
    """Load and flatten res.pickle data."""
    with open(res_path, 'rb') as f:
        data = pickle.load(f)
    
    inputs, predictions, ground_truths, logits = [], [], [], []
    
    for batch_idx in range(len(data['predictions'])):
        batch_inputs = data['inputs'][batch_idx]
        batch_preds = data['predictions'][batch_idx]
        batch_gt = data['ground_truth'][batch_idx]
        batch_logits = data['logits'][batch_idx]
        
        for sample_idx in range(len(batch_preds)):
            inputs.append(batch_inputs[sample_idx])
            predictions.append(batch_preds[sample_idx])
            ground_truths.append(batch_gt[sample_idx])
            logits.append(batch_logits[sample_idx])
    
    print(f"Loaded {len(predictions)} samples from {res_path}")
    return inputs, predictions, ground_truths, logits


def compute_original_cp_sets(predictions: List[np.ndarray],
                              ground_truths: List[np.ndarray],
                              inputs: List[np.ndarray],
                              calib_indices: np.ndarray,
                              test_indices: np.ndarray,
                              alpha: float = 0.1,
                              prop_model: str = 'SI',
                              pow_expected: float = 0.5) -> Tuple[List[Set[int]], List[Set[int]]]:
    """
    Compute original conformal prediction sets using recall score.
    
    Args:
        predictions: List of prediction arrays
        ground_truths: List of ground truth arrays
        inputs: List of input arrays (infected states)
        calib_indices: Indices of calibration samples
        test_indices: Indices of test samples
        alpha: Significance level
        prop_model: Propagation model ('SI' or 'SIR')
        pow_expected: Power expected for set truncation
    
    Returns:
        Tuple of (calib_cp_sets, test_cp_sets)
    """
    n_nodes = predictions[0].shape[0]
    n_calib = len(calib_indices)
    n_test = len(test_indices)
    
    # Compute calibration scores
    from utils.score_convert import set_truncate
    cfscore_calib = []
    for i in calib_indices:
        infected_nodes = np.nonzero(inputs[i])[0]
        pred_prob = predictions[i][:, 1]
        gt_one_hot = ground_truths[i]
        gt_part_one_hot = set_truncate(gt_one_hot, pred_prob, pow_expected)
        score = recall_score(pred_prob, gt_part_one_hot, prop_model, infected_nodes)
        cfscore_calib.append(score)
    cfscore_calib = np.array(cfscore_calib)
    
    # Compute test scores
    cfscore_test = []
    for i in test_indices:
        infected_nodes = np.nonzero(inputs[i])[0]
        cfscore = recall_score_gtunknown(predictions[i][:, 1], prop_model, infected_nodes)
        cfscore_test.append(cfscore)
    cfscore_test = np.array(cfscore_test)
    
    # Compute threshold
    tail_prop = (1 - alpha) * (1 + 1 / n_calib)
    threshold = cpquantile(cfscore_calib, tail_prop)
    
    # Compute prediction sets
    test_cp_sets = []
    for i in range(n_test):
        pred_set = set()
        for j in range(n_nodes):
            if cfscore_test[i][j] <= threshold:
                pred_set.add(j)
        test_cp_sets.append(pred_set)
    
    # Also compute "CP sets" for calibration samples (for LTT calibration)
    calib_cp_sets = []
    for idx, i in enumerate(calib_indices):
        infected_nodes = np.nonzero(inputs[i])[0]
        pred_prob = predictions[i][:, 1]
        
        # Use scores to determine set
        cfscore = recall_score_gtunknown(pred_prob, prop_model, infected_nodes)
        pred_set = set()
        for j in range(n_nodes):
            if cfscore[j] <= threshold:
                pred_set.add(j)
        calib_cp_sets.append(pred_set)
    
    return calib_cp_sets, test_cp_sets


def evaluate_results(test_cp_sets: List[Set[int]],
                     refined_sets: List[Set[int]],
                     ground_truths: List[np.ndarray],
                     test_indices: np.ndarray) -> Dict:
    """
    Evaluate original and refined prediction sets.
    
    Args:
        test_cp_sets: Original CP sets
        refined_sets: Refined sets after CLM
        ground_truths: Ground truth arrays
        test_indices: Test sample indices
    
    Returns:
        Dictionary with evaluation metrics
    """
    n_test = len(test_indices)
    
    # Original metrics
    orig_recalls = []
    orig_sizes = []
    
    # Refined metrics
    ref_recalls = []
    ref_sizes = []
    
    for i, idx in enumerate(test_indices):
        gt_sources = set(np.nonzero(ground_truths[idx])[0])
        
        # Original
        orig_set = test_cp_sets[i]
        orig_recall = len(orig_set & gt_sources) / len(gt_sources) if len(gt_sources) > 0 else 1.0
        orig_recalls.append(orig_recall)
        orig_sizes.append(len(orig_set))
        
        # Refined
        ref_set = refined_sets[i]
        ref_recall = len(ref_set & gt_sources) / len(gt_sources) if len(gt_sources) > 0 else 1.0
        ref_recalls.append(ref_recall)
        ref_sizes.append(len(ref_set))
    
    results = {
        'original': {
            'coverage': np.mean(np.array(orig_recalls) == 1.0),
            'mean_recall': np.mean(orig_recalls),
            'mean_size': np.mean(orig_sizes),
            'std_size': np.std(orig_sizes)
        },
        'refined': {
            'coverage': np.mean(np.array(ref_recalls) == 1.0),
            'mean_recall': np.mean(ref_recalls),
            'mean_size': np.mean(ref_sizes),
            'std_size': np.std(ref_sizes)
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CPNet-CLM: CLM-enhanced Conformal Prediction')
    parser.add_argument('--graph', type=str, default='highSchool', help='Graph name')
    parser.add_argument('--exp_name', type=str, 
                        default='SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16',
                        help='Experiment name (res.pickle folder)')
    parser.add_argument('--alpha', type=float, default=0.1, help='CP significance level')
    parser.add_argument('--calib_ratio', type=float, default=0.5, 
                        help='Calibration data ratio')
    parser.add_argument('--ltt_alpha', type=float, default=0.1, help='LTT recall tolerance')
    parser.add_argument('--ltt_delta', type=float, default=0.1, help='LTT failure probability')
    parser.add_argument('--gamma', type=float, default=1.0, help='Distance decay parameter')
    parser.add_argument('--n_grid', type=int, default=5, help='Lambda grid points per dimension')
    parser.add_argument('--output_dir', type=str, default='clm/output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("CPNet-CLM: Conformal Language Modeling for Source Detection")
    print("=" * 60)
    
    # Create output directory
    output_path = os.path.join(args.output_dir, args.graph, args.exp_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    G = load_graph(args.graph)
    n_nodes = G.number_of_nodes()
    
    res_path = f'SD-STGCN/output/test_res/{args.graph}/{args.exp_name}/res.pickle'
    inputs, predictions, ground_truths, logits = load_res_pickle(res_path)
    
    # Split data
    n_samples = len(predictions)
    n_calib = int(n_samples * args.calib_ratio)
    
    indices = np.random.permutation(n_samples)
    calib_indices = indices[:n_calib]
    test_indices = indices[n_calib:]
    
    print(f"  Total samples: {n_samples}")
    print(f"  Calibration: {n_calib}, Test: {len(test_indices)}")
    
    # Compute original CP sets
    print("\n2. Computing original CP sets...")
    calib_cp_sets, test_cp_sets = compute_original_cp_sets(
        predictions, ground_truths, inputs,
        calib_indices, test_indices,
        alpha=args.alpha
    )
    print(f"  Average original CP set size: {np.mean([len(s) for s in test_cp_sets]):.2f}")
    
    # Prepare calibration data for LTT
    print("\n3. Running LTT calibration...")
    ltt_data = {
        'probs': [predictions[i][:, 1] for i in calib_indices],
        'ground_truths': [ground_truths[i] for i in calib_indices],
        'cp_sets': calib_cp_sets,
        'infected_nodes': [set(np.nonzero(inputs[i])[0]) for i in calib_indices]
    }
    
    best_lambda, valid_lambdas = calibrate_ltt(
        ltt_data, G,
        alpha=args.ltt_alpha,
        delta=args.ltt_delta,
        epsilon=args.ltt_alpha,
        gamma=args.gamma,
        n_grid_points=args.n_grid,
        verbose=True
    )
    
    # Apply lambda rules to test sets
    print("\n4. Applying lambda rules to test sets...")
    
    if best_lambda is not None:
        rules = LambdaRules(
            lambda1=best_lambda[0],
            lambda2=best_lambda[1],
            lambda3=best_lambda[2],
            gamma=args.gamma
        )
    else:
        print("  Warning: No valid lambda found. Using defaults.")
        rules = LambdaRules(lambda1=0.5, lambda2=0.05, lambda3=1.0, gamma=args.gamma)
        best_lambda = (0.5, 0.05, 1.0)
    
    refined_sets = []
    for i, idx in enumerate(test_indices):
        probs = predictions[idx][:, 1]
        infected_nodes = set(np.nonzero(inputs[idx])[0])
        
        refined_set = rules.refine_cp_set(
            test_cp_sets[i], probs, G, infected_nodes
        )
        refined_sets.append(refined_set)
    
    print(f"  Average refined set size: {np.mean([len(s) for s in refined_sets]):.2f}")
    
    # Evaluate
    print("\n5. Evaluating results...")
    results = evaluate_results(test_cp_sets, refined_sets, ground_truths, test_indices)
    
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"\nOriginal CP:")
    print(f"  Coverage: {results['original']['coverage']:.3f}")
    print(f"  Mean Recall: {results['original']['mean_recall']:.3f}")
    print(f"  Mean Set Size: {results['original']['mean_size']:.2f} ± {results['original']['std_size']:.2f}")
    
    print(f"\nCLM-Refined:")
    print(f"  Coverage: {results['refined']['coverage']:.3f}")
    print(f"  Mean Recall: {results['refined']['mean_recall']:.3f}")
    print(f"  Mean Set Size: {results['refined']['mean_size']:.2f} ± {results['refined']['std_size']:.2f}")
    
    # Compute improvement
    size_reduction = (results['original']['mean_size'] - results['refined']['mean_size'])
    size_reduction_pct = 100 * size_reduction / results['original']['mean_size']
    print(f"\n  Size Reduction: {size_reduction:.2f} ({size_reduction_pct:.1f}%)")
    
    # Save results
    print("\n6. Saving results...")
    save_data = {
        'best_lambda': best_lambda,
        'valid_lambdas': valid_lambdas,
        'results': results,
        'test_indices': test_indices,
        'calib_indices': calib_indices,
        'original_cp_sets': test_cp_sets,
        'refined_sets': refined_sets,
        'params': vars(args)
    }
    
    save_file = os.path.join(output_path, 'clm_results.pickle')
    with open(save_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"  Saved to {save_file}")
    
    # Save valid lambdas as text
    lambda_file = os.path.join(output_path, 'lambda_valid.txt')
    with open(lambda_file, 'w') as f:
        f.write(f"Best Lambda: {best_lambda}\n")
        f.write(f"\nValid Lambdas ({len(valid_lambdas)} total):\n")
        for lam in valid_lambdas:
            f.write(f"  {lam}\n")
    print(f"  Saved lambdas to {lambda_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
