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
import logging
from datetime import datetime
from typing import List, Set, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Handle imports for both direct execution and module import
try:
    from lambda_rules import LambdaRules, evaluate_lambda_config
    from ltt import LTTCalibrator, calibrate_ltt
    from utils import recall_score, recall_score_gtunknown, cpquantile, set_truncate
except ImportError:
    from clm.lambda_rules import LambdaRules, evaluate_lambda_config
    from clm.ltt import LTTCalibrator, calibrate_ltt
    from clm.utils import recall_score, recall_score_gtunknown, cpquantile, set_truncate


def setup_logging(output_path: str, verbose: bool = True) -> logging.Logger:
    """
    Setup logging to both file and console.
    
    Args:
        output_path: Directory for log files
        verbose: If True, also log to console
    
    Returns:
        Logger instance
    """
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_path, f'clm_run_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('CLM')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler (info level)
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    
    logger.info(f"Logging to {log_file}")
    return logger


def analyze_sample_detailed(sample_idx: int, 
                           probs: np.ndarray,
                           ground_truths: np.ndarray,
                           orig_cp_set: Set[int],
                           refined_set: Set[int],
                           rules: 'LambdaRules',
                           G: nx.Graph,
                           infected_nodes: Set[int],
                           logger: logging.Logger) -> Dict:
    """
    Perform detailed analysis of a single sample.
    
    Tracks:
    - Which GT sources are falsely rejected
    - Which FPs are falsely included
    - Rejection reasons for each node
    
    Args:
        sample_idx: Sample index for logging
        probs: Probability array
        ground_truths: Ground truth array
        orig_cp_set: Original CP set
        refined_set: Refined set after CLM
        rules: LambdaRules instance
        G: NetworkX graph
        infected_nodes: Set of infected nodes
        logger: Logger instance
    
    Returns:
        Dictionary with detailed analysis
    """
    gt_sources = set(np.nonzero(ground_truths)[0])
    
    # Categorize nodes
    gt_in_cp = gt_sources & orig_cp_set
    gt_in_refined = gt_sources & refined_set
    gt_rejected = gt_in_cp - gt_in_refined  # GT sources falsely rejected
    
    fp_in_cp = orig_cp_set - gt_sources
    fp_in_refined = refined_set - gt_sources  # FPs falsely included
    fp_removed = fp_in_cp - refined_set  # FPs correctly removed
    
    # Detailed logging
    logger.debug(f"\n{'='*60}")
    logger.debug(f"Sample {sample_idx} Detailed Analysis")
    logger.debug(f"{'='*60}")
    logger.debug(f"Ground Truth sources: {sorted(gt_sources)} (n={len(gt_sources)})")
    logger.debug(f"GT probs: {[f'{probs[v]:.4f}' for v in sorted(gt_sources)]}")
    
    logger.debug(f"\nOriginal CP set size: {len(orig_cp_set)}")
    logger.debug(f"Refined set size: {len(refined_set)}")
    logger.debug(f"GT in original CP: {sorted(gt_in_cp)} (n={len(gt_in_cp)})")
    logger.debug(f"GT in refined: {sorted(gt_in_refined)} (n={len(gt_in_refined)})")
    
    # Analyze rejected GT sources
    if len(gt_rejected) > 0:
        logger.warning(f"\n⚠️  FALSELY REJECTED GT SOURCES: {sorted(gt_rejected)}")
        for node in sorted(gt_rejected):
            prob = probs[node]
            # Check which rule rejected it
            reason = analyze_rejection_reason(node, probs, rules, G, refined_set, infected_nodes)
            logger.warning(f"  Node {node}: prob={prob:.4f}, rejection_reason={reason}")
    
    # Analyze included FPs
    if len(fp_in_refined) > 0:
        logger.info(f"\n📌 False Positives STILL IN refined set: {sorted(fp_in_refined)} (n={len(fp_in_refined)})")
        for node in sorted(fp_in_refined)[:10]:  # Top 10 FPs
            prob = probs[node]
            logger.debug(f"  FP Node {node}: prob={prob:.4f}")
    
    # Log correctly removed FPs
    if len(fp_removed) > 0:
        logger.info(f"\n✓ False Positives correctly removed: n={len(fp_removed)}")
    
    # Calculate metrics
    orig_recall = len(gt_in_cp) / len(gt_sources) if len(gt_sources) > 0 else 1.0
    ref_recall = len(gt_in_refined) / len(gt_sources) if len(gt_sources) > 0 else 1.0
    orig_precision = len(gt_in_cp) / len(orig_cp_set) if len(orig_cp_set) > 0 else 0.0
    ref_precision = len(gt_in_refined) / len(refined_set) if len(refined_set) > 0 else 0.0
    
    logger.debug(f"\nMetrics:")
    logger.debug(f"  Original - Recall: {orig_recall:.3f}, Precision: {orig_precision:.3f}")
    logger.debug(f"  Refined  - Recall: {ref_recall:.3f}, Precision: {ref_precision:.3f}")
    
    return {
        'sample_idx': sample_idx,
        'n_gt': len(gt_sources),
        'n_gt_in_cp': len(gt_in_cp),
        'n_gt_in_refined': len(gt_in_refined),
        'gt_rejected': gt_rejected,
        'n_fp_in_cp': len(fp_in_cp),
        'n_fp_in_refined': len(fp_in_refined),
        'n_fp_removed': len(fp_removed),
        'orig_recall': orig_recall,
        'ref_recall': ref_recall,
        'orig_precision': orig_precision,
        'ref_precision': ref_precision
    }


def analyze_rejection_reason(node: int, probs: np.ndarray, rules: 'LambdaRules',
                             G: nx.Graph, current_selected: Set[int], 
                             infected_nodes: Set[int]) -> str:
    """
    Determine why a node was rejected.
    
    Returns:
        String describing rejection reason
    """
    reasons = []
    
    # Check quality rejection (λ₂)
    if probs[node] < rules.lambda2:
        reasons.append(f"QUALITY(prob={probs[node]:.4f} < λ₂={rules.lambda2:.4f})")
    
    # Check if not in infected nodes
    if infected_nodes and node not in infected_nodes:
        reasons.append("NOT_INFECTED")
    
    # Check diversity rejection (λ₁)
    if len(current_selected) > 0:
        max_score = 0.0
        closest_node = None
        for v_j in current_selected:
            try:
                dist = nx.shortest_path_length(G, source=node, target=v_j)
            except nx.NetworkXNoPath:
                dist = float('inf')
            kernel_weight = np.exp(-rules.gamma * dist)
            score = kernel_weight * probs[node]
            if score > max_score:
                max_score = score
                closest_node = v_j
        
        if max_score > rules.lambda1:
            reasons.append(f"DIVERSITY(score={max_score:.4f} > λ₁={rules.lambda1:.4f}, closest={closest_node})")
    
    # Check stopping rule (λ₃)
    # This is harder to check retroactively, but we can flag if cumsum might have stopped
    
    if len(reasons) == 0:
        reasons.append("STOPPING_RULE(λ₃)")
    
    return ", ".join(reasons)


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
    # Note: set_truncate is imported from clm.utils (no TensorFlow dependency)
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
    parser.add_argument('--verbose', action='store_true', default=True, 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Create output directory
    output_path = os.path.join(args.output_dir, args.graph, args.exp_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_path, verbose=args.verbose)
    
    logger.info("=" * 60)
    logger.info("CPNet-CLM: Conformal Language Modeling for Source Detection")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Graph: {args.graph}")
    logger.info(f"  Experiment: {args.exp_name}")
    logger.info(f"  Alpha (CP): {args.alpha}")
    logger.info(f"  LTT Alpha: {args.ltt_alpha}")
    logger.info(f"  LTT Delta: {args.ltt_delta}")
    logger.info(f"  Gamma: {args.gamma}")
    logger.info(f"  N_grid: {args.n_grid}")
    logger.info(f"  Seed: {args.seed}")
    
    # Load data
    logger.info("\n1. Loading data...")
    G = load_graph(args.graph)
    n_nodes = G.number_of_nodes()
    logger.debug(f"Graph nodes: {n_nodes}, edges: {G.number_of_edges()}")
    
    res_path = f'SD-STGCN/output/test_res/{args.graph}/{args.exp_name}/res.pickle'
    inputs, predictions, ground_truths, logits = load_res_pickle(res_path)
    
    # Split data
    n_samples = len(predictions)
    n_calib = int(n_samples * args.calib_ratio)
    
    indices = np.random.permutation(n_samples)
    calib_indices = indices[:n_calib]
    test_indices = indices[n_calib:]
    
    logger.info(f"  Total samples: {n_samples}")
    logger.info(f"  Calibration: {n_calib}, Test: {len(test_indices)}")
    
    # Compute original CP sets
    logger.info("\n2. Computing original CP sets...")
    calib_cp_sets, test_cp_sets = compute_original_cp_sets(
        predictions, ground_truths, inputs,
        calib_indices, test_indices,
        alpha=args.alpha
    )
    logger.info(f"  Average original CP set size: {np.mean([len(s) for s in test_cp_sets]):.2f}")
    
    # Analyze original CP coverage on test set
    logger.info("\n  Original CP Analysis on Test Set:")
    n_full_coverage = 0
    for i, idx in enumerate(test_indices):
        gt_sources = set(np.nonzero(ground_truths[idx])[0])
        cp_set = test_cp_sets[i]
        coverage = len(cp_set & gt_sources) / len(gt_sources) if len(gt_sources) > 0 else 1.0
        if coverage == 1.0:
            n_full_coverage += 1
        else:
            logger.debug(f"  Sample {i}: coverage={coverage:.2f}, missing GT: {gt_sources - cp_set}")
    logger.info(f"  Original CP full coverage rate: {n_full_coverage}/{len(test_indices)} ({100*n_full_coverage/len(test_indices):.1f}%)")
    
    # Prepare calibration data for LTT
    logger.info("\n3. Running LTT calibration...")
    ltt_data = {
        'probs': [predictions[i][:, 1] for i in calib_indices],
        'ground_truths': [ground_truths[i] for i in calib_indices],
        'cp_sets': calib_cp_sets,
        'infected_nodes': [set(np.nonzero(inputs[i])[0]) for i in calib_indices]
    }
    
    best_lambda, valid_lambdas, ltt_details = calibrate_ltt(
        ltt_data, G,
        alpha=args.ltt_alpha,
        delta=args.ltt_delta,
        epsilon=args.ltt_alpha,
        gamma=args.gamma,
        n_grid_points=args.n_grid,
        verbose=True,
        logger=logger
    )
    
    # Log LTT details
    logger.info(f"\n  LTT Calibration Results:")
    logger.info(f"    Best lambda: {best_lambda}")
    logger.info(f"    Valid lambdas found: {len(valid_lambdas)}")
    if ltt_details:
        logger.debug(f"    Pareto candidates tested: {ltt_details.get('n_pareto', 'N/A')}")
        logger.debug(f"    Total grid size: {ltt_details.get('grid_size', 'N/A')}")
    
    # Apply lambda rules to test sets
    logger.info("\n4. Applying lambda rules to test sets...")
    
    if best_lambda is not None:
        rules = LambdaRules(
            lambda1=best_lambda[0],
            lambda2=best_lambda[1],
            lambda3=best_lambda[2],
            gamma=args.gamma
        )
        logger.info(f"  Using λ₁={best_lambda[0]:.4f}, λ₂={best_lambda[1]:.4f}, λ₃={best_lambda[2]:.4f}")
    else:
        logger.warning("  Warning: No valid lambda found. Using defaults.")
        rules = LambdaRules(lambda1=0.5, lambda2=0.05, lambda3=1.0, gamma=args.gamma)
        best_lambda = (0.5, 0.05, 1.0)
    
    refined_sets = []
    detailed_analyses = []
    
    # Counters for summary
    total_gt_rejected = 0
    total_fp_included = 0
    total_fp_removed = 0
    samples_with_gt_rejected = 0
    
    for i, idx in enumerate(test_indices):
        probs = predictions[idx][:, 1]
        infected_nodes = set(np.nonzero(inputs[idx])[0])
        
        refined_set = rules.refine_cp_set(
            test_cp_sets[i], probs, G, infected_nodes
        )
        refined_sets.append(refined_set)
        
        # Detailed analysis for each sample
        analysis = analyze_sample_detailed(
            sample_idx=i,
            probs=probs,
            ground_truths=ground_truths[idx],
            orig_cp_set=test_cp_sets[i],
            refined_set=refined_set,
            rules=rules,
            G=G,
            infected_nodes=infected_nodes,
            logger=logger
        )
        detailed_analyses.append(analysis)
        
        # Update counters
        if len(analysis['gt_rejected']) > 0:
            samples_with_gt_rejected += 1
            total_gt_rejected += len(analysis['gt_rejected'])
        total_fp_included += analysis['n_fp_in_refined']
        total_fp_removed += analysis['n_fp_removed']
    
    logger.info(f"\n  Average refined set size: {np.mean([len(s) for s in refined_sets]):.2f}")
    
    # Summary of GT rejections and FP inclusions
    logger.info(f"\n" + "=" * 60)
    logger.info("DETAILED ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.warning(f"⚠️  Total GT sources falsely rejected: {total_gt_rejected}")
    logger.warning(f"⚠️  Samples with GT rejection: {samples_with_gt_rejected}/{len(test_indices)}")
    logger.info(f"✓  Total FPs correctly removed: {total_fp_removed}")
    logger.info(f"📌 Total FPs still included: {total_fp_included}")
    
    # Evaluate
    logger.info("\n5. Evaluating results...")
    results = evaluate_results(test_cp_sets, refined_sets, ground_truths, test_indices)
    
    logger.info("\n" + "=" * 40)
    logger.info("RESULTS")
    logger.info("=" * 40)
    logger.info(f"\nOriginal CP:")
    logger.info(f"  Coverage: {results['original']['coverage']:.3f}")
    logger.info(f"  Mean Recall: {results['original']['mean_recall']:.3f}")
    logger.info(f"  Mean Set Size: {results['original']['mean_size']:.2f} ± {results['original']['std_size']:.2f}")
    
    logger.info(f"\nCLM-Refined:")
    logger.info(f"  Coverage: {results['refined']['coverage']:.3f}")
    logger.info(f"  Mean Recall: {results['refined']['mean_recall']:.3f}")
    logger.info(f"  Mean Set Size: {results['refined']['mean_size']:.2f} ± {results['refined']['std_size']:.2f}")
    
    # Compute improvement
    size_reduction = (results['original']['mean_size'] - results['refined']['mean_size'])
    size_reduction_pct = 100 * size_reduction / results['original']['mean_size'] if results['original']['mean_size'] > 0 else 0
    logger.info(f"\n  Size Reduction: {size_reduction:.2f} ({size_reduction_pct:.1f}%)")
    
    # Coverage drop analysis
    coverage_drop = results['original']['coverage'] - results['refined']['coverage']
    logger.info(f"  Coverage Drop: {coverage_drop:.3f}")
    
    if coverage_drop > 0.05:
        logger.warning(f"⚠️  SIGNIFICANT COVERAGE DROP: {coverage_drop:.3f} > 0.05 threshold")
    
    # Save results
    logger.info("\n6. Saving results...")
    save_data = {
        'best_lambda': best_lambda,
        'valid_lambdas': valid_lambdas,
        'results': results,
        'test_indices': test_indices,
        'calib_indices': calib_indices,
        'original_cp_sets': test_cp_sets,
        'refined_sets': refined_sets,
        'detailed_analyses': detailed_analyses,
        'params': vars(args),
        'summary': {
            'total_gt_rejected': total_gt_rejected,
            'samples_with_gt_rejected': samples_with_gt_rejected,
            'total_fp_included': total_fp_included,
            'total_fp_removed': total_fp_removed
        }
    }
    
    save_file = os.path.join(output_path, 'clm_results.pickle')
    with open(save_file, 'wb') as f:
        pickle.dump(save_data, f)
    logger.info(f"  Saved to {save_file}")
    
    # Save valid lambdas as text
    lambda_file = os.path.join(output_path, 'lambda_valid.txt')
    with open(lambda_file, 'w') as f:
        f.write(f"Best Lambda: {best_lambda}\n")
        f.write(f"Lambda1 (Diversity): {best_lambda[0]:.4f}\n")
        f.write(f"Lambda2 (Quality): {best_lambda[1]:.4f}\n")
        f.write(f"Lambda3 (Stopping): {best_lambda[2]:.4f}\n")
        f.write(f"\nValid Lambdas ({len(valid_lambdas)} total):\n")
        for lam in valid_lambdas:
            f.write(f"  {lam}\n")
        f.write(f"\n--- Summary ---\n")
        f.write(f"GT sources falsely rejected: {total_gt_rejected}\n")
        f.write(f"Samples with GT rejection: {samples_with_gt_rejected}/{len(test_indices)}\n")
        f.write(f"FPs correctly removed: {total_fp_removed}\n")
        f.write(f"FPs still included: {total_fp_included}\n")
        f.write(f"Coverage drop: {coverage_drop:.4f}\n")
    logger.info(f"  Saved lambdas to {lambda_file}")
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
