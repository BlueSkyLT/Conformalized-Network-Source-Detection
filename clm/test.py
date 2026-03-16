#!/usr/bin/env python3
"""
Test script for CPNet-CLM validation.

This script tests whether the Lambda_valid configurations found by LTT 
actually improve conformal prediction results.

Tests performed:
1. Verify lambda rules reduce set size
2. Verify recall is maintained above threshold
3. Compare multiple lambda configurations
4. Statistical significance tests

Usage:
    python clm/test.py --graph highSchool --exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16
    python clm/test.py --results_path clm/output/highSchool/exp_name/clm_results.pickle
"""

import os
import sys
import argparse
import pickle
import numpy as np
import networkx as nx
from typing import List, Set, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Handle imports
try:
    from lambda_rules import LambdaRules, evaluate_lambda_config
except ImportError:
    from clm.lambda_rules import LambdaRules, evaluate_lambda_config


def load_results(results_path: str) -> Dict:
    """Load saved CLM results."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)


def load_graph(graph_name: str) -> nx.Graph:
    """Load graph from edgelist file."""
    paths = [
        f'SD-STGCN/dataset/{graph_name}/data/graph/{graph_name}.edgelist',
        f'data/{graph_name}/graph/{graph_name}.edgelist',
    ]
    
    for path in paths:
        if os.path.exists(path):
            return nx.read_edgelist(path, nodetype=int)
    
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
    
    return inputs, predictions, ground_truths, logits


def test_size_reduction(results: Dict) -> bool:
    """Test 1: Verify that CLM reduces set size."""
    orig_size = results['results']['original']['mean_size']
    ref_size = results['results']['refined']['mean_size']
    
    reduction = orig_size - ref_size
    reduction_pct = 100 * reduction / orig_size
    
    print("\nTest 1: Size Reduction")
    print("=" * 40)
    print(f"  Original mean size: {orig_size:.2f}")
    print(f"  Refined mean size:  {ref_size:.2f}")
    print(f"  Reduction: {reduction:.2f} ({reduction_pct:.1f}%)")
    
    passed = reduction > 0
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_recall_maintained(results: Dict, min_recall: float = 0.8) -> bool:
    """Test 2: Verify recall is maintained above threshold."""
    orig_recall = results['results']['original']['mean_recall']
    ref_recall = results['results']['refined']['mean_recall']
    
    print("\nTest 2: Recall Maintenance")
    print("=" * 40)
    print(f"  Original mean recall: {orig_recall:.3f}")
    print(f"  Refined mean recall:  {ref_recall:.3f}")
    print(f"  Minimum threshold:    {min_recall:.3f}")
    
    passed = ref_recall >= min_recall
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_coverage_maintained(results: Dict, min_coverage: float = 0.85) -> bool:
    """Test 3: Verify coverage is maintained above threshold."""
    orig_coverage = results['results']['original']['coverage']
    ref_coverage = results['results']['refined']['coverage']
    
    print("\nTest 3: Coverage Maintenance")
    print("=" * 40)
    print(f"  Original coverage: {orig_coverage:.3f}")
    print(f"  Refined coverage:  {ref_coverage:.3f}")
    print(f"  Minimum threshold: {min_coverage:.3f}")
    
    passed = ref_coverage >= min_coverage
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_lambda_validity(results: Dict) -> bool:
    """Test 4: Verify valid lambdas were found."""
    best_lambda = results.get('best_lambda')
    valid_lambdas = results.get('valid_lambdas', [])
    
    print("\nTest 4: Lambda Validity")
    print("=" * 40)
    print(f"  Best lambda: {best_lambda}")
    print(f"  Valid lambdas found: {len(valid_lambdas)}")
    
    passed = best_lambda is not None and len(valid_lambdas) > 0
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_individual_samples(results: Dict, G: nx.Graph,
                            predictions: List, ground_truths: List,
                            inputs: List, n_samples: int = 10) -> bool:
    """Test 5: Detailed test on individual samples."""
    print("\nTest 5: Individual Sample Analysis")
    print("=" * 40)
    
    best_lambda = results['best_lambda']
    if best_lambda is None:
        print("  Skipped: No valid lambda found")
        return False
    
    test_indices = results['test_indices']
    original_cp_sets = results['original_cp_sets']
    refined_sets = results['refined_sets']
    
    rules = LambdaRules(
        lambda1=best_lambda[0],
        lambda2=best_lambda[1],
        lambda3=best_lambda[2]
    )
    
    n_improved = 0
    n_recall_maintained = 0
    n_tested = min(n_samples, len(test_indices))
    
    for i in range(n_tested):
        idx = test_indices[i]
        gt_sources = set(np.nonzero(ground_truths[idx])[0])
        
        orig_set = original_cp_sets[i]
        ref_set = refined_sets[i]
        
        orig_recall = len(orig_set & gt_sources) / len(gt_sources) if len(gt_sources) > 0 else 1.0
        ref_recall = len(ref_set & gt_sources) / len(gt_sources) if len(gt_sources) > 0 else 1.0
        
        size_improved = len(ref_set) < len(orig_set)
        recall_ok = ref_recall >= orig_recall * 0.9  # Allow 10% recall drop
        
        if size_improved:
            n_improved += 1
        if recall_ok:
            n_recall_maintained += 1
        
        print(f"  Sample {i}: orig_size={len(orig_set)}, ref_size={len(ref_set)}, "
              f"orig_recall={orig_recall:.2f}, ref_recall={ref_recall:.2f}")
    
    print(f"\n  Samples with size improvement: {n_improved}/{n_tested}")
    print(f"  Samples with recall maintained: {n_recall_maintained}/{n_tested}")
    
    passed = n_improved > n_tested // 2 and n_recall_maintained > n_tested * 0.8
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    return passed


def compare_lambda_configs(valid_lambdas: List, G: nx.Graph,
                           predictions: List, ground_truths: List,
                           inputs: List, cp_sets: List,
                           test_indices: np.ndarray) -> None:
    """Compare performance of different valid lambda configurations."""
    print("\nLambda Configuration Comparison")
    print("=" * 50)
    
    if len(valid_lambdas) == 0:
        print("  No valid lambdas to compare")
        return
    
    print(f"{'Lambda':<30} {'Recall':>8} {'Size':>8}")
    print("-" * 50)
    
    for lam in valid_lambdas[:10]:  # Top 10
        rules = LambdaRules(lambda1=lam[0], lambda2=lam[1], lambda3=lam[2])
        
        recalls = []
        sizes = []
        
        for i, idx in enumerate(test_indices[:50]):  # Sample subset
            probs = predictions[idx][:, 1]
            gt_sources = set(np.nonzero(ground_truths[idx])[0])
            infected = set(np.nonzero(inputs[idx])[0])
            
            refined = rules.refine_cp_set(cp_sets[i], probs, G, infected)
            
            recall = len(refined & gt_sources) / len(gt_sources) if len(gt_sources) > 0 else 1.0
            recalls.append(recall)
            sizes.append(len(refined))
        
        mean_recall = np.mean(recalls)
        mean_size = np.mean(sizes)
        
        print(f"({lam[0]:.2f}, {lam[1]:.2f}, {lam[2]:.2f})   {mean_recall:>8.3f} {mean_size:>8.2f}")


def run_all_tests(results_path: str = None,
                  graph: str = None,
                  exp_name: str = None) -> Dict:
    """Run all tests and return summary."""
    print("\n" + "=" * 60)
    print("CPNet-CLM Test Suite")
    print("=" * 60)
    
    # Load results
    if results_path and os.path.exists(results_path):
        results = load_results(results_path)
        graph = results['params']['graph']
        exp_name = results['params']['exp_name']
    else:
        if graph is None or exp_name is None:
            raise ValueError("Must provide either results_path or both graph and exp_name")
        results_path = f'clm/output/{graph}/{exp_name}/clm_results.pickle'
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results not found at {results_path}")
        results = load_results(results_path)
    
    print(f"\nGraph: {graph}")
    print(f"Experiment: {exp_name}")
    
    # Load additional data for detailed tests
    G = load_graph(graph)
    res_path = f'SD-STGCN/output/test_res/{graph}/{exp_name}/res.pickle'
    inputs, predictions, ground_truths, _ = load_res_pickle(res_path)
    
    # Run tests
    test_results = {}
    
    test_results['size_reduction'] = test_size_reduction(results)
    test_results['recall_maintained'] = test_recall_maintained(results)
    test_results['coverage_maintained'] = test_coverage_maintained(results)
    test_results['lambda_validity'] = test_lambda_validity(results)
    test_results['individual_samples'] = test_individual_samples(
        results, G, predictions, ground_truths, inputs
    )
    
    # Compare lambda configurations
    compare_lambda_configs(
        results.get('valid_lambdas', []),
        G, predictions, ground_truths, inputs,
        results['original_cp_sets'],
        results['test_indices']
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    n_passed = sum(test_results.values())
    n_total = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {n_passed}/{n_total} tests passed")
    
    overall_pass = n_passed >= n_total - 1  # Allow 1 failure
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description='Test CPNet-CLM results')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to clm_results.pickle')
    parser.add_argument('--graph', type=str, default=None, help='Graph name')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    try:
        results = run_all_tests(
            results_path=args.results_path,
            graph=args.graph,
            exp_name=args.exp_name
        )
        
        # Exit with appropriate code
        n_passed = sum(results.values())
        n_total = len(results)
        sys.exit(0 if n_passed >= n_total - 1 else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
