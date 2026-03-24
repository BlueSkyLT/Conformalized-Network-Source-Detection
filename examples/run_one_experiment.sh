#!/bin/bash
# ============================================================================
# Example: Run One Conformal Prediction Experiment
# ============================================================================
#
# This script demonstrates how to run a single conformal prediction experiment
# on the highSchool network.
#
# PREREQUISITES:
# 1. Graph file exists at:
#    SD-STGCN/dataset/highSchool/data/graph/highSchool.edgelist
#
# 2. Test simulation data exists at:
#    SD-STGCN/dataset/highSchool/data/SIR/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16_entire.pickle
#
# 3. Trained model exists at:
#    SD-STGCN/output/models/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16/
#
# If prerequisites are not met, see docs/pipeline.md for data generation steps.
#
# EXPECTED OUTPUT:
# - Results in: results/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16/pow_expected0.5/
# - See examples/expected_outputs.md for full list of output files
# ============================================================================

set -e  # Exit on error

# Change to repository root directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Running Conformal Prediction Experiment"
echo "=============================================="
echo ""
echo "Graph: highSchool"
echo "Train experiment: SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16"
echo "Test experiment: SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16"
echo "Power threshold: 0.5"
echo "Methods: set_recall, set_prec, set_min"
echo "Monte Carlo runs: 10"
echo ""

python main.py \
    --graph highSchool \
    --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 \
    --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 \
    --pow_expected 0.5 \
    --prop_model SI \
    --calib_ratio 0.95 \
    --confi_levels 0.05 0.10 0.15 0.20 \
    --set_recall 1 \
    --set_prec 1 \
    --set_min 1 \
    --mc_runs 10

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  results/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16/pow_expected0.5/"
echo ""
echo "Output files:"
ls -la results/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16/pow_expected0.5/ 2>/dev/null || echo "  (run the script to generate outputs)"
