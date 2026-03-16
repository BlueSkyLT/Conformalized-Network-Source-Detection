# CLM (Conformal Language Modeling) Framework for CPNet

This folder implements the CLM-enhanced conformal prediction framework for network source detection. The approach applies rejection and stopping rules inspired by Conformal Language Modeling (https://doi.org/10.48550/arXiv.2306.10193) to refine conformal prediction sets.

## Overview

CPNet-CLM uses three lambda-based rules to refine conformal prediction sets:

1. **λ₁ - Diversity Rejection**: Rejects nodes too close (in graph distance) to already-selected sources
2. **λ₂ - Quality Rejection**: Rejects nodes with prediction probability below threshold  
3. **λ₃ - Stopping Rule**: Stops adding nodes when cumulative probability exceeds threshold

The lambda parameters are calibrated using the **Learn Then Test (LTT)** framework with Pareto testing.

---

## Files

| File | Description |
|------|-------------|
| `lambda_rules.py` | Lambda rules implementation (diversity, quality, stopping) |
| `ltt.py` | Learn Then Test framework with Pareto testing |
| `main.py` | Main script to run full CLM-CP process |
| `test.py` | Test script to validate Lambda_valid configurations |
| `plot_gt_distances.py` | Visualization: GT source distance histogram |
| `plot_sdstgcn_probs.py` | Visualization: Probability bar chart with CP/CLM status |
| `output/` | Directory for saved results (Lambda_valid, metrics) |

---

## Lambda Rules (`lambda_rules.py`)

### λ₁: Diversity Rejection

Rejects node v if it's too similar to already-selected sources:

$$\text{Reject } v \text{ if: } \max_{v_j \in \mathcal{C}_\lambda} \{ K(\text{Dist}(v, v_j)) \cdot \hat{\pi}(v) \} > \lambda_1$$

- **K(d)**: Distance decay kernel `exp(-γd)`
- **Dist(v, vⱼ)**: Shortest path distance in graph

### λ₂: Quality Rejection

Rejects nodes with low prediction probability:

$$\text{Reject } v \text{ if: } \hat{\pi}(v) < \lambda_2$$

### λ₃: Stopping Rule

Stops adding nodes when cumulative probability is sufficient:

$$\text{Stop if: } \sum_{v \in \mathcal{C}_\lambda} \hat{\pi}(v) \ge \lambda_3$$

---

## LTT Framework (`ltt.py`)

The Learn Then Test framework finds valid lambda configurations with statistical guarantees:

1. **Generate candidate grid** Λ of lambda configurations
2. **Split calibration data** into optimization (D_opt) and validation (D_val) sets
3. **Stage 1 - Pareto Testing**: Find Pareto-optimal candidates on D_opt (balancing recall and efficiency)
4. **Stage 2 - Sequential Testing**: Test candidates on D_val with FWER control

```python
from ltt import calibrate_ltt

best_lambda, valid_lambdas = calibrate_ltt(
    calibration_data, graph,
    alpha=0.1,      # Recall error tolerance
    delta=0.1,      # Statistical failure probability
    epsilon=0.1,    # Target risk level
    n_grid_points=10
)
```

---

## Usage

### Quick Start

```bash
# Run full CLM process
python clm/main.py --graph highSchool \
    --exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16

# Test if found lambdas improve results
python clm/test.py --graph highSchool \
    --exp_name SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16
```

### Visualization

```bash
# Plot top 100 probability nodes with CP set and CLM rejection status
python clm/plot_sdstgcn_probs.py \
    --res_path SD-STGCN/output/test_res/highSchool/SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16/res.pickle \
    --sample_index 0 \
    --top_k 100 \
    --clm_results clm/output/highSchool/SIR_nsrc7.../clm_results.pickle \
    --output plot.png
```

**Color coding in visualization:**
- 🟢 **Green**: Ground truth source nodes
- 🔵 **Blue**: Nodes accepted by CLM (in refined set)
- 🟠 **Orange**: Nodes rejected by CLM (was in original CP set)
- ⚪ **Gray**: Nodes not in CP set

---

## Output Files

Results are saved to `clm/output/<graph>/<exp_name>/`:

| File | Contents |
|------|----------|
| `clm_results.pickle` | Full results including lambda, CP sets, refined sets, metrics |
| `lambda_valid.txt` | Text file listing valid lambda configurations |

---

## Example Workflow

```python
# 1. Load data
from lambda_rules import LambdaRules
from ltt import calibrate_ltt
import pickle
import networkx as nx

# Load graph and predictions
G = nx.read_edgelist('graph.edgelist', nodetype=int)
with open('res.pickle', 'rb') as f:
    data = pickle.load(f)

# 2. Prepare calibration data
calibration_data = {
    'probs': [...],           # List of probability arrays
    'ground_truths': [...],   # List of ground truth arrays
    'cp_sets': [...],         # List of original CP sets
    'infected_nodes': [...]   # List of infected node sets
}

# 3. Run LTT calibration
best_lambda, valid_lambdas = calibrate_ltt(calibration_data, G)

# 4. Apply lambda rules
rules = LambdaRules(
    lambda1=best_lambda[0],
    lambda2=best_lambda[1],
    lambda3=best_lambda[2]
)

refined_set = rules.refine_cp_set(cp_set, probs, G, infected_nodes)
```

---

## Citation

This implementation is inspired by:

```bibtex
@article{quach2023conformal,
  title={Conformal Language Modeling},
  author={Quach, Victor and Fisch, Adam and Schuster, Tal and others},
  journal={arXiv preprint arXiv:2306.10193},
  year={2023}
}
```

---

## Legacy Scripts

### `plot_gt_distances.py` - Ground Truth Source Distance Histogram

Plots histogram of pairwise shortest-path distances between ground truth source nodes.

```bash
python clm/plot_gt_distances.py --graph highSchool \
    --res_path path/to/res.pickle \
    --sample_index 0 \
    --output distances.png
```

### `plot_sdstgcn_probs.py` - SD-STGCN Probability Bar Chart

Plots bar chart of P(source) for each node with optional CP/CLM status visualization.

```bash
# Basic usage
python clm/plot_sdstgcn_probs.py --res_path path/to/res.pickle --sample_index 0

# With CLM results
python clm/plot_sdstgcn_probs.py --res_path path/to/res.pickle \
    --sample_index 0 \
    --clm_results clm/output/.../clm_results.pickle \
    --top_k 100 \
    --output probs.png
```
