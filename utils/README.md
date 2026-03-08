# Utils Module

This directory contains utility functions for the conformal prediction pipeline.

## Files

### `score_convert.py`

Core conformity score functions used for conformal prediction.

#### Key Functions

| Function | Description |
|----------|-------------|
| `recall_score(pred_prob, ground_truth, prop_model, infected_nodes)` | Computes recall-based conformity score |
| `recall_score_gtunknown(pred_prob, prop_model, infected_nodes)` | Recall score for all possible labels (test set) |
| `avg_score(pred_prob, ground_truth, prop_model, infected_nodes)` | Computes precision-based conformity score |
| `avg_score_gtunknown(pred_prob, prop_model, infected_nodes)` | Precision score for all possible labels |
| `min_score(pred_prob, ground_truth, prop_model, infected_nodes)` | Computes minimum-probability conformity score |
| `min_score_gtunknown(pred_prob, prop_model, infected_nodes)` | Min score for all possible labels |
| `set_truncate(set_onehot, prob, pow)` | Truncates a set to top-probability elements |
| `nodewise_APS_score(pred_prob, ground_truth, infected_nodes, prop_model)` | Adaptive Prediction Sets score |
| `APS_score(pred_score, ground_truth)` | Basic APS score computation |
| `PGMscore(Y, S, edges, alpha, beta)` | PGM-based set scoring |
| `ArbiTreescore(Y, Y_hat, edges, alpha, beta)` | ArbiTree-based set scoring |

#### Score Functions Explained

**Recall Score** (`recall_score`):
- Measures what proportion of probability mass is "wasted" on nodes ranked higher than the worst true source
- Lower scores indicate better predictions
- Used with `--set_recall` flag

**Precision/Average Score** (`avg_score`):
- Measures average predicted probability of nodes in the upper set
- Penalizes including low-probability nodes
- Used with `--set_prec` flag

**Min Score** (`min_score`):
- Simply the negative of the minimum predicted probability among true sources
- Leads to CRC-style (Conformal Risk Control) prediction sets
- Used with `--set_min` flag

---

### `functions.py`

General helper functions including model evaluation and tree-based methods.

#### Key Functions

| Function | Description |
|----------|-------------|
| `get_test_results(data_file, model_file, res_path, n_node, ...)` | Runs SD-STGCN model on test data, saves `res.pickle` |
| `get_opfeatures(data_file, model_file, train_pct, n_node, ...)` | Extracts output features from model |
| `cpquantile(scores, proportion)` | Computes conformal quantile for calibration |
| `PGMTree(Y, S, from_graph, G)` | Learns tree structure for PGM-CQC method |
| `MPmaxscore(Y_hat, edges, alpha, beta)` | Message-passing for maximum score computation |
| `ArbiTree(Y, Y_hat, ...)` | Learns ArbiTree structure and parameters |

---

## The `res.pickle` Schema

The `get_test_results()` function generates the critical `res.pickle` file that bridges SD-STGCN predictions with conformal prediction.

### Location

```
SD-STGCN/output/test_res/<graph>/<test_exp_name>/res.pickle
```

### Schema

```python
{
    'predictions': list,      # List of batch arrays
    'ground_truth': list,     # List of batch arrays  
    'inputs': list,           # List of batch arrays
    'logits': list            # List of batch arrays
}
```

### Detailed Field Descriptions

#### `predictions`

**Type**: `list` of `numpy.ndarray`

**Shape**: Each array is `(batch_size, n_nodes, 2)`

**Description**: Softmax probabilities from SD-STGCN.
- `predictions[batch][sample, node, 0]` = P(node is NOT a source)
- `predictions[batch][sample, node, 1]` = P(node IS a source)

**Usage**:
```python
# Get source probability for all nodes in first sample
source_prob = predictions[0][0, :, 1]  # Shape: (n_nodes,)
```

#### `ground_truth`

**Type**: `list` of `numpy.ndarray`

**Shape**: Each array is `(batch_size, n_nodes)`

**Description**: One-hot encoded true source labels.
- `ground_truth[batch][sample, node] = 1` if node is a source
- `ground_truth[batch][sample, node] = 0` otherwise

**Usage**:
```python
# Get indices of true sources for first sample
true_sources = np.nonzero(ground_truth[0][0])[0]
```

#### `inputs`

**Type**: `list` of `numpy.ndarray`

**Shape**: Each array is `(batch_size, n_nodes)`

**Description**: Initial epidemic state at the earliest observed time.
- `0` = Susceptible (never infected)
- `1` = Infected (currently infected)
- `2` = Recovered (was infected, now recovered)

**Usage**:
```python
# Get initially infected nodes
infected = np.where(inputs[0][0] >= 1)[0]  # Infected or recovered
```

#### `logits`

**Type**: `list` of `numpy.ndarray`

**Shape**: Each array is `(batch_size, n_nodes, 2)`

**Description**: Raw logits before softmax (for debugging/analysis).

---

## Example: Loading and Using res.pickle

```python
import pickle
import numpy as np

# Load the file
with open('SD-STGCN/output/test_res/highSchool/exp1/res.pickle', 'rb') as f:
    data = pickle.load(f)

# Flatten batches into single lists
all_predictions = []
all_ground_truths = []
all_inputs = []

for batch_idx in range(len(data['predictions'])):
    batch_preds = data['predictions'][batch_idx]
    batch_gt = data['ground_truth'][batch_idx]
    batch_inputs = data['inputs'][batch_idx]
    
    for sample_idx in range(len(batch_preds)):
        all_predictions.append(batch_preds[sample_idx])
        all_ground_truths.append(batch_gt[sample_idx])
        all_inputs.append(batch_inputs[sample_idx])

# Now work with individual samples
n_samples = len(all_predictions)
print(f"Total samples: {n_samples}")

# Example: Get source probabilities for sample 0
sample_0_probs = all_predictions[0][:, 1]  # P(source) for each node
sample_0_true = np.nonzero(all_ground_truths[0])[0]  # True source indices
sample_0_infected = np.nonzero(all_inputs[0])[0]  # Initially infected nodes

print(f"Sample 0:")
print(f"  True sources: {sample_0_true}")
print(f"  Initially infected: {len(sample_0_infected)} nodes")
print(f"  Max source prob: {sample_0_probs.max():.4f}")
```

---

## Dependencies

The utils module depends on:
- `numpy`
- `networkx`
- `tensorflow` (for `get_test_results`)
- `sklearn` (for some helper functions)
- `SD-STGCN/data_loader/data_utils` (data loading utilities)
