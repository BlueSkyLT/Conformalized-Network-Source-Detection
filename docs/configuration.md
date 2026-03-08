# Configuration Reference

This document describes all command-line arguments for `main.py`.

## Usage

```bash
python main.py [OPTIONS]
```

## Core Arguments

### `--graph`

**Type**: `str`  
**Default**: `highSchool`

The network graph to use. Must correspond to a directory in `SD-STGCN/dataset/`.

**Available options**:
- `highSchool` - High school contact network (327 nodes)
- `bkFratB` - Fraternity network (58 nodes)
- `sfhh` - SFHH conference network (403 nodes)
- `ER` - Erdős-Rényi random graph
- `BA` - Barabási-Albert scale-free graph
- Plus others in `SD-STGCN/dataset/`

**Example**:
```bash
python main.py --graph bkFratB
```

---

### `--train_exp_name`

**Type**: `str`  
**Default**: `SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16`

Name of the training experiment. This determines which pre-trained SD-STGCN model to load.

**Path constructed**: `SD-STGCN/output/models/<graph>/<train_exp_name>/`

**Naming convention**:
```
SIR_nsrc<sources>_Rzero<R0>_beta<beta>_gamma<gamma>_T<time>_ls<samples>_nf<frames>
```
or for mixed parameters:
```
SIR_nsrc<min>-<max>_Rzero<min>-<max>_gamma<min>-<max>_ls<samples>_nf<frames>
```

**Example**:
```bash
python main.py --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16
```

---

### `--test_exp_name`

**Type**: `str`  
**Default**: `SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16`

Name of the test experiment. Determines which test dataset to use.

**Paths constructed**:
- Test data: `SD-STGCN/dataset/<graph>/data/SIR/<test_exp_name>_entire.pickle`
- Results: `SD-STGCN/output/test_res/<graph>/<test_exp_name>/res.pickle`
- Output: `results/<graph>/<test_exp_name>/pow_expected<pow_expected>/`

**Example**:
```bash
python main.py --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16
```

---

## Conformal Prediction Parameters

### `--pow_expected`

**Type**: `float`  
**Default**: `0.5`

The expected power (recall) threshold. This is the minimum proportion of true sources that should be included in the prediction set for it to be considered "covering" the ground truth.

**Range**: `0.0` to `1.0`

- `0.5` = At least 50% of true sources must be in the prediction set
- `1.0` = All true sources must be in the prediction set (strict coverage)
- `0.3` = At least 30% of true sources must be included

**Example**:
```bash
python main.py --pow_expected 0.7
```

---

### `--calib_ratio`

**Type**: `float`  
**Default**: `0.95`

Proportion of data used for calibration (vs. testing).

**Range**: `0.0` to `1.0`

With `n_samples` total:
- Calibration set size: `n_samples * calib_ratio`
- Test set size: `n_samples * (1 - calib_ratio)`

**Example**:
```bash
python main.py --calib_ratio 0.90
```

---

### `--confi_levels`

**Type**: `list of float`  
**Default**: `[0.05, 0.07, 0.10, 0.15, 0.20]`

List of confidence levels (α) to evaluate. Lower α means higher confidence.

**Interpretation**: For α = 0.05, the method targets 95% coverage.

**Example**:
```bash
python main.py --confi_levels 0.05 0.10 0.15
```

---

### `--mc_runs`

**Type**: `int`  
**Default**: `50`

Number of Monte Carlo repetitions. Each run uses a different random calibration/test split to assess variability.

**Note**: Higher values give more stable estimates but increase runtime.

**Example**:
```bash
python main.py --mc_runs 20
```

---

### `--prop_model`

**Type**: `str`  
**Default**: `SI`

Propagation model used in the epidemic simulation.

**Options**:
- `SI` - Susceptible-Infected (no recovery)
- `SIR` - Susceptible-Infected-Recovered

**Note**: The `ADiT_DSI` method only supports `SI`.

**Example**:
```bash
python main.py --prop_model SIR
```

---

## Method Selection Flags

Enable specific conformal prediction methods by setting these to `1`.

### `--set_recall`

**Type**: `int`  
**Default**: `0`

Enable recall-based conformal prediction (CLM-Recall).

**Score function**: `recall_score()` from `utils/score_convert.py`

**Example**:
```bash
python main.py --set_recall 1
```

---

### `--set_prec`

**Type**: `int`  
**Default**: `0`

Enable precision-based conformal prediction (CLM-Precision).

**Score function**: `avg_score()` from `utils/score_convert.py`

**Example**:
```bash
python main.py --set_prec 1
```

---

### `--set_min`

**Type**: `int`  
**Default**: `0`

Enable minimum-probability conformal prediction (CLM-Min).

**Score function**: `min_score()` from `utils/score_convert.py`

**Example**:
```bash
python main.py --set_min 1
```

---

### `--ADiT_DSI`

**Type**: `int`  
**Default**: `0`

Enable ADiT-DSI baseline method (from "Diffusion Source Identification" paper).

**Note**: Only works with `--prop_model SI`.

**Example**:
```bash
python main.py --ADiT_DSI 1 --prop_model SI
```

---

### `--PGM_CQC`

**Type**: `int`  
**Default**: `0`

Enable PGM-CQC method (Probabilistic Graphical Model with Conformal Quantile Calibration).

**Note**: Requires additional calibration data partition.

---

### `--ArbiTree_CQC`

**Type**: `int`  
**Default**: `0`

Enable ArbiTree-CQC method (Arbitrary Tree structured conformal prediction).

**Example**:
```bash
python main.py --ArbiTree_CQC 1
```

---

## ADiT-DSI Parameters

### `--m_l`

**Type**: `int`  
**Default**: `20`

Number of Monte Carlo samples for likelihood estimation in ADiT-DSI.

---

### `--m_p`

**Type**: `int`  
**Default**: `20`

Number of Monte Carlo samples for p-value computation in ADiT-DSI.

---

## PGM-CQC Parameters

### `--n_learn_tree`

**Type**: `int`  
**Default**: `1000`

Number of samples used to learn the tree structure in PGM-CQC.

---

### `--n_jobs_Arbitree`

**Type**: `int`  
**Default**: `5`

Number of parallel jobs for ArbiTree computation.

---

## Example Commands

### Basic Single Method

```bash
python main.py \
    --graph highSchool \
    --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 \
    --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 \
    --pow_expected 0.5 \
    --prop_model SI \
    --set_recall 1
```

### Multiple Methods

```bash
python main.py \
    --graph highSchool \
    --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 \
    --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 \
    --pow_expected 0.5 \
    --prop_model SI \
    --set_recall 1 \
    --set_prec 1 \
    --set_min 1 \
    --ArbiTree_CQC 1
```

### Quick Test Run

```bash
python main.py \
    --graph highSchool \
    --train_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16 \
    --test_exp_name SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16 \
    --mc_runs 5 \
    --confi_levels 0.10 \
    --set_recall 1
```

### High Power Requirement

```bash
python main.py \
    --graph bkFratB \
    --pow_expected 0.9 \
    --set_recall 1 \
    --set_prec 1
```

---

## Environment Variables

### `CUDA_VISIBLE_DEVICES`

By default, the code runs on CPU (set in `utils/functions.py`):
```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

To use GPU, modify this line or set the environment variable:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py ...
```

---

## Output Summary

After running `main.py`, results are saved to:
```
results/<graph>/<test_exp_name>/pow_expected<pow_expected>/
```

See [output_format.md](output_format.md) for detailed output descriptions.
