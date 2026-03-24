# Expected Outputs

This document lists all files that should appear after successfully running the conformal prediction pipeline.

## After Running `examples/run_one_experiment.sh`

Assuming all prerequisites are met and the script completes successfully, the following files will be created in:

```
results/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16/pow_expected0.5/
```

### Summary Tables (CSV)

| File | Description |
|------|-------------|
| `coverage_table.csv` | Coverage rates for each method and α level |
| `set_size_table.csv` | Average prediction set sizes |
| `time_cost_table.csv` | Computation time in seconds |

### Visualizations (PDF)

| File | Description |
|------|-------------|
| `coverage_boxplot.pdf` | Box plots of coverage across methods |
| `set_size_boxplot.pdf` | Box plots of set sizes across methods |

### Cached Results (Pickle)

For each Monte Carlo run (0 to mc_runs-1):

| File Pattern | Description |
|--------------|-------------|
| `set_recall_repeat<i>.pickle` | Cached recall method results |
| `set_prec_repeat<i>.pickle` | Cached precision method results |
| `set_min_repeat<i>.pickle` | Cached min method results |

### Calibration Indices (NumPy)

| File Pattern | Description |
|--------------|-------------|
| `calib_index_repeat<i>.npy` | Calibration set indices for run i |

---

## Example Directory Listing

After running with `--mc_runs 10`:

```
results/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16/pow_expected0.5/
├── coverage_table.csv
├── set_size_table.csv
├── time_cost_table.csv
├── coverage_boxplot.pdf
├── set_size_boxplot.pdf
├── calib_index_repeat0.npy
├── calib_index_repeat1.npy
├── calib_index_repeat2.npy
├── calib_index_repeat3.npy
├── calib_index_repeat4.npy
├── calib_index_repeat5.npy
├── calib_index_repeat6.npy
├── calib_index_repeat7.npy
├── calib_index_repeat8.npy
├── calib_index_repeat9.npy
├── set_recall_repeat0.pickle
├── set_recall_repeat1.pickle
├── ...
├── set_recall_repeat9.pickle
├── set_prec_repeat0.pickle
├── set_prec_repeat1.pickle
├── ...
├── set_prec_repeat9.pickle
├── set_min_repeat0.pickle
├── set_min_repeat1.pickle
├── ...
└── set_min_repeat9.pickle
```

---

## Intermediate Files Generated

### Cached Graph Copy

```
data/highSchool/graph/highSchool.edgelist
```

### Cached Test Results

```
data/highSchool/test_res/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16/res.pickle
```

### SD-STGCN Predictions (if not already present)

```
SD-STGCN/output/test_res/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16/res.pickle
```

---

## After Running `visualization.py`

```bash
python visualization.py --sample_index 6
```

Creates files in:
```
results/<graph>/<test_exp>/pow_expected<p>/visualizations/
```

Example outputs:
```
visualizations/
├── sample6_set_recall_alpha0.10_pow0.5_x[0.5-0.8]_y[0.2-0.6]_sources.png
├── sample6_set_recall_alpha0.10_pow0.5_x[0.5-0.8]_y[0.2-0.6]_sources_only.png
└── sample6_set_recall_alpha0.10_pow0.5_x[0.5-0.8]_y[0.2-0.6]_plain_graph.png
```

---

## Verifying Successful Completion

### 1. Check Console Output

The script should print:
```
finished.
```

And display coverage/set size tables before that.

### 2. Check File Counts

With `--mc_runs 10` and 3 methods enabled:
- 3 CSV files
- 2 PDF files
- 10 calibration index files
- 30 result pickle files (10 per method)

Total: ~45 files

### 3. Verify Coverage Values

In `coverage_table.csv`, coverage values should be:
- Close to or above (1-α) for each confidence level
- For α=0.05, expect coverage ≥ 0.95
- For α=0.10, expect coverage ≥ 0.90

---

## Common Issues

### Missing Files

If some files are missing:

1. **Check log output** for errors
2. **Verify prerequisites** exist (graph, model, test data)
3. **Check method flags** - only enabled methods produce output

### Empty Results

If CSV files are empty or missing rows:
- Ensure at least one method flag is set (`--set_recall 1`, etc.)

### Partial Results

If only some MC runs completed:
- Check for timeout or memory errors
- Reduce `--mc_runs` for testing
