# Pipeline Usage Guide

This guide covers advanced pipeline workflows. For basic commands and setup, see the [README](../README.md#automated-pipeline-runners).

---

## PRiSM-Only Mode

PRiSM-only mode loads an existing black-box model and cached partial responses, then re-runs the PRiSM analysis under different conditions -- for example, an alternative LASSO lambda selection strategy, a different random seed, or a non-inferiority threshold. Preprocessing and model training are skipped entirely.

PRiSM-only mode activates automatically when a config contains `prism_only_source_dir` or `prism_only_source_config`, or when the `--prism-only` CLI flag is passed.

### Specifying the source baseline

There are two ways to point at the baseline run whose models and partial responses you want to reuse.

**Option 1: `prism_only_source_dir` (standalone -- recommended for most cases)**

Points at a completed pipeline output directory on disk. Works regardless of how or when the baseline was produced.

```yaml
# In your variant config YAML:
prism_only_source_dir: 'example_notebooks/pipeline_results/20260115_htx_example'
```

Run it standalone:
```bash
prism run my_variant_config
```

**Option 2: `prism_only_source_config` (batch runs only)**

References another config that will be processed in the *same* pipeline invocation. The referenced config must appear **before** the variant config in the command:

```bash
# baseline_cfg runs first, variant_cfg reuses its output
prism run baseline_cfg variant_cfg
```

```yaml
# In variant_cfg.yaml:
prism_only_source_config: 'baseline_cfg'
```

If you run `prism run variant_cfg` alone, the pipeline raises a validation error because `baseline_cfg` is not in the invocation. Use `prism_only_source_dir` for standalone runs.

### What gets validated

Before analysis begins, the pipeline checks that the source directory contains:

- Preprocessing metadata (`preprocessing_metadata_{dataset}_*.json`)
- Model files for each requested model (`{dataset}_{model}_model_*.pt`)
- Processed train/test data (either self-contained within the source, or in the standard data directory)

---

## Partial Response and PRN Caching

In prism-only mode, previously computed partial responses and LASSO results are automatically loaded from the source directory when available. This can save hours of GPU computation.

### Blackbox caching

Cached blackbox partial responses are copied from the source and loaded during analysis. To force recalculation (e.g. after changing the integration method or trim quantile):

```yaml
force_recalculate_partial_responses: true
```

### PRN caching

For experimenting with PRN lambda selection without retraining the network:

```yaml
load_cached_prn: false                          # Load trained PRN model from cache
force_recalculate_prn_partial_responses: false   # Recalculate PRN partial responses
force_recalculate_prn_lasso: false               # Recalculate PRN LASSO sweep
```

When `load_cached_prn: true`:
- The trained PRN model is loaded from cache (training is skipped)
- PRN partial responses are loaded if available, otherwise recalculated
- PRN LASSO results are loaded if available (allowing different lambda selection)
- The blackbox LASSO lambda config must match the source (validated at startup)

The pipeline fails fast if required cached files are missing.

---

## Batch Runs and Config Ordering

### Multi-config invocation

Process multiple configs sequentially in one command:

```bash
prism run htx_example my_config
```

### Batch files

Load a list of configs from a YAML file:

```bash
prism run -f example_notebooks/config/my_batch.yaml
```

Batch file format:
```yaml
configs:
  - baseline_cfg
  - variant_cfg
```

### Ordering constraints

When using `prism_only_source_config`, the referenced config must appear **before** the config that references it. The pipeline validates this ordering upfront and raises an error if violated.

```bash
# Correct: baseline_cfg processed first
prism run baseline_cfg variant_cfg

# Error: variant_cfg references baseline_cfg but it hasn't run yet
prism run variant_cfg baseline_cfg
```

---

## Self-Contained Mode

Copies all input data into the output directory, creating a fully self-contained and portable result folder.

```bash
prism run htx_example --self-contained
```

Or in YAML:
```yaml
self_contained: true
```

The output directory will include `data/interim/` and `data/processed/` subdirectories with copies of all data used for that run. Uses more disk space but ensures complete reproducibility without relying on external data paths.

---

## Multi-GPU Parallel Execution

Distributes model training across multiple NVIDIA GPUs using `run-parallel`:

```bash
# Auto-detect available GPUs
prism run-parallel htx_example

# Specify GPUs explicitly
prism run-parallel htx_example --gpus 0,1,2,3
```

Requires at least 2 GPUs. For single-GPU systems, use `prism run` instead.

### What runs in parallel

**Only the models within a single config are parallelised.** Configs themselves are processed sequentially. Within each config:

1. Preprocessing runs on the main process (sequential)
2. Model training + PRiSM analysis run in parallel across GPUs (one model per GPU, round-robin assignment)
3. Post-processing (prediction concatenation, reproducibility artifacts) runs on the main process

The maximum number of concurrent workers is `min(number_of_models, number_of_GPUs)`.

**Implication:** parallelism helps most when a single config trains multiple models (e.g. `models: [mlp, xgb, rf, logreg]`). With many configs that each train only one model, the pipeline is largely sequential.

---

## Output Directory Structure

Results are saved under `example_notebooks/pipeline_results/`:

```
pipeline_results/
  pipeline_run_{YYYYMMDD_HHMMSS}.log       # Run log
  20260114_htx_example/                     # {date}_{config_name}
    preprocessing_metadata_{dataset}_*.json
    01_preprocessing.pdf
    reproducibility/
      config_name.yaml                      # Config copy
      {dataset}_raw.zip                     # Raw input data
      {dataset}_splits.zip                  # Train/test/val splits
      data_hashes.json                      # SHA256 checksums
      *_best_params.json                    # Tuning results (if applicable)
      prism_only_source_reference.json      # Source tracking (prism-only)
    mlp/
      02_train_mlp.pdf
      03_prism_analysis_mlp.pdf
      models/{dataset}_mlp/
        {dataset}_mlp_model_*.pt
      predictions/
        {dataset}_mlp_preds_*.csv
      partial_responses/
        blackbox_{dataset}_mlp_*_partial_responses.pt
        prn_{dataset}_mlp_*_partial_responses.pt
      lasso_results/
        blackbox_{dataset}_mlp_*_lasso.pt
        prn_{dataset}_mlp_*_lasso.pt
      nomogram/
        *.json
    xgb/
      ...  (same structure)
```

If a directory with the same date and config name already exists and would conflict, the pipeline enumerates (`_1`, `_2`, etc.).

Self-contained mode adds `data/raw/`, `data/interim/`, and `data/processed/` subdirectories within the output folder.

---

## Logging

- **Sequential runs:** `pipeline_results/pipeline_run_{YYYYMMDD_HHMMSS}.log`
- **Parallel runs:** `pipeline_results/pipeline_parallel_run_{YYYYMMDD_HHMMSS}.log`

Logs are duplicated to both the terminal and the log file. ANSI colour codes are stripped from the file output. Container logs typically use UTC timestamps; compare with `date -u` on the host if timestamps appear offset.
