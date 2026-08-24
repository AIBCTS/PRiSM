# Configuration files

## Datasets

### htx_example (demo dataset)

Lightweight synthetic heart transplant example dataset for testing and demonstrating
the pipeline. Approximates the statistical properties of UNOS registry data (2000-2022)
without replicating actual patient data.

- `htx_example.yaml` - baseline lambda selection (max test AUC)
- `htx_example_sparse.yaml` - sparse lambda selection (noninferiority)

## Lambda selection strategies

Each dataset can have two config variants:

- **Baseline** (`max_test_auc`): selects the lambda preserving >=99.8% of maximum
  tuning AUROC. Produces a full-complexity nomogram.
- **Sparse** (`non_inferiority`): selects the sparsest lambda whose tuning AUROC
  remains noninferior to the best (within 10% of useful AUC above chance). Produces
  a reduced-complexity nomogram with fewer terms.

Sparse configs use `prism_only_source_config` to load models from the corresponding
baseline run, so the baseline config must be run first.

## Batch execution

Run multiple configs in sequence:

```bash
prism run -f example_notebooks/config/example_batch.yaml
```

## Custom datasets

See `example_config.yaml` for a reference covering every configuration
option, including preprocessing, model training, and PRiSM analysis settings.
