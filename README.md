# PRiSM - Partial Responses in Structured Models

[![PyPI version](https://img.shields.io/pypi/v/prism-iml.svg)](https://pypi.org/project/prism-iml/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/AIBCTS/PRiSM/blob/main/LICENSE.md)
[![Python 3.11-3.14](https://img.shields.io/badge/python-3.11--3.14-blue.svg)](https://www.python.org/downloads/)

PRiSM is a **model-agnostic framework that converts a trained probabilistic binary classifier
for tabular data into a standalone nomogram that can be audited term by term**. It takes the
*shape* of each nonlinear effect and interaction from the source model (the black box being
converted), not merely which variables mattered, and lets the observed outcome select and
weight them. The result is an additive model that *replaces* the original classifier as the
deployed predictor, with a single regularization parameter setting where you sit on the
sparsity/performance trade-off.

> **Repository:** [https://github.com/AIBCTS/PRiSM](https://github.com/AIBCTS/PRiSM)

---

## PRiSM method overview

```txt
Phase I: decomposition
  source model -> functional ANOVA -> LASSO -> baseline PRiSM nomogram
                  (partial responses)   |
                                        | selected effects
                                        v
Phase II: refinement (optional)
  PRN -> re-decompose -> second LASSO -> PRN-refined nomogram
```

Phase I carries the source model's learned effect shapes across, and the observed outcome
decides which of them survive. Phase II relearns those shapes in a network that admits no
interactions beyond the selected ones. Either nomogram can be deployed in the source model's
place.

---

## Ways to use PRiSM

| Entry point                                      | Use case                                 | Needs |
| ------------------------------------------------ | ---------------------------------------- | ----- |
| `import prism` ([library usage](#library-usage)) | Your own code, any classifier            | pip   |
| `prism_demonstration.ipynb`                      | First exploration, learning the method   | clone |
| Three-notebook pipeline plus `.env`              | Interactive work on a configured dataset | clone |
| `prism run` CLI                                  | Orchestrated runs, saved artifacts       | clone |

---

## Installation

### From PyPI (library only)

```bash
pip install prism-iml
```

### From source (full pipeline with notebooks)

```bash
git clone https://github.com/AIBCTS/PRiSM.git
cd PRiSM

# Python 3.11-3.14 required
python -m venv venv_prism
# Windows: if `python` isn't 3.11-3.14, create with the launcher instead: py -3.12 -m venv venv_prism
# Windows: .\venv_prism\Scripts\activate
# Linux/Mac: source venv_prism/bin/activate

pip install -e ".[notebooks]"
```

`make create_environment` does the same in one step; see
[Environment setup](#environment-setup).

### Library usage

Phase I of the method, end to end on the bundled example dataset: train a black box, decompose
it, and read the result as a nomogram.

```python
from prism.data import load_htx_example_splits
from prism.device_tools import get_device
from prism.lasso import LassoRegression
from prism.maskedmlp import train_mlp
from prism.nomogram_plot import nomogram
from prism.partial_responses import partial_responses

device = get_device()

# 1. Bundled example data: preprocessed, encoded, split (scaler + one-hot groups included)
data = load_htx_example_splits()
X_tr = data.scaler.to_tensor(data.X_train, device=device)
X_te = data.scaler.to_tensor(data.X_test, device=device)

# 2. Train any probabilistic classifier (PRiSM only ever calls predict_proba)
blackbox = train_mlp(
    X_tr, data.y_train, X_te, data.y_test,
    n_hidden=36, lr=0.0014, weight_decay=6e-5, batch_size=512,
    scale_lr=False, device=str(device), plot_loss=False,
)

# 3. Decompose the black box's log-odds into main effects and pairwise interactions
#    ('lebesgue' averages over the data distribution; 'dirac' is faster but ignores it)
pr_params = dict(
    x_train=X_tr, method="lebesgue", device=device, scaler=data.scaler,
    group_manager=data.group_manager, feature_names=data.feature_names,
)
pr_tr = partial_responses(X_tr, blackbox, **pr_params)
pr_te = partial_responses(X_te, blackbox, **pr_params)

# 4. LASSO against the observed outcome picks the terms worth keeping, and weights them
lasso = LassoRegression(
    nlambda=75, min_lambda=0.05, max_lambda=50, max_features=30,
    real_time_plot=False, seed=257, base_model_name="mlp",
)
lasso_results, _ = lasso.fit(
    pr_tr, pr_te, data.y_train, data.y_test,
    feature_names=data.collapsed_feature_names,
)
lasso_results.select_lambda_max_test_auc(target_ratio=0.998)

# 5. The selected terms *are* the predictor: read it as a nomogram
nomogram(
    lasso_results, x=X_tr, model=blackbox, method="lebesgue", x_train=X_tr,
    device=str(device), scaler=data.scaler, feature_names=data.feature_names,
    onehot_group_manager=data.group_manager, show_fig=True, show_conversion_line=True,
)
```

Takes about 2-3 minutes on CPU and selects a nomogram of roughly nine terms. Note that the
outcome enters only at step 4: the decomposition sees just the model's input-output behaviour,
while the labels decide which of its effects are kept. For Phase II (PRN refinement), the
reconstruction check, and held-out validation of all four predictors, see
`example_notebooks/prism_demonstration.ipynb`.

> **Split terminology:** PRiSM says **test** for the split used to tune hyperparameters and the
> LASSO lambda, and **validation** for the final held-out split, which is untouched until the
> end. That is why step 4 above passes `data.y_test`. The order is train 60% / test 20% /
> validation 20%. Some conventions use these two words the other way round.

> **Which should I use?**
>
> Both routes take your own data. They differ in what you write and what you get back.
>
> The **library** (`pip install prism-iml`, `import prism`) works with any probabilistic
> classifier, since PRiSM only ever calls `predict_proba`: a model trained elsewhere, or a
> published score that can no longer be retrained. Supply data with
> `prism.data.prepare_dataset()` and drive the steps yourself.
>
> The **config-driven pipeline** trains the shipped models (MLP, XGBoost, random forest,
> logistic regression) from a YAML config, and adds what an experiment needs: several models
> and configs per run, Optuna tuning, multi-GPU execution, and a dated output directory holding
> models, partial responses, LASSO results, nomograms, and a reproducibility folder with the
> config, zipped splits, and checksums. It lives in the repository and is *not* packaged, so it
> **requires a `git clone`**. See [Using your own data](#using-your-own-data).
>
> Advanced: `PRISM_PROJECT_DIR` relocates the `data/` and `models/` directories the library
> reads and writes, for a pip install working on your own data. It does not replace a clone:
> the `prism` CLI resolves configs and notebooks from `example_notebooks/` in the current
> working directory and ignores it.

---

## Quick start (from cloned repository)

After the [from-source install](#from-source-full-pipeline-with-notebooks) above, open
`example_notebooks/prism_demonstration.ipynb` in VSCode or JupyterLab, select `venv_prism` as
the Jupyter kernel, and Run All. No `.env` or config setup is needed.

Expect **~5 min on a GPU, ~10-20 min on CPU**; set `method = "dirac"` in its setup cell for a
fast (~1-2 min) first pass. What the notebook covers is described under
[Demonstration notebook](#0-demonstration-notebook-start-here) below.

For the production config-driven workflow (YAML configs, caching, artifact export), see
[Using the notebooks](#using-the-notebooks) below.

---

## Environment setup

### Option A: virtual environment (recommended)

Works on all platforms (Windows, Linux, macOS). Best for most users.
[Installation](#from-source-full-pipeline-with-notebooks) above does this by hand; `make` does
the same thing in one step, checking your Python version and installing `requirements.txt`:

```bash
make create_environment
# Windows: .\venv_prism\Scripts\activate
# Linux/Mac: source venv_prism/bin/activate
```

**macOS:** XGBoost needs the OpenMP runtime (`libomp`), which its wheel does not bundle.
Install it once with `brew install libomp`; see the
[macOS Setup guide](https://github.com/AIBCTS/PRiSM/blob/main/docs/SETUP_MACOS.md). Without it,
XGBoost/RF models fail to load.

**GPU support (optional):**

- **NVIDIA CUDA:** Install PyTorch with CUDA before requirements (specify your CUDA version in
  the URL, e.g. `cu126` for CUDA 12.6):
  ```bash
  pip3 install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
  pip install -r requirements.txt
  ```
  Verify: `python -c "import torch; print('CUDA:', torch.cuda.is_available())"`

  Legacy GPU machines whose drivers only support torch < 2.4.1 (e.g. cu118/cu121-era wheels)
  must stay on numpy 1.x, since those torch wheels crash on import under numpy 2:
  ```bash
  pip install "numpy>=1.26.4,<2" "torch<2.4.1" --index-url https://download.pytorch.org/whl/cu121
  ```

- **GPU-optimized inference:** For faster XGBoost/RF partial response calculations:
  ```bash
  # CUDA 12.x (most local installs):
  make requirements-gpu
  # Or: pip install -e ".[gpu]"

  # CUDA 13.x (NGC containers, e.g. DGX Spark/DGX2):
  pip install -e ".[gpu-cuda13x]"
  ```
  This installs cupy for zero-copy GPU tensor operations (speedup on large batches).

- **Apple Silicon (MPS):** Works automatically with standard setup. Verify:
  `python -c "import torch; print('MPS:', torch.backends.mps.is_available())"`

PRiSM picks a device automatically (CUDA, then MPS, then CPU). To force one, set `PRISM_DEVICE`
(`auto`, `cpu`, `cuda`, `cuda:N`, or `mps`); in the config-driven pipeline a config's `device:`
key takes precedence over it.

### Option B: Docker (NVIDIA GPU systems)

Docker provides isolated, reproducible environments. **Requires NVIDIA GPU and drivers.**

| User type                     | Recommended setup                                                |
| ----------------------------- | ---------------------------------------------------------------- |
| Quick start (any system)      | Virtual environment                                              |
| Mac (Apple Silicon)           | Virtual environment only (MPS not accessible in Docker)          |
| Windows (with or without GPU) | Virtual environment (venv with CUDA is simpler than Docker+WSL2) |
| Linux with NVIDIA GPU         | Docker recommended                                               |
| DGX/Multi-GPU systems         | Docker recommended                                               |

```bash
# Quick start with GPU
bash docker-run.sh

# Or using docker compose
docker compose run --rm prism
```

For CPU-only Docker usage:
```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm prism
```

See [README_DOCKER.md](https://github.com/AIBCTS/PRiSM/blob/main/docs/README_DOCKER.md) for
complete Docker documentation.

### Platform-specific setup guides

Detailed setup instructions for each platform:

- [Windows Setup](https://github.com/AIBCTS/PRiSM/blob/main/docs/SETUP_WINDOWS.md)
- [Linux Setup](https://github.com/AIBCTS/PRiSM/blob/main/docs/SETUP_LINUX.md)
- [macOS Setup](https://github.com/AIBCTS/PRiSM/blob/main/docs/SETUP_MACOS.md)

---

## Using the notebooks

### 0. Demonstration notebook (start here)

`example_notebooks/prism_demonstration.ipynb` is a self-contained, pedagogical walkthrough of
the whole PRiSM method on the bundled `htx_example` dataset. It runs entirely in memory (no
`.env`, no YAML config, no intermediate files) and explains the key concepts (functional ANOVA
decomposition, LASSO selection, the reconstruction identity, PRN refinement, nomogram reading)
alongside the code. Start here to learn the method; use the config-driven pipeline below for
real projects.

### Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Two configuration modes are available:

```bash
# Option A: Full config (loads example_notebooks/config/{name}.yaml)
PRISM_CONFIG=htx_example

# Option B: Quick mode (uses CSV in data/raw/ with defaults)
PRISM_DATASET=my_data
```

Optional directory overrides, also set in `.env`:

```bash
PRISM_MODELS_DIR=           # Custom models directory
PRISM_INTERIM_DATA_DIR=     # Custom interim data directory
PRISM_PROCESSED_DATA_DIR=   # Custom processed data directory
```

> **Important:** Restart your Jupyter kernel after changing `.env` for changes to take effect.

### Notebook pipeline

Run notebooks in `example_notebooks/` in this order:

1. **Preprocessing** (`preprocessing.ipynb`)
   - Loads raw data, handles missing values, encodes categoricals
   - Splits into train/test/validation sets
   - Outputs to `data/interim/` and `data/processed/`

   > **Note on terminology:** "test" is the tuning split and "validation" the final held-out
   > split, as described under [Library usage](#library-usage) above.

2. **Model training** (`modelling/` subdirectory)
   - `train_mlp.ipynb` - Multilayer Perceptron
   - `train_xgb.ipynb` - XGBoost
   - `train_rf.ipynb` - Random Forest
   - `train_logreg.ipynb` - Logistic Regression

3. **PRiSM analysis** (`prism_analysis.ipynb`)
   - **Phase I - Black-box decomposition:** functional ANOVA decomposes predictions into
     univariate main effects and bivariate interactions (partial responses); LASSO selects
     important terms and assembles a baseline PRiSM nomogram
   - **Phase II - PRN refinement:** trains a Partial Response Network (structured MLP
     constrained to selected effects), re-decomposes its predictions, and applies a second
     LASSO pass to produce the final PRN-refined nomogram

> **IMPACT:** `modelling/train_impact.ipynb` wraps the published IMPACT risk index, a fixed
> coefficient score for mortality after heart transplantation. There is no training step: a
> published score converts exactly as a trained model does. It needs a dataset preprocessed to
> the 18 IMPACT variables (`prism.impact.IMPACTModel.get_feature_column_names()`), so unlike
> the four notebooks above it does not run on `htx_example`.

### Using your own data

1. Place your CSV file in `data/raw/` (e.g., `data/raw/my_data.csv`)
2. Create a config file `example_notebooks/config/my_data.yaml`:
   ```yaml
   dataset: my_data
   models: [mlp]
   target_candidates: ['your_target_column']
   splitting_method: 'random'
   split_ratios: [0.6, 0.2, 0.2]
   ```
3. Set `PRISM_CONFIG=my_data` in `.env`
4. Run the notebook pipeline

See [Configuration reference](#configuration-reference) below for the full set of options.

To do the same in code, without a clone, swap `load_htx_example_splits()` for
`prism.data.prepare_dataset(df, target_column=...)` in step 1 of
[Library usage](#library-usage); the rest of that snippet is unchanged. Splitting and encoding
become keyword arguments rather than config keys, and nothing is written to disk.

> **OpenML datasets:** You can also use datasets from [OpenML](https://www.openml.org/)
> directly by setting `dataset: openml_<id>` in your config (e.g., `openml_31` for credit-g).
> The dataset will be fetched automatically via the OpenML API. Requires the `openml` extra:
> `pip install -e ".[openml]"`.

### Example datasets

- **htx_example**: Fully synthetic heart transplant data (5,000 rows) approximating the
  statistical properties and temporal split of the data used in the
  [published study](#citation). No patient data is replicated.

  Expected full-pipeline runtime (`prism run htx_example`, single MLP model): **~5 min on a GPU
  (CUDA), ~10 min on CPU**. The PRiSM analysis stage dominates; set
  `partial_response_method: dirac` in the config for a faster CPU run.

  `htx_example_sparse.yaml` runs the same dataset under the sparse lambda strategy, which
  trades a little discrimination for a nomogram with fewer terms. It loads the models from the
  baseline run, so run `htx_example` first.

- **credit-g**: The Statlog German Credit dataset (H. Hofmann, 1994; 1,000 rows), obtained via
  [OpenML dataset 31](https://www.openml.org/d/31), original source UCI Machine Learning
  Repository. Used by the test suite as a small mixed-type dataset.

---

## Automated pipeline runners

For batch experiments and reproducible research, use the `prism` command-line runners. **Run
them from the root of a cloned repository**: they drive the notebooks in `example_notebooks/`,
which the PyPI package does not contain.

```bash
prism run htx_example                       # Run full pipeline
prism run htx_example my_config             # Multiple configs
prism run htx_example --skip-preprocessing  # Use existing preprocessed data
prism run-parallel htx_example --gpus 0,1   # Multi-GPU execution
prism tune htx_example                      # Optuna hyperparameter tuning
prism list-configs                          # List available configs
```

See the
[Pipeline Usage Guide](https://github.com/AIBCTS/PRiSM/blob/main/docs/PIPELINE_USAGE.md) for
multi-GPU execution, PRiSM-only mode, batch runs, caching, output structure, and other advanced
workflows.

Results are saved to `example_notebooks/pipeline_results/{date}_{config}/{model}/`.

> **Deprecated:** the root `run_prism_pipeline.py`, `run_prism_parallel.py` and
> `run_hyperparameter_tuning.py` scripts are thin wrappers kept so that existing invocations
> keep working. They will be removed in a future release; use the `prism` commands above.

---

## Configuration reference

YAML config files in `example_notebooks/config/` control all aspects of the pipeline:

```yaml
# Required
dataset: my_dataset          # Maps to data/raw/{dataset}.csv
models: [mlp, xgb]          # Models to train

# Preprocessing
random_seed: 257
splitting_method: 'random'   # 'random', 'temporal', or 'predefined'
split_ratios: [0.6, 0.2, 0.2]
target_candidates: ['outcome', 'target']
id_candidates: ['id', 'patient_id']

# Categorical encoding
integer_encoding:
  education: ['Elementary', 'High School', 'Bachelor', 'Master', 'PhD']

# PRiSM analysis
partial_response_method: 'lebesgue'  # or 'dirac'
save_nomogram_json: true

# LASSO lambda selection
# 'max_test_auc': sparsest lambda holding target_ratio of the maximum tuning AUROC
# 'non_inferiority': sparsest lambda still noninferior to the best, by ni_level
lasso_lambda_selection:
  blackbox:
    method: 'max_test_auc'
    target_ratio: 0.998
  prn:
    method: 'max_test_auc'
    target_ratio: 0.998

# Hyperparameter tuning (Optuna)
hyperparameter_tuning:
  mlp:
    enabled: true
    n_trials: 25
  prn:
    enabled: true
    n_trials: 15
```

See `example_notebooks/config/example_config.yaml` for complete documentation of all options.

---

## Project organization

```txt
PRiSM/
|-- Makefile                <- Convenience commands (make create_environment, etc.)
|-- README.md               <- This file
|-- Dockerfile              <- Image used by docker-compose.yml and docker-run.sh
|-- run_*.py                <- Deprecated stubs (use the `prism` CLI instead)
|
|-- docs/                   <- Setup, pipeline, and Docker guides
|-- tests/                  <- Test suite (make test)
|
|-- data/                   <- Data directory
|   |-- raw/                <- Original data files (ships with htx_example.csv)
|   |-- interim/            <- Intermediate transformed data
|   +-- processed/          <- Final datasets for modeling
|
|-- models/                 <- Trained models, predictions, results
|
|-- example_notebooks/      <- Jupyter notebooks
|   |-- preprocessing.ipynb <- Data preprocessing
|   |-- prism_analysis.ipynb <- PRiSM analysis
|   |-- config/             <- YAML configuration files
|   +-- modelling/          <- Model training notebooks
|
|-- prism/                  <- Source code package
|   |-- config.py           <- Dataset configuration and paths
|   |-- preprocessing.py    <- Data preprocessing functions
|   |-- device_tools.py     <- GPU/CPU device management
|   |-- maskedmlp.py        <- Masked MLP model
|   |-- cli/                <- CLI entry points (prism run, tune, etc.)
|   |-- data/               <- Bundled example datasets (htx_example)
|   |-- lasso/              <- LASSO feature selection
|   |-- partial_responses/  <- Partial response calculation
|   +-- plotting/           <- Visualization pipeline
|
|-- requirements.txt        <- Python dependencies (editable install)
+-- pyproject.toml          <- Package metadata and dependency spec
```

---

## Development

Install development dependencies:

```bash
make create_environment_dev
# Or: pip install -e ".[dev,test]"
```

Run tests:
```bash
make test              # Run all tests
make test-coverage     # With coverage report
```

Format and lint:
```bash
make format            # Format with black
make lint              # Run flake8, isort, black checks
```

Export notebooks to PDF/Python:
```bash
nbautoexport export example_notebooks
```
Requires TeX and Pandoc. See
[LATEX_TROUBLESHOOTING.md](https://github.com/AIBCTS/PRiSM/blob/main/docs/LATEX_TROUBLESHOOTING.md)
for help.

---

## Data availability

`htx_example`, the bundled example dataset, is fully synthetic: generated to approximate the
statistical properties of UNOS heart transplant registry data. No patient data is replicated.
The registry data is available from SRTR
(https://www.srtr.org/requesting-srtr-data/data-requests/) under authorized license.

---

## Citation

If you use PRiSM in published work, please cite:

> Pigot H, Lisboa PJG, Ortega-Martorell S, Olier I, Mahon J, Nilsson J. Translation of
> black-box clinical prediction models into standalone transparent nomograms: temporal external
> validation in heart transplantation.

The paper is in preparation; its preprint link and DOI will be added here once posted.

## Attribution

Developed by Henry Pigot, Paulo J. G. Lisboa, and Johan Nilsson, and released under the BSD
3-Clause License (see [LICENSE.md](https://github.com/AIBCTS/PRiSM/blob/main/LICENSE.md)).

Parts of the code and documentation were written with AI assistance. The authors reviewed all
AI-assisted contributions and take responsibility for the content of this repository.

```
Assisted-by: Anthropic Claude
```
