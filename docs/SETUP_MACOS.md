# PRiSM macOS setup guide

Step-by-step setup for running PRiSM on macOS.

## Prerequisites

- macOS 10.15 Catalina or later
- Python 3.11 or higher ([python.org](https://www.python.org/downloads/))
- OpenMP runtime for XGBoost: `brew install libomp` (required; see below)
- Visual Studio Code (optional, for notebook editing)

### OpenMP runtime (required for XGBoost)

XGBoost is a core dependency, and its macOS wheel does not bundle the OpenMP
runtime: it loads `libomp.dylib` at import/train time. Install it with Homebrew:

```bash
brew install libomp
```

Without it you will see `XGBoost Library (libxgboost.dylib) could not be loaded ...
Library not loaded: @rpath/libomp.dylib` the first time an XGBoost model is used.

## Setup steps

### 1. Install Python

Download Python 3.11+ from [python.org](https://www.python.org/downloads/) or use Homebrew:
```bash
brew install python@3.11
```

### 2. Create virtual environment

Navigate to the project directory and create the environment:
```bash
cd path/to/PRiSM

# Using make (recommended)
make create_environment

# Or manually
python3.11 -m venv venv_prism --clear --copies
source venv_prism/bin/activate
pip install -r requirements.txt
```

### 3. Configure dataset

```bash
cp .env.example .env
```

The default configuration uses `htx_example`. Edit `.env` to change datasets.

### 4. Run notebooks

**Option A: VSCode (recommended)**
1. Install VSCode with Python and Jupyter extensions
2. Open Command Palette (`Cmd+Shift+P`) -> "Python: Select Interpreter" -> `venv_prism`
3. Open notebooks and select `venv_prism` kernel

**Option B: JupyterLab**
```bash
source venv_prism/bin/activate
pip install jupyterlab
jupyter lab
```

Run notebooks in order: `preprocessing.ipynb` -> `modelling/train_mlp.ipynb` -> `prism_analysis.ipynb`

## GPU support (Apple Silicon MPS)

MPS (Metal Performance Shaders) is automatically available on Apple Silicon Macs. No additional setup required.

Verify:
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

**Note:** Docker cannot access MPS - use native venv for GPU acceleration on Mac.

## Performance on macOS (OpenMP threading)

On import, PRiSM sets `OMP_NUM_THREADS=1` on macOS (only when you have not set it
yourself). This prevents a crash caused by PyTorch and XGBoost each loading a
separate copy of the OpenMP runtime (`libomp`); see Troubleshooting for the
mechanism.

This setting is **process-global**: it serializes OpenMP CPU parallelism for
XGBoost, PyTorch CPU operations, and typically numpy/scipy. It does **not**
affect MPS/Metal GPU compute; on Apple Silicon PRiSM auto-selects MPS, so GPU
tensor work (e.g. the Lebesgue partial-response decomposition) runs on the GPU
and is unaffected. The practical cost is mainly single-threaded XGBoost training
plus any CPU-bound (or MPS-fallback) tensor/BLAS work.

If you run CPU-heavy workloads on macOS and want full multithreading, make
PyTorch and XGBoost share PyTorch's single OpenMP runtime by pointing the dynamic
loader at it *before* launching Python (it is read only at process start):

```bash
export DYLD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
```

With that set, both libraries load one shared `libomp`, so there is no conflict
and OpenMP stays multithreaded.

## Troubleshooting

- **Kernel restart required:** After changing `.env`, restart the Jupyter kernel
- **Make not found:** Install Xcode Command Line Tools: `xcode-select --install`
- **Python version:** Specify `python3.11` explicitly if multiple versions installed
- **XGBoost fails to load (`libomp.dylib` not found):** Install the OpenMP runtime with `brew install libomp` (see Prerequisites).
- **XGBoost crashes or hangs on macOS:** PyTorch and XGBoost each load their own copy of the OpenMP runtime (`libomp`); with both in one process, XGBoost's parallel initialization can collide with the second runtime (a deadlock on older toolchains, a segfault on newer ones). PRiSM handles this automatically; see [Performance on macOS](#performance-on-macos-openmp-threading) for the mechanism and the multithreading opt-in.

See the main [README.md](../README.md) for complete documentation.
