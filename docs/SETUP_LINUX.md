# PRiSM Linux Setup Guide

Step-by-step setup for running PRiSM on Linux.

## Prerequisites

- Linux (Ubuntu 20.04+, Debian, or similar)
- Python 3.11 or higher
- Visual Studio Code (optional, for notebook editing)

## Setup Steps

### 1. Install Python

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**Other distributions:** Use your package manager or download from [python.org](https://www.python.org/downloads/).

### 2. Create Virtual Environment

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

### 3. Configure Dataset

```bash
cp .env.example .env
```

The default configuration uses `htx_example`. Edit `.env` to change datasets.

### 4. Run Notebooks

**Option A: VSCode (recommended)**
1. Install VSCode with Python and Jupyter extensions
2. Open Command Palette (`Ctrl+Shift+P`) -> "Python: Select Interpreter" -> `venv_prism`
3. Open notebooks and select `venv_prism` kernel

**Option B: JupyterLab**
```bash
source venv_prism/bin/activate
pip install jupyterlab
jupyter lab
```

Run notebooks in order: `preprocessing.ipynb` -> `modelling/train_mlp.ipynb` -> `prism_analysis.ipynb`

## GPU Support (NVIDIA CUDA)

For CUDA support, install PyTorch with CUDA before other dependencies:
```bash
source venv_prism/bin/activate
pip3 install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
pip install -r requirements.txt
```

Verify:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Docker Alternative

For isolated environments with GPU support, see [README_DOCKER.md](README_DOCKER.md).

## Troubleshooting

- **Kernel restart required:** After changing `.env`, restart the Jupyter kernel
- **Python version:** Ensure `python3.11` is available; you may need to install it separately
- **Make not found:** Install with `sudo apt install make` or use manual commands above

See the main [README.md](../README.md) for complete documentation.
