# PRiSM Windows Setup Guide

Step-by-step setup for running PRiSM on Windows.

## Prerequisites

- Windows 10/11
- Python 3.11 or higher ([python.org](https://www.python.org/downloads/))
- Visual Studio Code ([code.visualstudio.com](https://code.visualstudio.com/))

## Setup Steps

### 1. Install Python

Download Python 3.11+ from [python.org](https://www.python.org/downloads/).

**Important:** Check "Add Python to PATH" during installation.

### 2. Enable PowerShell Scripts (if needed)

Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

### 3. Create Virtual Environment

Open PowerShell/Terminal and navigate to the project directory:
```powershell
cd path\to\PRiSM
```

Create and activate the virtual environment:
```powershell
# Using make (recommended)
make create_environment

# Or manually
python -m venv venv_prism --clear --copies
.\venv_prism\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure VSCode

1. Install VSCode extensions: **Python** and **Jupyter**
2. Open Command Palette (`Ctrl+Shift+P`)
3. Select "Python: Select Interpreter"
4. Choose the interpreter from `venv_prism`

### 5. Configure Dataset

```powershell
copy .env.example .env
```

The default configuration uses `htx_example`. Edit `.env` to change datasets.

### 6. Run Notebooks

1. Open any notebook in `example_notebooks/`
2. Select `venv_prism` as the Jupyter kernel (top right)
3. Run cells in order: `preprocessing.ipynb` -> `modelling/train_mlp.ipynb` -> `prism_analysis.ipynb`

## GPU Support (NVIDIA CUDA)

For CUDA support, install PyTorch with CUDA before other dependencies:
```powershell
.\venv_prism\Scripts\activate
pip3 install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
pip install -r requirements.txt
```

Verify:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Troubleshooting

- **Kernel restart required:** After changing `.env`, restart the Jupyter kernel
- **Make not found:** Install via Chocolatey (`choco install make`) or use manual commands above
- **Permission denied:** Run PowerShell as Administrator

See the main [README.md](../README.md) for complete documentation.
