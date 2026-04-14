# PRiSM macOS Setup Guide

Step-by-step setup for running PRiSM on macOS.

## Prerequisites

- macOS 10.15 Catalina or later
- Python 3.11 or higher ([python.org](https://www.python.org/downloads/))
- Visual Studio Code (optional, for notebook editing)

## Setup Steps

### 1. Install Python

Download Python 3.11+ from [python.org](https://www.python.org/downloads/) or use Homebrew:
```bash
brew install python@3.11
```

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
2. Open Command Palette (`Cmd+Shift+P`) -> "Python: Select Interpreter" -> `venv_prism`
3. Open notebooks and select `venv_prism` kernel

**Option B: JupyterLab**
```bash
source venv_prism/bin/activate
pip install jupyterlab
jupyter lab
```

Run notebooks in order: `preprocessing.ipynb` -> `modelling/train_mlp.ipynb` -> `prism_analysis.ipynb`

## GPU Support (Apple Silicon MPS)

MPS (Metal Performance Shaders) is automatically available on Apple Silicon Macs. No additional setup required.

Verify:
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

**Note:** Docker cannot access MPS - use native venv for GPU acceleration on Mac.

## Troubleshooting

- **Kernel restart required:** After changing `.env`, restart the Jupyter kernel
- **Make not found:** Install Xcode Command Line Tools: `xcode-select --install`
- **Python version:** Specify `python3.11` explicitly if multiple versions installed

See the main [README.md](../README.md) for complete documentation.
