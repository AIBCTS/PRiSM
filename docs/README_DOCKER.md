# Docker Usage Guide for PRiSM

This guide explains how to use Docker to run the PRiSM project in a containerized environment.

## Table of Contents

- [Is Docker Right for You?](#is-docker-right-for-you)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Container](#running-the-container)
- [GPU Support](#gpu-support)
- [CPU-Only Usage](#cpu-only-usage)
- [Multi-GPU / DGX Systems](#multi-gpu--dgx-systems)
- [Volume Mounts](#volume-mounts)
- [Common Use Cases](#common-use-cases)
- [Graceful Process Termination](#graceful-process-termination)
- [Troubleshooting](#troubleshooting)
- [VSCode Remote Development](#vscode-remote-development)

---

## Is Docker Right for You?

Docker provides isolation and reproducibility, but may not be the best choice for everyone.

| Your Setup                      | Recommended Approach     | Why                                                |
| ------------------------------- | ------------------------ | -------------------------------------------------- |
| **Quick exploration**           | Virtual environment      | Simpler setup, works immediately                   |
| **Mac (Apple Silicon)**         | Virtual environment only | MPS GPU not accessible from Docker (runs Linux VM) |
| **Windows (with or without GPU)** | Virtual environment    | Docker GPU on Windows requires WSL2 setup; venv with CUDA is simpler |
| **Linux with NVIDIA GPU**       | Docker                   | Better isolation, reproducibility                  |
| **DGX / Multi-GPU server**      | Docker                   | Optimized, includes devcontainer support           |

**Bottom line:** If you have an NVIDIA GPU on Linux/WSL2, Docker is great. Otherwise, use a virtual environment (see main [README.md](../README.md)).

---

## Prerequisites

1. **Docker** - See [Docker installation guide](https://docs.docker.com/get-docker/)
2. **NVIDIA Container Toolkit** (for GPU) - See [NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify installation:
```bash
docker --version

# Verify GPU support (NVIDIA systems only)
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

```bash
# Recommended: Use the helper script (auto-detects GPU)
bash docker-run.sh

# Or with docker-compose
docker-compose run --rm prism
```

This will:
- Build the image if needed
- Detect and configure GPU support
- Mount project directories
- Start an interactive bash shell

---

## Building the Docker Image

The image is based on NVIDIA PyTorch (`nvcr.io/nvidia/pytorch:25.11-py3`) with all dependencies.

```bash
# Auto-build via helper script
bash docker-run.sh

# Or build manually
docker build -t prism:latest .

# Force clean rebuild
docker build --no-cache -t prism:latest .
```

Rebuild after changing `pyproject.toml` or `Dockerfile`.

---

## Running the Container

### Method 1: Helper Script (Recommended)

```bash
# Interactive shell
bash docker-run.sh

# Run a command
bash docker-run.sh python run_prism_pipeline.py htx_example
```

### Method 2: Docker Compose

```bash
docker-compose run --rm prism
docker-compose run --rm prism python run_prism_pipeline.py htx_example
```

### Method 3: Direct Docker Command

```bash
docker run --gpus all -it --rm --ipc=host \
    -v "$(pwd)":/workspace \
    -w /workspace \
    prism:latest \
    /bin/bash
```

---

## GPU Support

### Automatic Detection

The `docker-run.sh` script automatically detects NVIDIA GPUs and configures `--gpus all`.

### Verifying GPU Access

Inside the container:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi
```

### Selecting Specific GPUs

```bash
docker run --gpus '"device=0,1"' -it --rm --ipc=host \
    -v "$(pwd)":/workspace \
    -w /workspace \
    prism:latest /bin/bash
```

---

## CPU-Only Usage

For systems without NVIDIA GPUs, use the CPU override file:

```bash
# Using docker-compose (recommended)
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm prism

# Or modify the direct command (remove --gpus flag)
docker run -it --rm --ipc=host \
    -v "$(pwd)":/workspace \
    -w /workspace \
    prism:latest /bin/bash
```

The CPU override file (`docker-compose.cpu.yml`) removes NVIDIA-specific settings while preserving all other functionality.

---

## Multi-GPU / DGX Systems

On multi-GPU NVIDIA systems (e.g., DGX), the `docker-run.sh` script automatically applies optimizations:
- `--ulimit memlock=-1` -- removes memory lock limits (NCCL needs pinned memory for multi-GPU IPC)
- `--ulimit stack=67108864` -- increases stack size to 64MB (needed by NCCL's recursive graph algorithm)

These flags are NVIDIA-specific and are not needed for CPU-only usage.

For parallel pipeline execution across GPUs, see `run_prism_parallel.py` in the main [README.md](../README.md).

---

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./` | `/workspace` | Project root |
| `./data` | `/workspace/data` | Data files |
| `./models` | `/workspace/models` | Trained models |

Changes in mounted directories sync immediately between host and container.

---

## Common Use Cases

### Running the PRiSM Pipeline

```bash
bash docker-run.sh python run_prism_pipeline.py htx_example
```

### Running Jupyter Lab

```bash
bash docker-run.sh jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Access at `http://localhost:8888` (port-forward if on remote server).

### Running Tests

```bash
bash docker-run.sh pytest tests/
```

### Interactive Development

```bash
bash docker-run.sh
# Inside container:
jupyter lab
python script.py
pytest tests/
```

---

## Graceful Process Termination

When running long pipeline jobs, use SIGINT (Ctrl+C equivalent) for graceful shutdown:

```bash
# Find the process
ps aux | grep run_prism_pipeline.py

# Send SIGINT for graceful shutdown
pkill -2 -f "run_prism_pipeline.py"

# Or by PID
kill -2 <PID>
```

This allows the pipeline to save progress and clean up properly.

For running jobs that survive disconnection, see [docs/VSCODE_REMOTE_DOCKER.md](VSCODE_REMOTE_DOCKER.md) for `nohup`, `screen`, and `tmux` instructions.

---

## Troubleshooting

### Permission Denied on docker-run.sh

Use `bash docker-run.sh` instead of `./docker-run.sh` (common on shared filesystems).

### GPU Not Detected

1. Verify NVIDIA runtime: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
2. Check Docker daemon config includes NVIDIA runtime
3. Verify GPU accessible on host: `nvidia-smi`

### Container Can't Find Data

1. Verify files exist in `./data/raw/` on host
2. Check mounts: `docker inspect <container_id> | grep Mounts`
3. Inside container: `ls -la /workspace/data/`

### Out of Memory

- Reduce batch sizes in code
- Use CPU-only mode (see [CPU-Only Usage](#cpu-only-usage))
- Increase Docker Desktop memory limit (if using Desktop)

### Image Build Fails

1. Check internet connection
2. Clear cache: `docker builder prune`
3. Rebuild: `docker build --no-cache -t prism:latest .`

### Port Already in Use

```bash
# Use different port
docker run -p 8889:8888 ...

# Or stop conflicting container
docker ps
docker stop <container_id>
```

---

## VSCode Remote Development

For advanced workflows including SSH to remote servers, devcontainer attachment, and long-running jobs, see:

**[docs/VSCODE_REMOTE_DOCKER.md](VSCODE_REMOTE_DOCKER.md)**

This covers:
- SSH configuration to DGX/remote servers
- Attaching VSCode to running containers
- Running jobs that survive disconnection (nohup, screen, tmux)
- Git operations inside containers
- Daily workflow for remote development

---

## Environment Variables

The container sets:
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `NVIDIA_VISIBLE_DEVICES=all` - All GPUs visible (when GPU enabled)

Override or add variables:
```bash
docker run -e MY_VAR=value ...
```

Or in `docker-compose.yml`:
```yaml
environment:
  - MY_VAR=value
```

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PRiSM Main README](../README.md)
