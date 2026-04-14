# Dockerfile for PRiSM project using NVIDIA PyTorch container
# Works on DGX Spark, DGX Station A100, and other NVIDIA GPU systems
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy project files (needed for editable install from pyproject.toml)
COPY . /workspace/

# Install Python dependencies
# Note: PyTorch is already installed in the base image (CUDA-enabled).
# We install in two steps to avoid pip replacing the pre-installed CUDA torch:
#   1. Editable install with --no-deps (registers the package without touching deps)
#   2. Read deps from pyproject.toml, filter out torch, install the rest
# cupy-cuda13x enables GPU-optimized XGBoost inference (zero-copy GPU arrays)
RUN pip install --no-deps -e ".[notebooks]" && \
    python -c "\
import tomllib, pathlib; \
d = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); \
deps = d['project']['dependencies'] + d['project']['optional-dependencies'].get('notebooks', []); \
open('/tmp/deps.txt','w').write('\n'.join(x for x in deps if 'torch' not in x.lower()))" && \
    pip install -r /tmp/deps.txt && \
    rm /tmp/deps.txt && \
    pip install "cupy-cuda13x>=13,<14"

# Set default command
CMD ["/bin/bash"]
