#!/bin/bash
# Helper script to run PRiSM in Docker container

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}PRiSM Docker Runner${NC}"
echo "===================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA GPU support may not be available. Continuing anyway..."
    GPU_FLAG=""
    EXTRA_FLAGS=""
else
    GPU_FLAG="--gpus all"
    echo -e "${GREEN}OK${NC} NVIDIA GPU support detected"

    # Check and enable GPU persistence mode for long-running jobs
    # Persistence mode keeps the NVIDIA driver loaded, preventing connection loss
    # during extended runs (12+ hours). Uses ~5-15W extra power per GPU at idle.
    PERSISTENCE_STATUS=$(nvidia-smi -q | grep "Persistence Mode" | head -1 | awk '{print $NF}')
    if [ "$PERSISTENCE_STATUS" = "Disabled" ]; then
        echo -e "${BLUE}INFO${NC} GPU persistence mode is disabled"
        if [ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null; then
            echo "Enabling GPU persistence mode for stable long runs..."
            sudo nvidia-smi -pm 1 2>/dev/null || nvidia-smi -pm 1 2>/dev/null || true
            echo -e "${GREEN}OK${NC} GPU persistence mode enabled"
        else
            echo -e "Warning: Cannot enable GPU persistence mode (requires sudo)"
            echo "         For long runs (12+ hours), manually run: sudo nvidia-smi -pm 1"
        fi
    else
        echo -e "${GREEN}OK${NC} GPU persistence mode already enabled"
    fi

    # Detect multi-GPU systems (e.g., DGX Station A100 with 4 GPUs)
    # Add optimizations for systems with multiple discrete GPUs
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo -e "${GREEN}OK${NC} Multi-GPU system detected ($GPU_COUNT GPUs)"
        # Memory lock and stack size optimizations for multi-GPU training
        EXTRA_FLAGS="--ulimit memlock=-1 --ulimit stack=67108864"
    else
        EXTRA_FLAGS=""
    fi
fi

# Get the project directory (parent of this script)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if image exists, if not build it
if ! docker images | grep -q "prism.*latest"; then
    echo "Building Docker image..."
    docker build -t prism:latest "$PROJECT_DIR"
fi

# Run the container
echo "Starting container..."
docker run $GPU_FLAG $EXTRA_FLAGS -it --rm --ipc=host --pid=host \
    -v "$PROJECT_DIR":/workspace \
    -v "$PROJECT_DIR/data":/workspace/data \
    -v "$PROJECT_DIR/models":/workspace/models \
    -w /workspace \
    prism:latest \
    "$@"
