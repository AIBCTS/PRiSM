# VSCode Remote Development with Docker

This guide covers advanced workflows for developing PRiSM on remote servers using VSCode with Docker containers.

> **Prerequisite:** Basic Docker setup covered in [README_DOCKER.md](README_DOCKER.md). Complete that guide first.

---

## Overview

**Workflow:**
1. VSCode connects to remote server via SSH
2. Start/attach to Docker container on remote
3. Develop with full GPU access inside container

**Best for:** DGX systems, remote GPU servers, persistent development environments.

---

## SSH Configuration

### Add SSH Config Entry

On your local machine, edit `~/.ssh/config`:

```
# Windows: C:\Users\<username>\.ssh\config
# Linux/Mac: ~/.ssh/config

Host dgx-a100
    HostName your-server.example.com
    User your_username
    Port 22
    ForwardAgent yes
```

### SSH Key Authentication (Recommended)

```bash
# Generate key if needed
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy to server (adjust for your config)
ssh-copy-id -i ~/.ssh/id_ed25519.pub your_username@your-server.example.com
```

### Test Connection

```bash
ssh dgx-a100
```

---

## Connect VSCode to Remote Server

1. Install **Remote Development Extension Pack** in VSCode
2. Press `Ctrl+Shift+P` -> "Remote-SSH: Connect to Host..."
3. Select your configured host (e.g., `dgx-a100`)
4. New VSCode window opens connected to the server

---

## Start a Persistent Development Container

On the remote server (via VSCode terminal):

```bash
cd ~/PRiSM

# Start container in background (persists after disconnect)
# Note: --ulimit flags are for NVIDIA multi-GPU systems (DGX, etc.)
#   memlock=-1: removes memory lock limits (NCCL multi-GPU IPC)
#   stack=67108864: increases stack to 64MB (NCCL graph algorithm)
# For CPU-only or single-GPU, these can be omitted.
docker run --gpus all -d --name prism-dev \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd)":/workspace \
    -w /workspace \
    prism:latest \
    tail -f /dev/null

# Verify running
docker ps
```

The `tail -f /dev/null` keeps the container running indefinitely.

---

## Attach VSCode to Container

### Method A: Command Palette

1. Press `Ctrl+Shift+P`
2. Type "Dev Containers: Attach to Running Container..."
3. Select `prism-dev`
4. Open `/workspace` when prompted

### Method B: Docker Extension

1. Install Docker extension in VSCode (on remote)
2. Click Docker icon in sidebar
3. Right-click `prism-dev` under Containers
4. Select "Attach Visual Studio Code"

---

## Running Long Jobs (Survive Disconnection)

For jobs that should continue after closing VSCode:

### nohup (Simplest)

```bash
# Start job
nohup python ./run_prism_pipeline.py htx_example > pipeline.log 2>&1 &

# Check status
ps aux | grep python
tail -f pipeline.log

# Later, view results
cat pipeline.log
```

### screen (More Flexible)

```bash
# Start session
screen -S pipeline

# Run job (you see output)
python ./run_prism_pipeline.py htx_example

# Detach: Ctrl+A, then D

# Later, reattach
screen -ls        # List sessions
screen -r pipeline
```

### tmux (Alternative)

```bash
tmux new -s pipeline
python ./run_prism_pipeline.py htx_example
# Detach: Ctrl+B, then D

tmux attach -t pipeline
```

### Graceful Shutdown

To stop a running pipeline cleanly:

```bash
# Find process
ps aux | grep run_prism_pipeline.py

# Graceful shutdown (SIGINT)
pkill -2 -f "run_prism_pipeline.py"
# Or by PID: kill -2 <PID>
```

---

## Git Operations Inside Container

### Fix Ownership Warning

First time using git in container:
```bash
git config --global --add safe.directory /workspace
```

### Pull Latest Changes

```bash
git pull origin main

# If prism package changed
pip install -e .
```

### Rebuild After Dependency Changes

If `requirements.txt` changed:
```bash
# On host (not in container)
cd ~/PRiSM
docker build -t prism:latest .

# Restart container (ulimit flags are for NVIDIA multi-GPU systems; omit for CPU-only)
docker stop prism-dev && docker rm prism-dev
docker run --gpus all -d --name prism-dev \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd)":/workspace -w /workspace \
    prism:latest tail -f /dev/null
```

---

## Typical Workflow

### Starting a Session

```bash
# SSH to server
ssh dgx-a100

# Start or restart container (ulimit flags are for NVIDIA multi-GPU; omit for CPU-only)
docker start prism-dev 2>/dev/null || \
docker run --gpus all -d --name prism-dev \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd)":/workspace -w /workspace \
    prism:latest tail -f /dev/null
```

Then in VSCode:
1. `Ctrl+Shift+P` -> "Remote-SSH: Connect to Host..." -> your host
2. `Ctrl+Shift+P` -> "Dev Containers: Attach to Running Container..." -> `prism-dev`

### Ending a Session

Just close VSCode. The container keeps running.

To stop container:
```bash
docker stop prism-dev
```

---

## Using devcontainer.json

For one-click container setup, create `.devcontainer/devcontainer.json`:

> **Note:** The `--ulimit` args below are for NVIDIA multi-GPU systems. Remove them (and `--gpus all`) for CPU-only usage.

```json
{
    "name": "PRiSM",
    "image": "prism:latest",
    "runArgs": [
        "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864"
    ],
    "workspaceFolder": "/workspace",
    "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-toolsai.jupyter"
            ]
        }
    }
}
```

Then: Connect to remote -> Open folder -> "Reopen in Container" when prompted.

---

## Port Forwarding

### Automatic

VSCode auto-forwards ports when services start. Check the "Ports" tab in bottom panel.

### Manual

1. Click "Ports" tab in VSCode bottom panel
2. Click "Forward a Port"
3. Enter port (e.g., 8888 for Jupyter)
4. Access via `localhost:8888`

### Jupyter Lab Example

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

VSCode will prompt to open in browser.

---

## Troubleshooting

### Container Not Appearing

```bash
docker ps -a          # List all containers
docker start prism-dev  # Start if stopped
```

### SSH Connection Timeout

```bash
ping your-server.example.com
ssh -v dgx-a100       # Verbose output for debugging
```

### Extensions Not Working in Container

Install extensions specifically in container context:
1. Attach to container
2. Go to Extensions sidebar
3. Look for "Install in Container" button

### Jupyter Notebooks Not Running

```bash
pip install ipykernel
# Then: Ctrl+Shift+P -> "Python: Select Interpreter"
```

---

## References

- [VSCode Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)
- [Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Remote - SSH Documentation](https://code.visualstudio.com/docs/remote/ssh)
