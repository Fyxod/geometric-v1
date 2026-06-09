# Ubuntu A6000 GPU Setup

This folder contains an Ubuntu-focused setup profile for running `geometric-v1` on an NVIDIA RTX A6000 or similar 48 GB/50 GB-class VRAM GPU server.

Files:

```text
linux-gpu/
  Readme.md
  install_linux_a6000.sh
  pipeline.json
  brute.json
  batch_brute.json
  parameters.md
```

## Assumptions

- Ubuntu 20.04, 22.04, or 24.04.
- NVIDIA RTX A6000-class GPU with a working NVIDIA driver.
- `nvidia-smi` works before installing project dependencies.
- You are running from the repository root.
- Python 3.11 is used for this project.

The A6000 has 48 GB VRAM, so the configs in this folder use a larger diffusion size than the laptop defaults while still avoiding aggressive multi-combo GPU parallelism.

## One-Command Install

From the repository root:

```bash
bash linux-gpu/install_linux_a6000.sh
```

The script will:

1. Install Ubuntu apt packages needed for Python builds, OpenCV runtime libraries, CMake, and `dlib`.
2. Install Python 3.11 if it is missing.
3. Create `.venv-linux-gpu`.
4. Install CUDA-enabled PyTorch.
5. Install `requirements.txt`.
6. Install the final `typing-extensions` override needed by PyTorch.
7. Verify PyTorch CUDA visibility and TensorFlow import.

Default PyTorch wheel target:

```bash
PYTORCH_CUDA=cu128
```

Override examples:

```bash
PYTORCH_CUDA=cu126 bash linux-gpu/install_linux_a6000.sh
PYTORCH_CUDA=cu118 bash linux-gpu/install_linux_a6000.sh
SKIP_TORCH=1 bash linux-gpu/install_linux_a6000.sh
INSTALL_SYSTEM_PACKAGES=0 bash linux-gpu/install_linux_a6000.sh
```

Use `SKIP_TORCH=1` if your server image already has a known-good CUDA PyTorch build.

## Manual Install

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake curl git wget pkg-config \
  libgl1 libglib2.0-0 libhdf5-dev libsm6 libxext6 libxrender1 \
  python3.11 python3.11-dev python3.11-venv

python3.11 -m venv .venv-linux-gpu
source .venv-linux-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
CMAKE_ARGS="-DDLIB_USE_CUDA=OFF" python -m pip install -r requirements.txt
python -m pip install "typing-extensions>=4.14,<5"
```

Verify:

```bash
python - <<'PY'
import torch
import tensorflow as tf

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("cuda", torch.version.cuda)
print("tensorflow", tf.__version__)
PY
```

## Run Commands

Full pipeline:

```bash
source .venv-linux-gpu/bin/activate
python -m geometric_v1.pipeline --config linux-gpu/pipeline.json
```

Plain brute force:

```bash
python -m geometric_v1.brute_force --config linux-gpu/brute.json
```

Batch brute force:

```bash
python -m geometric_v1.batch_brute_force --config linux-gpu/batch_brute.json
```

Local UI:

```bash
python run_ui.py --host 0.0.0.0 --port 7860
```

If the server is remote, tunnel the port:

```bash
ssh -L 7860:127.0.0.1:7860 user@server
```

Then open:

```text
http://127.0.0.1:7860
```

## Driver Notes

The script does not install or replace NVIDIA drivers. On managed GPU servers, the driver is usually already installed by the image/provider. Check first:

```bash
nvidia-smi
```

If `nvidia-smi` fails, fix the NVIDIA driver before installing Python dependencies.

## Why Batch Parallelism Defaults To 1

The A6000 has enough VRAM for larger single-combo diffusion work, but batch brute force can start multiple diffusion jobs if `parallel_combinations` is greater than `1`. That can create avoidable CUDA memory spikes. The provided config keeps:

```json
"parallel_combinations": 1
```

Increase it only after a few successful runs and while watching `nvidia-smi`.
