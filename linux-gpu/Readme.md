# Ubuntu A6000 GPU Setup

This folder contains an Ubuntu-focused setup profile for running `geometric-v1` on an NVIDIA RTX A6000 or similar 48 GB/50 GB-class VRAM GPU server.

Files:

```text
linux-gpu/
  Readme.md
  install_linux_a6000.sh
  constraints-a6000.txt
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
- You do not need root access.
- You do not need `sudo`.
- You do not need `apt`.
- `git` is available on the server path. The FLUX.2 Klein pipeline currently requires installing Diffusers from the Hugging Face GitHub branch.
- Stable/final Python 3.11 is used for this project. If Python 3.11 is missing or the server only has a pre-release build such as `3.11.0rc1`, the install script can create a local Python 3.11 environment through micromamba in your home directory.

The A6000 has 48 GB VRAM, so the configs in this folder use a larger diffusion size than the laptop defaults while still avoiding aggressive multi-combo GPU parallelism.

## One-Command Install

From the repository root:

```bash
bash linux-gpu/install_linux_a6000.sh
```

The script will:

1. Use an existing Python 3.11 if available.
2. If Python 3.11 is missing, download micromamba into `~/.local/bin` and create a local Python 3.11 environment.
3. Create or reuse `.venv-linux-gpu`.
4. Install CUDA-enabled PyTorch with pip.
5. Install core `requirements.txt` with pip while constraining the already-installed PyTorch wheel set and the Linux A6000 dependency graph.
6. Install the final `typing-extensions` override needed by PyTorch.
7. Install UI/backend dependencies from `requirements-ui.txt`.
8. Verify PyTorch CUDA visibility and TensorFlow import.

The script does not run `sudo`, `apt`, or any root-level install command.

The installer protects `torch`, `torchvision`, and `torchaudio` after installing them. This matters because otherwise `pip install -r requirements.txt` can replace a working CUDA PyTorch build with a different PyPI Torch build, leaving mismatched packages such as `torchvision` requiring one Torch version while `torch` has been downgraded.

The installer also uses `linux-gpu/constraints-a6000.txt` to keep pip from spending a long time backtracking across `transformers`, `huggingface-hub`, `scipy`, TensorFlow, DeepFace, and related packages. The Torch packages are intentionally not pinned in that file because the installer writes their exact installed versions to a temporary constraints file at runtime.

If you need to test a custom constraints file:

```bash
LINUX_GPU_CONSTRAINTS=path/to/constraints.txt bash linux-gpu/install_linux_a6000.sh
```

If the static constraints ever become stale and you want to temporarily disable them:

```bash
LINUX_GPU_CONSTRAINTS= bash linux-gpu/install_linux_a6000.sh
```

Dependency note: `albumentations` is pinned to `1.3.1` in the project requirements. Do not upgrade it to `2.x` while using `tensorflow==2.12.1`; `albumentations` 2.x requires `numpy>=1.24.4`, but TensorFlow 2.12.1 requires `numpy<=1.24.3`.

FLUX.2 Klein note: `requirements.txt` installs Diffusers with `git+https://github.com/huggingface/diffusers.git` so `Flux2KleinPipeline` is available. If `git` is not installed on the managed server image, ask the cluster/admin team for a module or image that includes Git, or install the dependency from another environment that already has Git.

Default PyTorch wheel target:

```bash
PYTORCH_CUDA=cu118
```

This is the conservative default for the observed server driver profile, `NVIDIA-SMI 550.107.02` with reported CUDA `12.4`. You can override it if the server image is updated or already has a tested PyTorch build.

Override examples:

```bash
PYTORCH_CUDA=cu126 bash linux-gpu/install_linux_a6000.sh
PYTORCH_CUDA=cu118 bash linux-gpu/install_linux_a6000.sh
SKIP_TORCH=1 bash linux-gpu/install_linux_a6000.sh
USE_MICROMAMBA_IF_NEEDED=0 bash linux-gpu/install_linux_a6000.sh
```

Use `SKIP_TORCH=1` if your server image already has a known-good CUDA PyTorch build.

## Manual No-Root Install With Existing Python 3.11

```bash
python3.11 -m venv .venv-linux-gpu
source .venv-linux-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python - <<'PY' > /tmp/geometric_torch_constraints.txt
from importlib.metadata import version
for package in ("torch", "torchvision", "torchaudio"):
    print(f"{package}=={version(package)}")
PY
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118 python -m pip install -r requirements.txt \
  -c /tmp/geometric_torch_constraints.txt \
  -c linux-gpu/constraints-a6000.txt
python -m pip install "typing-extensions>=4.14,<5"
python -m pip install -r requirements-ui.txt
```

If `python3.11 -m venv` fails because the server Python was built without venv support, run the script normally and let it use micromamba:

```bash
bash linux-gpu/install_linux_a6000.sh
```

## Manual No-Root Install With Micromamba

Use this if the server does not have Python 3.11:

```bash
mkdir -p ~/.local/bin
curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba.tar.bz2
tar -xjf /tmp/micromamba.tar.bz2 -C /tmp
mv /tmp/bin/micromamba ~/.local/bin/micromamba
chmod +x ~/.local/bin/micromamba

export MAMBA_ROOT_PREFIX="$HOME/.local/micromamba"
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba create -y -p "$PWD/.venv-linux-gpu" -c conda-forge \
  python=3.11 pip setuptools wheel
micromamba activate "$PWD/.venv-linux-gpu"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python - <<'PY' > /tmp/geometric_torch_constraints.txt
from importlib.metadata import version
for package in ("torch", "torchvision", "torchaudio"):
    print(f"{package}=={version(package)}")
PY
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118 python -m pip install -r requirements.txt \
  -c /tmp/geometric_torch_constraints.txt \
  -c linux-gpu/constraints-a6000.txt
python -m pip install "typing-extensions>=4.14,<5"
python -m pip install -r requirements-ui.txt
```

Verify:

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("cuda", torch.version.cuda)
PY

CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
import tensorflow as tf

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

The script does not install or replace NVIDIA drivers because that requires admin access. On managed GPU servers, the driver is usually already installed by the image/provider. Check first:

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
