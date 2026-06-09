#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv-linux-gpu}"
PYTORCH_CUDA="${PYTORCH_CUDA:-cu128}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"
INSTALL_DEADSNAKES="${INSTALL_DEADSNAKES:-1}"
SKIP_TORCH="${SKIP_TORCH:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

need_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

install_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get was not found. Skipping system package install."
    return
  fi

  log "Installing Linux build/runtime packages"
  need_sudo apt-get update
  need_sudo apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libhdf5-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    pkg-config \
    wget

  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    log "${PYTHON_BIN} not found. Installing Python 3.11 packages."
    need_sudo apt-get install -y software-properties-common
    if [[ "${INSTALL_DEADSNAKES}" == "1" ]] && command -v add-apt-repository >/dev/null 2>&1; then
      need_sudo add-apt-repository -y ppa:deadsnakes/ppa
      need_sudo apt-get update
    fi
    need_sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
  fi
}

install_torch() {
  if [[ "${SKIP_TORCH}" == "1" ]]; then
    log "Skipping PyTorch install because SKIP_TORCH=1"
    return
  fi

  case "${PYTORCH_CUDA}" in
    cu128|cu126|cu118)
      log "Installing PyTorch CUDA wheel set: ${PYTORCH_CUDA}"
      python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
      ;;
    cpu)
      log "Installing CPU-only PyTorch wheel set"
      python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
      ;;
    *)
      printf 'Unsupported PYTORCH_CUDA=%s. Use cu128, cu126, cu118, or cpu.\n' "${PYTORCH_CUDA}" >&2
      exit 2
      ;;
  esac
}

verify_gpu() {
  log "Checking NVIDIA visibility"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    log "nvidia-smi not found. Install the NVIDIA driver before running GPU workloads."
  fi
}

main() {
  cd "${REPO_ROOT}"
  log "Repository root: ${REPO_ROOT}"
  verify_gpu

  if [[ "${INSTALL_SYSTEM_PACKAGES}" == "1" ]]; then
    install_apt_packages
  else
    log "Skipping system package install because INSTALL_SYSTEM_PACKAGES=0"
  fi

  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf '%s was not found. Install Python 3.11 or set PYTHON_BIN.\n' "${PYTHON_BIN}" >&2
    exit 2
  fi

  log "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  log "Upgrading pip tooling"
  python -m pip install --upgrade pip setuptools wheel

  install_torch

  log "Installing project requirements"
  CMAKE_ARGS="-DDLIB_USE_CUDA=OFF" python -m pip install -r requirements.txt
  python -m pip install "typing-extensions>=4.14,<5"

  log "Verifying Python, PyTorch, CUDA, TensorFlow, and package imports"
  python - <<'PY'
import sys
import tensorflow as tf
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("torch cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch cuda version", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total_vram_gb", round(props.total_memory / (1024 ** 3), 2))
print("tensorflow", tf.__version__)
PY

  log "Done"
  cat <<EOF

Activate this environment with:
  source ${VENV_DIR}/bin/activate

Run the A6000 profile:
  python -m geometric_v1.pipeline --config linux-gpu/pipeline.json
  python -m geometric_v1.brute_force --config linux-gpu/brute.json
  python -m geometric_v1.batch_brute_force --config linux-gpu/batch_brute.json

EOF
}

main "$@"
