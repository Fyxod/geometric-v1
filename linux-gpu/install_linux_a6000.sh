#!/usr/bin/env bash
set -euo pipefail

# No sudo. No apt. This script installs into user-writable locations only.
#
# Defaults:
# - Use python3.11 if it exists.
# - Otherwise download micromamba into ~/.local/bin and create Python 3.11 env locally.
# - Install CUDA PyTorch and project requirements with pip.

PYTHON_BIN="${PYTHON_BIN:-auto}"
ENV_DIR="${ENV_DIR:-.venv-linux-gpu}"
PYTORCH_CUDA="${PYTORCH_CUDA:-cu118}"
SKIP_TORCH="${SKIP_TORCH:-0}"
NO_VENV="${NO_VENV:-0}"
USE_MICROMAMBA_IF_NEEDED="${USE_MICROMAMBA_IF_NEEDED:-1}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
MICROMAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT_PREFIX:-$HOME/.local/micromamba}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PATH="${REPO_ROOT}/${ENV_DIR}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 2
}

python_is_311() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
}

find_python311() {
  if [[ "${PYTHON_BIN}" != "auto" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || return 1
    python_is_311 "${PYTHON_BIN}" || return 1
    command -v "${PYTHON_BIN}"
    return
  fi

  for candidate in python3.11 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_is_311 "${candidate}"; then
      command -v "${candidate}"
      return
    fi
  done
  return 1
}

download_micromamba() {
  if [[ -x "${MICROMAMBA_BIN}" ]]; then
    return
  fi

  command -v curl >/dev/null 2>&1 || die "curl is required to download micromamba. Ask the admin for curl, or set PYTHON_BIN to an existing Python 3.11."
  command -v tar >/dev/null 2>&1 || die "tar is required to unpack micromamba. Ask the admin for tar, or set PYTHON_BIN to an existing Python 3.11."

  log "Downloading micromamba into user space: ${MICROMAMBA_BIN}"
  tmp_dir="$(mktemp -d)"
  mkdir -p "$(dirname "${MICROMAMBA_BIN}")"
  curl -L "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "${tmp_dir}/micromamba.tar.bz2"
  tar -xjf "${tmp_dir}/micromamba.tar.bz2" -C "${tmp_dir}"
  mv "${tmp_dir}/bin/micromamba" "${MICROMAMBA_BIN}"
  chmod +x "${MICROMAMBA_BIN}"
  rm -rf "${tmp_dir}"
}

create_micromamba_env() {
  download_micromamba
  export MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT_PREFIX}"

  if [[ ! -d "${ENV_PATH}" ]]; then
    log "Creating local micromamba environment: ${ENV_PATH}"
    "${MICROMAMBA_BIN}" create -y -p "${ENV_PATH}" -c conda-forge \
      python=3.11 \
      pip \
      setuptools \
      wheel
  else
    log "Using existing environment: ${ENV_PATH}"
  fi

  # shellcheck disable=SC1090
  eval "$("${MICROMAMBA_BIN}" shell hook -s bash)"
  micromamba activate "${ENV_PATH}"
}

create_venv() {
  local python_bin="$1"

  if [[ "${NO_VENV}" == "1" ]]; then
    log "NO_VENV=1, using current Python environment"
    return
  fi

  if [[ ! -d "${ENV_PATH}" ]]; then
    log "Creating virtual environment: ${ENV_PATH}"
    "${python_bin}" -m venv "${ENV_PATH}" || die "Could not create venv. If python3.11-venv is unavailable and you do not have root, rerun with USE_MICROMAMBA_IF_NEEDED=1."
  else
    log "Using existing virtual environment: ${ENV_PATH}"
  fi

  # shellcheck disable=SC1091
  source "${ENV_PATH}/bin/activate"
}

install_torch() {
  if [[ "${SKIP_TORCH}" == "1" ]]; then
    log "Skipping PyTorch install because SKIP_TORCH=1"
    return
  fi

  case "${PYTORCH_CUDA}" in
    cu128|cu126|cu118)
      log "Installing PyTorch CUDA wheel set with pip: ${PYTORCH_CUDA}"
      python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
      ;;
    cpu)
      log "Installing CPU-only PyTorch wheel set with pip"
      python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
      ;;
    *)
      die "Unsupported PYTORCH_CUDA=${PYTORCH_CUDA}. Use cu128, cu126, cu118, or cpu."
      ;;
  esac
}

install_requirements() {
  log "Installing core project requirements with pip"
  if ! command -v git >/dev/null 2>&1; then
    die "git is required because requirements.txt installs the current Diffusers main branch for Flux2KleinPipeline."
  fi
  if ! python -m pip install -r requirements.txt; then
    cat >&2 <<'EOF'

The dependency install failed. On no-root servers, the most common cause is a pip
resolver conflict or a server image with incompatible preinstalled packages.

Options:
  1. Use the default micromamba fallback:
       USE_MICROMAMBA_IF_NEEDED=1 bash linux-gpu/install_linux_a6000.sh

  2. Make sure your repo is updated, then rerun:
       git pull
       bash linux-gpu/install_linux_a6000.sh

EOF
    exit 1
  fi

  python -m pip install "typing-extensions>=4.14,<5"
  log "Installing UI/backend requirements with pip"
  python -m pip install -r requirements-ui.txt
}

verify_gpu() {
  log "Checking NVIDIA visibility"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    log "nvidia-smi not found in PATH. The Python install can continue, but GPU runs need the NVIDIA driver visible."
  fi
}

verify_python_stack() {
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
}

print_activation_instructions() {
  cat <<EOF

Done.

Activate this environment with:
  source ${ENV_PATH}/bin/activate

If this was created by micromamba and normal activation does not work:
  export MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT_PREFIX}"
  eval "\$(${MICROMAMBA_BIN} shell hook -s bash)"
  micromamba activate "${ENV_PATH}"

Run the A6000 profile:
  python -m geometric_v1.pipeline --config linux-gpu/pipeline.json
  python -m geometric_v1.brute_force --config linux-gpu/brute.json
  python -m geometric_v1.batch_brute_force --config linux-gpu/batch_brute.json

EOF
}

main() {
  cd "${REPO_ROOT}"
  log "Repository root: ${REPO_ROOT}"
  verify_gpu

  if python_bin="$(find_python311)"; then
    log "Using Python 3.11: ${python_bin}"
    create_venv "${python_bin}"
  elif [[ "${USE_MICROMAMBA_IF_NEEDED}" == "1" ]]; then
    log "Python 3.11 was not found. Creating a no-root micromamba Python 3.11 environment."
    create_micromamba_env
  else
    die "Python 3.11 was not found and USE_MICROMAMBA_IF_NEEDED=0."
  fi

  log "Upgrading pip tooling"
  python -m pip install --upgrade pip setuptools wheel

  install_torch
  install_requirements
  verify_python_stack
  print_activation_instructions
}

main "$@"
