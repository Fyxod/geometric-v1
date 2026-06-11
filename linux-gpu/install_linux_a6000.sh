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
LINUX_GPU_CONSTRAINTS="${LINUX_GPU_CONSTRAINTS:-linux-gpu/constraints-a6000.txt}"
SKIP_TORCH="${SKIP_TORCH:-0}"
NO_VENV="${NO_VENV:-0}"
USE_MICROMAMBA_IF_NEEDED="${USE_MICROMAMBA_IF_NEEDED:-1}"
INSTALL_LPIPS="${INSTALL_LPIPS:-0}"
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
is_stable_311 = sys.version_info[:2] == (3, 11) and sys.version_info.releaselevel == "final"
raise SystemExit(0 if is_stable_311 else 1)
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

write_torch_constraints() {
  local constraints_path="$1"
  python - "${constraints_path}" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

path = sys.argv[1]
packages = ("torch", "torchvision", "torchaudio")
with open(path, "w", encoding="utf-8") as handle:
    for package in packages:
        try:
            handle.write(f"{package}=={version(package)}\n")
        except PackageNotFoundError:
            pass
PY
}

write_core_requirements_without_accelerate() {
  local requirements_path="$1"
  python - "${requirements_path}" <<'PY'
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
lines = []
for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped.startswith("accelerate"):
        continue
    lines.append(line)
output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

install_requirements() {
  log "Installing core project requirements with pip"
  if ! command -v git >/dev/null 2>&1; then
    die "git is required because requirements.txt installs the current Diffusers main branch for Flux2KleinPipeline."
  fi
  core_requirements="$(mktemp)"
  torch_constraints="$(mktemp)"
  write_core_requirements_without_accelerate "${core_requirements}"
  write_torch_constraints "${torch_constraints}"
  log "Protecting installed PyTorch packages while resolving project requirements"
  cat "${torch_constraints}"
  log "Installing core requirements first; accelerate is installed after the TensorFlow/PyTorch typing-extensions handoff"

  constraints_args=(-c "${torch_constraints}")
  if [[ -n "${LINUX_GPU_CONSTRAINTS}" ]]; then
    if [[ ! -f "${LINUX_GPU_CONSTRAINTS}" ]]; then
      rm -f "${torch_constraints}" "${core_requirements}"
      die "LINUX_GPU_CONSTRAINTS points to a missing file: ${LINUX_GPU_CONSTRAINTS}"
    fi
    log "Using Linux GPU dependency constraints: ${LINUX_GPU_CONSTRAINTS}"
    constraints_args+=(-c "${LINUX_GPU_CONSTRAINTS}")
  fi

  if ! PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/${PYTORCH_CUDA}" python -m pip install -r "${core_requirements}" "${constraints_args[@]}"; then
    rm -f "${torch_constraints}" "${core_requirements}"
    cat >&2 <<'EOF'

The dependency install failed. On no-root servers, the most common cause is a pip
resolver conflict or a server image with incompatible preinstalled packages.

Options:
  1. Use the default micromamba fallback:
       USE_MICROMAMBA_IF_NEEDED=1 bash linux-gpu/install_linux_a6000.sh

  2. Make sure your repo is updated, then rerun:
       git pull
       bash linux-gpu/install_linux_a6000.sh

  3. If the static Linux constraints become stale, update them or temporarily
     disable them:
       LINUX_GPU_CONSTRAINTS= bash linux-gpu/install_linux_a6000.sh

EOF
    exit 1
  fi
  rm -f "${core_requirements}" "${torch_constraints}"

  python -m pip install "typing-extensions>=4.14,<5"
  log "Installing Accelerate after restoring PyTorch-compatible typing-extensions"
  post_constraints_args=()
  if [[ -n "${LINUX_GPU_CONSTRAINTS}" ]]; then
    post_constraints_args+=(-c "${LINUX_GPU_CONSTRAINTS}")
  fi
  python -m pip install "psutil" "${post_constraints_args[@]}"
  python -m pip install --no-deps "accelerate>=0.30" "${post_constraints_args[@]}"
  log "Installing UI/backend requirements with pip"
  python -m pip install -r requirements-ui.txt

  if [[ "${INSTALL_LPIPS}" == "1" ]]; then
    log "Installing optional LPIPS metric package without reopening dependency resolution"
    python -m pip install --no-deps "lpips>=0.1.4"
  else
    log "Skipping optional LPIPS install. Set INSTALL_LPIPS=1 to enable loss.json objective.beta.use_lpips."
  fi
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
  log "Verifying PyTorch and CUDA"
  python - <<'PY'
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("torch cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch cuda version", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total_vram_gb", round(props.total_memory / (1024 ** 3), 2))
PY

  log "Verifying TensorFlow import separately with CUDA hidden"
  CUDA_VISIBLE_DEVICES="-1" python - <<'PY'
import tensorflow as tf

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
  python -m geometric_v1.loss_pipeline --config loss.json

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
