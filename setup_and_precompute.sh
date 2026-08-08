#!/usr/bin/env bash
# Install project dependencies and precompute sliding-window FC caches.
#
# Usage from the project root:
#   bash setup_and_precompute.sh
#   SCDFC_CONDA_ENV=GCN_mri bash setup_and_precompute.sh  # create/reuse this env
#   SCDFC_WINDOWS="83 42 125" bash setup_and_precompute.sh
#   SCDFC_OVERWRITE=1 bash setup_and_precompute.sh

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${SCDFC_CONDA_ENV:-GCN_mri}"
WINDOWS_TEXT="${SCDFC_WINDOWS:-83}"
read -r -a WINDOWS <<< "${WINDOWS_TEXT}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/setup_precompute_$(date -u +%Y%m%dT%H%M%SZ).log"

# Mirror all output to the terminal and keep a persistent copy for later review.
exec > >(tee -a "${LOG_FILE}") 2>&1

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

trap 'echo "[ERROR] Failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Requested FC window lengths: ${WINDOWS[*]}"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

# The Starlight container manual provides this proxy for package downloads.
# It is loaded only inside this script process.
if [[ -f /app/bin/proxy.sh ]]; then
    echo "[INFO] Loading Starlight container network proxy."
    # shellcheck disable=SC1091
    source /app/bin/proxy.sh
fi

# Create or activate the project Conda environment. The default name follows
# the project's documented environment name, GCN_mri. Set SCDFC_CONDA_ENV to
# use another name.
command -v conda >/dev/null 2>&1 || fail "conda was not found. Load a Conda module or install Conda before running this script."
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk -v env_name="${CONDA_ENV_NAME}" 'NF && $1 !~ /^#/ && $1 == env_name {found=1} END {exit !found}'; then
    echo "[INFO] Reusing Conda environment: ${CONDA_ENV_NAME}"
else
    echo "[INFO] Creating Conda environment: ${CONDA_ENV_NAME} (Python 3.11)."
    conda create -y -n "${CONDA_ENV_NAME}" python=3.11 pip
fi
conda activate "${CONDA_ENV_NAME}"
echo "[INFO] Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"

command -v python >/dev/null 2>&1 || fail "python was not found in PATH."
python - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        f"Python >= 3.11 is required, but this interpreter is {sys.version.split()[0]}"
    )
print(f"[INFO] Python: {sys.version.split()[0]}")
PY

echo "[INFO] Python executable: $(command -v python)"
python -m pip --version

# Conda is created/activated above, so its site-packages are writable without
# --user. This also avoids accidentally installing into the system Python.
PIP_FLAGS=()

echo "[INFO] Installing dependencies from requirements.txt."
python -m pip install "${PIP_FLAGS[@]}" -r "${PROJECT_ROOT}/requirements.txt"

# Install the local package so that the `scdfc` entry point is available.
echo "[INFO] Installing the local scdfc package."
python -m pip install "${PIP_FLAGS[@]}" --no-deps -e "${PROJECT_ROOT}"

echo "[INFO] Checking installed packages."
python - <<'PY'
import importlib

modules = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "torch": "torch",
    "zarr": "zarr",
    "numcodecs": "numcodecs",
}

for module, package in modules.items():
    imported = importlib.import_module(module)
    print(f"[INFO] {package}: {getattr(imported, '__version__', 'installed')}")

import torch

print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
print(f"[INFO] CUDA runtime: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[WARN] CUDA is not available; preprocessing can continue, but use --device cuda only after fixing PyTorch/CUDA.")
PY

OVERWRITE_FLAGS=()
if [[ "${SCDFC_OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_FLAGS+=(--overwrite)
fi

echo "[INFO] Starting sliding-window FC preprocessing."
echo "[INFO] Progress is emitted at the first item, every 25 subject/run items, and the final item."
python -m scdfc.cli precompute \
    --config "${PROJECT_ROOT}/configs/default.yaml" \
    --windows "${WINDOWS[@]}" \
    "${OVERWRITE_FLAGS[@]}"

for window in "${WINDOWS[@]}"; do
    cache_path="${PROJECT_ROOT}/data/cache/dfc/window_${window}.zarr"
    if [[ -d "${cache_path}" ]]; then
        echo "[INFO] Cache created: ${cache_path}"
        du -sh "${cache_path}"
    else
        echo "[WARN] Expected cache directory was not found: ${cache_path}"
    fi
done

echo "[DONE] Setup and FC preprocessing completed."
echo "[DONE] Review the live/replayable log at: ${LOG_FILE}"
