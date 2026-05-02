#!/usr/bin/env bash
# Bootstrap an Amazon Linux 2023 EC2 instance for the Diagonality 3D
# pipeline. Idempotent — running it twice is safe.
#
# What it does:
#   1. Update OS + install ffmpeg, git, build tools.
#   2. Install miniconda (silent, ~/miniconda3).
#   3. Create conda env 'diag' with Python 3.10.
#   4. pip install -r requirements.txt inside that env.
#   5. Print the activation command so the next shell can run the pipeline.
#
# Assumes: repo already cloned at $REPO_ROOT (default $HOME/Diagonality_3D).
# Data + cache are NOT copied here — they go up via scp from the dev box
# (or are regenerated on AWS by `regenerate_caches.py` if data/ is uploaded).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/Diagonality_3D}"
CONDA_ENV="${CONDA_ENV:-diag}"
PY_VERSION="3.10"
MINICONDA="$HOME/miniconda3"

log() { echo -e "\033[1;32m[setup]\033[0m $*"; }

log "1/5 — system packages"
sudo dnf -y update >/dev/null
sudo dnf -y install \
    ffmpeg \
    git \
    gcc gcc-c++ make \
    bzip2 wget curl tar \
    >/dev/null

log "2/5 — miniconda"
if [[ ! -d "$MINICONDA" ]]; then
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    wget -q -O /tmp/miniconda.sh "$URL"
    bash /tmp/miniconda.sh -b -p "$MINICONDA"
    rm /tmp/miniconda.sh
fi

# shellcheck disable=SC1091
source "$MINICONDA/etc/profile.d/conda.sh"

log "3/5 — conda env '$CONDA_ENV' (python $PY_VERSION)"
if ! conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
    conda create -y -n "$CONDA_ENV" "python=$PY_VERSION" pip >/dev/null
fi
conda activate "$CONDA_ENV"

log "4/5 — pip install requirements"
cd "$REPO_ROOT"
if [[ ! -f requirements.txt ]]; then
    echo "ERROR: requirements.txt not found in $REPO_ROOT"
    exit 1
fi
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

log "5/5 — sanity check"
python3 - <<'PY'
import sys, importlib
mods = ["numpy","pandas","scipy","sklearn","statsmodels","matplotlib","mplsoccer","pyarrow","PIL"]
print(f"python {sys.version.split()[0]}")
for m in mods:
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  {m:14s} {v}")
    except Exception as e:
        print(f"  {m:14s} MISSING ({e})")
PY

log "DONE. To run the pipeline:"
echo "    source $MINICONDA/etc/profile.d/conda.sh"
echo "    conda activate $CONDA_ENV"
echo "    cd $REPO_ROOT"
echo "    python3 scripts/aws_pipeline.py"
