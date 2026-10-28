#!/usr/bin/env bash
# Builds .venv on the Jetson: NVIDIA's torch wheel, matching torchvision, ultralytics, then runs preflight.
# Safe to rerun. Every command is printed before it runs.
set -e
cd "$(dirname "$0")"

TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
TORCHVISION="torchvision==0.15.1"

run() { echo "+ $*"; "$@"; }

# 1. venv tooling (only apt use besides libopenblas below)
if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -m pip --version >/dev/null 2>&1; then
    run sudo apt-get install -y python3-venv python3-pip
fi
[ -d .venv ] || run python3 -m venv .venv
PIP=".venv/bin/python -m pip"
run $PIP install --upgrade pip wheel

# 2. torch: NVIDIA wheel, never from PyPI (that is a CPU-only aarch64 build). Skipped when already installed.
if ! .venv/bin/python -c "import torch" >/dev/null 2>&1; then
    # the NVIDIA wheel links libopenblas.so.0
    if ! ldconfig -p | grep -q libopenblas.so.0; then
        echo "libopenblas missing; the torch wheel needs it"
        run sudo apt-get install -y libopenblas-dev
    fi
    run $PIP install "numpy==1.24.4"
    run $PIP install "$TORCH_WHEEL"
fi

# 3. torchvision: no NVIDIA wheel for JetPack 5, so the PyPI release matching torch 2.0.0, without dependencies.
#    (Fallback if the import below fails: build from source, see README.)
if ! .venv/bin/python -c "import torchvision" >/dev/null 2>&1; then
    run $PIP install --no-deps "$TORCHVISION"
fi

# 4. ultralytics without dependencies, then only the packages it needs; numpy stays below 1.25
ULTRA=$(grep '^ultralytics==' requirements-jetson.txt)
run $PIP install --no-deps "$ULTRA"
grep -v -e '^#' -e '^$' -e '^ultralytics==' requirements-jetson.txt > .rest_requirements.txt
run $PIP install -r .rest_requirements.txt
rm -f .rest_requirements.txt

# 5. make sure nothing replaced NVIDIA's torch
.venv/bin/python -c "import torch; assert 'nv' in torch.__version__, 'torch was replaced by: ' + torch.__version__; print('torch', torch.__version__)"

echo
echo "+ preflight (needs sudo for the power rails)"
sudo .venv/bin/python 00_preflight.py
