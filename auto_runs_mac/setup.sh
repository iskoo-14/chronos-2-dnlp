#!/bin/bash

echo "============================================"
echo "DNLP PROJECT - SETUP (macOS / Linux)"
echo "============================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python >= 3.9"
    exit 1
fi

PY_VERSION=$(python3 - <<EOF
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)

echo "[INFO] Python version: $PY_VERSION"

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "[INFO] Virtual environment already exists"
fi

# Activate venv
echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip tools
echo "[INFO] Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements
echo "[INFO] Installing requirements..."
pip install -r requirements.txt

# Test torch
python - <<EOF
import torch
print("Torch OK:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# Test Chronos
python - <<EOF
from chronos import Chronos2Pipeline
print("Chronos-2 imported successfully")
EOF

echo "============================================"
echo "SETUP COMPLETE"
echo "Run:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo "============================================"
