#!/bin/bash
echo "============================================"
echo "DNLP PROJECT - SETUP (macOS / Linux)"
echo "============================================"

# Check Python 3.10
if ! command -v python3.10 &> /dev/null
then
    echo "ERROR: python3.10 not found."
    echo "Install with: brew install python@3.10"
    exit 1
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv)"
    python3.10 -m venv .venv
else
    echo "[INFO] Virtual environment already exists"
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip safely
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

echo "============================================"
echo "SETUP COMPLETE"
echo "To run:"
echo "  source .venv/bin/activate"
echo "  ./auto_runs_mac/run_main.sh"
echo "============================================"
