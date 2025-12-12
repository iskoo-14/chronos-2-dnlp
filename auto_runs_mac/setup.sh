#!/bin/bash
set -e

echo "============================================"
echo "DNLP PROJECT - SETUP (macOS / Linux)"
echo "============================================"

if [ ! -d ".venv" ]; then
  echo "[INFO] Creating virtual environment..."
  python3 -m venv .venv
else
  echo "[INFO] Virtual environment already exists"
fi

echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

echo "[INFO] Installing requirements..."
pip install -r requirements.txt

echo "[DONE] Setup completed successfully"
