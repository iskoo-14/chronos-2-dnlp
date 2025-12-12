#!/bin/bash
set -e

echo "============================================"
echo "DNLP PROJECT - GENERATING PLOTS"
echo "============================================"

source .venv/bin/activate
export PYTHONPATH=$(pwd)

python src/visualization/generate_plots.py

echo "[DONE] Plots generated"
