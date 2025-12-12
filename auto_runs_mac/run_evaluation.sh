#!/bin/bash
set -e

echo "============================================"
echo "DNLP PROJECT - EVALUATION"
echo "============================================"

source .venv/bin/activate
export PYTHONPATH=$(pwd)

echo "[STEP 1] Comparing results"
python src/evaluation/compare_results.py

echo "[STEP 2] Generating plots"
python src/visualization/generate_plots.py

echo "[DONE] Evaluation complete"
