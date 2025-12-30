#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - GENERATING PLOTS"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "------------------------------------------------------------"
echo "[STEP] Generating forecast plots (generate_plots.py)"
echo "------------------------------------------------------------"
python src/visualization/generate_plots.py 

echo "------------------------------------------------------------"
echo "[DONE] Figures generated in outputs/figures"
echo "------------------------------------------------------------"
