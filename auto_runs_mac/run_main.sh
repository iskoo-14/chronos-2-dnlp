#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - MAIN PIPELINE EXECUTION"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "------------------------------------------------------------"
echo "[STEP 1] Running main pipeline (preprocessing + forecasts)"
echo "------------------------------------------------------------"
python main.py

echo "------------------------------------------------------------"
echo "[DONE] Main pipeline completed successfully."
echo "Output files saved in /outputs"
echo "------------------------------------------------------------"
