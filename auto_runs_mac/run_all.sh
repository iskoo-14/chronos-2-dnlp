#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - FULL PIPELINE (main + evaluation + plots)"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "------------------------------------------------------------"
echo "[STEP 1] Running main pipeline"
echo "------------------------------------------------------------"
python main.py
if [ $? -ne 0 ]; then
  echo "[ERROR] main.py failed. Aborting."
  exit 1
fi

echo "------------------------------------------------------------"
echo "[STEP 2] Aggregating results (compare_results)"
echo "------------------------------------------------------------"
python src/evaluation/compare_results.py
if [ $? -ne 0 ]; then
  echo "[ERROR] compare_results failed. Aborting."
  exit 1
fi

echo "------------------------------------------------------------"
echo "[STEP 3] Generating plots (sampled stores only)"
echo "------------------------------------------------------------"
python src/visualization/generate_plots.py

echo "------------------------------------------------------------"
echo "[DONE] Full pipeline completed."
echo "Outputs in /outputs, reports in /reports"
echo "------------------------------------------------------------"
