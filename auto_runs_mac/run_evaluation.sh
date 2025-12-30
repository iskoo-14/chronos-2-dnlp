#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - EVALUATION"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "------------------------------------------------------------"
echo "[STEP 1] Comparing results (compare_results.py)"
echo "------------------------------------------------------------"
python src/evaluation/compare_results.py

echo "------------------------------------------------------------"
echo "[DONE] Evaluation complete."
echo "Files generated:"
echo "  - outputs/comparison_report.txt"
echo "  - reports/wql_per_store.csv"
echo "  - reports/wql_by_context.csv"
echo "------------------------------------------------------------"
