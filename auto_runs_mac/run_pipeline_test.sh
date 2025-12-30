#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - PIPELINE + ROBUSTNESS TESTS"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "------------------------------------------------------------"
echo "[STEP 1] Running main pipeline"
echo "------------------------------------------------------------"
python main.py

echo "------------------------------------------------------------"
echo "[STEP 2] Running robustness tests"
echo "------------------------------------------------------------"
pytest tests/test_robustness.py

echo "------------------------------------------------------------"
echo "[DONE] Pipeline + tests completed."
echo "Final outputs in /outputs"
echo "------------------------------------------------------------"
