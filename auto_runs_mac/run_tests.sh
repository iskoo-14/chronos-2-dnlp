#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - RUNNING ROBUSTNESS TESTS"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH="$(pwd)"

echo "------------------------------------------------------------"
echo "[STEP 1] Running pytest on test_robustness.py"
echo "------------------------------------------------------------"
pytest tests/test_robustness.py

echo "------------------------------------------------------------"
echo "[DONE] Robustness tests completed."
echo "Generated files (outputs/):"
echo "  - noise_output.csv"
echo "  - shuffle_output.csv"
echo "  - missing_future_output.csv"
echo "------------------------------------------------------------"
