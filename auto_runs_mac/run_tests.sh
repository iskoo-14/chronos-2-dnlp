#!/bin/bash
set -e

echo "============================================"
echo "DNLP PROJECT - RUNNING ROBUSTNESS TESTS"
echo "============================================"

source .venv/bin/activate
export PYTHONPATH=$(pwd)

pytest tests/test_robustness.py

echo "[DONE] Robustness tests completed"
