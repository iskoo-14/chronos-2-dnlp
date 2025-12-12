#!/bin/bash
set -e

echo "============================================"
echo "DNLP PROJECT - PIPELINE + TESTS"
echo "============================================"

source .venv/bin/activate
export PYTHONPATH=$(pwd)

python main.py
pytest tests/test_robustness.py

echo "[DONE] Pipeline + tests completed"
