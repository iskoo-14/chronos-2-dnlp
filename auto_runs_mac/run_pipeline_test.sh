#!/bin/bash

echo "================================================"
echo "DNLP PROJECT - PIPELINE + ROBUSTNESS TESTS"
echo "================================================"

source .venv/bin/activate

python main.py
pytest tests/test_robustness.py -q
