#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - PIPELINE + TESTS"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH=$(pwd)

python main.py
pytest -q
