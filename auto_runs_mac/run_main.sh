#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - RUN MAIN PIPELINE"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH=$(pwd)

python main.py
