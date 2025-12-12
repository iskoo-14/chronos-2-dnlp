#!/bin/bash
set -e

echo "============================================"
echo "DNLP PROJECT - RUN MAIN PIPELINE"
echo "============================================"

source .venv/bin/activate

export PYTHONPATH=$(pwd)

python main.py

echo "[DONE] Main pipeline completed"
