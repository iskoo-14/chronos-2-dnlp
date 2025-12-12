#!/bin/bash

echo "============================================"
echo "DNLP PROJECT - EVALUATION"
echo "============================================"

source .venv/bin/activate

python src/evaluation/compare_results.py
python src/visualization/generate_plots.py
