#!/bin/bash
echo "============================================================"
echo "DNLP PROJECT - RUNNING TESTS"
echo "============================================================"

source .venv/bin/activate
export PYTHONPATH=$(pwd)

pytest -q
