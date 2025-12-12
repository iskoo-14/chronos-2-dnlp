#!/bin/bash

echo "============================================"
echo "DNLP PROJECT - RUN TESTS"
echo "============================================"

source .venv/bin/activate

pytest -q
