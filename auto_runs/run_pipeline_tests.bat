@echo off
echo ============================================================
echo DNLP PROJECT - PIPELINE + ROBUSTNESS TESTS
echo ============================================================

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate
set PYTHONPATH=%CD%

echo ------------------------------------------------------------
echo [STEP 1] Running main pipeline 
echo ------------------------------------------------------------
python main.py

echo ------------------------------------------------------------
echo [STEP 2] Running robustness tests
echo ------------------------------------------------------------
pytest tests/test_robustness.py

echo ------------------------------------------------------------
echo [DONE] Pipeline + tests completed.
echo Final outputs in /outputs
echo ------------------------------------------------------------

echo Press any key to exit.
exit /b
