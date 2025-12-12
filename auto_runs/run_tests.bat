@echo off
echo ============================================================
echo DNLP PROJECT - RUNNING ROBUSTNESS TESTS
echo ============================================================

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Setting PYTHONPATH=%CD%
set PYTHONPATH=%CD%

echo ------------------------------------------------------------
echo [STEP 1] Running pytest on test_robustness.py
echo ------------------------------------------------------------
pytest tests/test_robustness.py

echo ------------------------------------------------------------
echo [DONE] Robustness tests completed.
echo Generated files:
echo   - noise_output.csv
echo   - shuffle_output.csv
echo   - missing_future_output.csv
echo ------------------------------------------------------------

echo Press any key to exit.
exit /b