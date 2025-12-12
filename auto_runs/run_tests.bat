@echo off
echo ============================================================
echo DNLP Project - Run All Tests
echo ============================================================

IF NOT EXIST .venv (
    echo ERROR: Virtual environment not found.
    echo Run setup.bat or make env first.
    pause
    exit /b
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Adding project root to PYTHONPATH...
set PYTHONPATH=%CD%

echo ------------------------------------------------------------
echo Running environment tests...
echo ------------------------------------------------------------
pytest tests/test_environment.py -q

echo ------------------------------------------------------------
echo Running robustness tests...
echo ------------------------------------------------------------
pytest tests/test_robustness.py -q

echo ------------------------------------------------------------
echo All tests completed. Check 'outputs/' for results.
echo ------------------------------------------------------------

pause
