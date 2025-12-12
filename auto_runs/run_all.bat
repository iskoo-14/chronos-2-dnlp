@echo off
echo ============================================================
echo DNLP Project - Run Pipeline + All Tests
echo ============================================================

IF NOT EXIST .venv (
    echo ERROR: No virtual environment found.
    echo Run setup.bat or make env first.
    pause
    exit /b
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Adding project root to PYTHONPATH...
set PYTHONPATH=%CD%

echo ------------------------------------------------------------
echo Running forecasting pipeline (main.py)...
echo ------------------------------------------------------------
python main.py

IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: main.py failed. Tests aborted.
    pause
    exit /b
)

echo ------------------------------------------------------------
echo Running environment tests...
echo ------------------------------------------------------------
pytest tests/test_environment.py -q

echo ------------------------------------------------------------
echo Running robustness tests...
echo ------------------------------------------------------------
pytest tests/test_robustness.py -q

echo ------------------------------------------------------------
echo All tasks completed successfully.
echo Check output files in: outputs/
echo ------------------------------------------------------------

pause
