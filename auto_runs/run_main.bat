@echo off
echo ============================================================
echo DNLP Project - Run Pipeline + All Tests
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
echo Running forecasting pipeline (main.py)...
echo ------------------------------------------------------------
python main.py

echo ------------------------------------------------------------
echo All tasks completed successfully.
echo Check output files in: outputs/
echo ------------------------------------------------------------

pause
