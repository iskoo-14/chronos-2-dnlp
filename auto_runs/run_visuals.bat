@echo off
echo ============================================================
echo DNLP Project - Generate Visualizations
echo ============================================================

IF NOT EXIST .venv (
    echo ERROR: Virtual environment not found.
    echo Run setup.bat or make env first.
    pause
    exit /b
)

IF NOT EXIST src\visualization\generate_plots.py (
    echo ERROR: Visualization script not found:
    echo    src\visualization\generate_plots.py
    echo Create this file before using run_visuals.bat
    pause
    exit /b
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Adding project root to PYTHONPATH...
set PYTHONPATH=%CD%

echo ------------------------------------------------------------
echo Running visualization script...
echo ------------------------------------------------------------
python src/visualization/generate_plots.py

echo ------------------------------------------------------------
echo Visualization complete.
echo Figures saved inside: outputs/figures/
echo ------------------------------------------------------------

pause
