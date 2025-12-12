@echo off
echo ============================================================
echo DNLP PROJECT - GENERATE VISUAL PLOTS
echo ============================================================

if not exist .venv (
    echo [ERROR] Virtual environment not found.
    echo Run setup.bat or make env first.
    pause >nul
    exit /b
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Setting PYTHONPATH=%CD%
set PYTHONPATH=%CD%

echo ------------------------------------------------------------
echo [STEP] Generating forecast plots (generate_plots.py)
echo ------------------------------------------------------------
python src/visualization/generate_plots.py

echo ------------------------------------------------------------
echo [DONE] Figures generated in outputs/figures
echo Press any key to exit.
echo ------------------------------------------------------------

exit /b
