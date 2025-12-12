@echo off
echo ============================================================
echo DNLP Project - Evaluation & Comparison Pipeline
echo ============================================================

REM ------------------------------------------------------------
REM  Check for virtual environment
REM ------------------------------------------------------------
IF NOT EXIST .venv (
    echo ERROR: Virtual environment not found.
    echo Run setup.bat or make env first.
    pause
    exit /b
)

echo Activating Python virtual environment...
call .venv\Scripts\activate

echo Adding project root to PYTHONPATH...
set PYTHONPATH=%CD%

REM ------------------------------------------------------------
REM  Run comparison analysis
REM ------------------------------------------------------------
echo ------------------------------------------------------------
echo Running comparison analysis (compare_results.py)...
echo ------------------------------------------------------------
cmd /k python src/evaluation/compare_results.py

REM ------------------------------------------------------------
REM  Run visualization script (optional)
REM ------------------------------------------------------------
IF EXIST src\visualization\generate_plots.py (
    echo ------------------------------------------------------------
    echo Generating plots (generate_plots.py)...
    echo ------------------------------------------------------------
    cmd /k python src/visualization/generate_plots.py
) ELSE (
    echo WARNING: src/visualization/generate_plots.py not found.
    echo Skipping plotting step.
)

echo ============================================================
echo Evaluation completed.
echo Results saved in:
echo   outputs/comparison_report.txt
echo   outputs/figures/
echo ============================================================

pause
