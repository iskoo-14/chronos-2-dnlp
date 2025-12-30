@echo off
echo ============================================================
echo DNLP PROJECT - FULL PIPELINE (main + evaluation + plots)
echo ============================================================

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate
set PYTHONPATH=%CD%

echo ------------------------------------------------------------
echo [STEP 1] Running main pipeline
echo ------------------------------------------------------------
python main.py
if errorlevel 1 (
    echo [ERROR] main.py failed. Aborting.
    exit /b 1
)

echo ------------------------------------------------------------
echo [STEP 2] Aggregating results (compare_results)
echo ------------------------------------------------------------
python src\evaluation\compare_results.py
if errorlevel 1 (
    echo [ERROR] compare_results failed. Aborting.
    exit /b 1
)

echo ------------------------------------------------------------
echo [STEP 3] Generating plots (sampled stores only)
echo ------------------------------------------------------------
python src\visualization\generate_plots.py

echo ------------------------------------------------------------
echo [DONE] Full pipeline completed.
echo Outputs in /outputs, reports in /reports
echo ------------------------------------------------------------
exit /b
