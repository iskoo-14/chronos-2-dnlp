@echo off
echo ============================================================
echo DNLP PROJECT - EVALUATION
echo ============================================================

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo ------------------------------------------------------------
echo [STEP 1] Comparing results (compare_results.py)
echo ------------------------------------------------------------
python src\evaluation\compare_results.py

echo ------------------------------------------------------------
echo [STEP 2] Generating plots from CSV outputs
echo ------------------------------------------------------------
python src\visualization\generate_plots.py

echo ------------------------------------------------------------
echo [DONE] Evaluation complete.
echo Files generated:
echo   - outputs\comparison_report.txt
echo   - outputs\figures\*.png
echo ------------------------------------------------------------
exit /b
