# DNLP Project – Chronos 2 Time Series Forecasting

!!!! NEEDED PYTHON 3.10.X !!!

This project implements a complete zero shot forecasting pipeline using Chronos 2 by Amazon. The goal is to evaluate univariate forecasting, multivariate forecasting with covariates, and the robustness of the model under controlled perturbations. We use the Rossmann Store Sales dataset, which contains daily sales and external features such as promotions and holidays.

All experiments are zero shot. No model training is required.

===============================================================================
Project Structure
===============================================================================

chronos-2-dnlp/
│
├── main.py                        main forecasting pipeline
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py        loading, cleaning, feature engineering
│   │   ├── train.csv              raw input
│   │   └── store.csv              raw input
│   │
│   ├── features/
│   │   └── build_features.py      extracts target and covariates
│   │
│   ├── models/
│       ├── predict_model.py       Chronos 2 inference
│       └── robustness.py          noise, shuffle, missing future tests
│
├── evaluation/
│   └── compare_results.py         compares univariate, covariate, noise, shuffle, missing future
│
└── tests/
    ├── test_environment.py        pytest environment checks
    └── test_robustness.py         pytest robustness tests

Outputs are stored in: outputs/

===============================================================================
1. Data Preparation
===============================================================================

make_dataset.py performs:
- merge of train.csv and store.csv
- removal of days when store was closed
- filling missing values
- creation of time features such as day, month, year, week number
- conversion of mixed types and categorical values

The processed dataset is saved at:
src/data/processed_rossmann.csv

===============================================================================
2. Feature Extraction
===============================================================================

We extract:
- Sales column as target
- all engineered columns as covariates

Covariates are converted into tensors with shape required by Chronos 2:
- past covariates: number_of_features x sequence_length
- future covariates: number_of_features x prediction_horizon

===============================================================================
3. Chronos 2 Model
===============================================================================

Model used:
amazon/chronos-2
https://huggingface.co/amazon/chronos-2

Features:
- works zero shot
- accepts multivariate inputs
- uses past and future covariates
- produces quantile forecasts: p10, median, p90
- scales automatically

===============================================================================
4. Forecasting Outputs
===============================================================================

### Univariate Forecast
Uses only Sales.
Output: outputs/univariate.csv

### Covariate Forecast
Uses Sales plus all covariates.
Output: outputs/covariate.csv

===============================================================================
5. Robustness Experiments
===============================================================================

We evaluate whether Chronos 2 uses covariates meaningfully.

### Noise Test
Adds a purely random noise covariate.
A robust model should ignore it.
Output: outputs/noise_output.csv

### Shuffle Test
Randomly shuffles the Promo column.
Breaks correlation. Forecasts should degrade.
Output: outputs/shuffle_output.csv

### Missing Future Test
Masks future values of SchoolHoliday.
Removes known future information. Accuracy should worsen.
Output: outputs/missing_future_output.csv

===============================================================================
Using the Makefile
===============================================================================

### Create virtual environment
make env

### Install PyTorch with GPU support (optional)
make gpu

### Run main forecasting pipeline
make run

### Run environment tests
make envtest

### Run robustness tests
make robustness

### Run all tests
make test

### Full project run for users who already have an environment
make fullrun

===============================================================================
Running the Project Without Make (Windows)
===============================================================================

Use PowerShell setup script:

powershell -ExecutionPolicy Bypass -File setup.ps1

This script:
- creates the virtual environment
- installs dependencies
- installs GPU specific PyTorch if requested
- verifies Chronos 2 installation
- checks the environment

After setup:
.venv\Scripts\python main.py

To run tests manually:
.venv\Scripts\pytest -q

===============================================================================
Running the Project Using Windows Batch Files
===============================================================================

These files automate the pipeline for Windows users.

All .bat files:
- activate the virtual environment
- set PYTHONPATH automatically
- show clear status messages
- pause at the end so you can read results

If a script ends with:
Press any key to continue...
you can close it by pressing Enter, Space, or any key.

---

### run_main.bat
Runs only the forecasting pipeline.
Double click:
run_main.bat

---

### run_tests.bat
Runs pytest tests:
- environment tests
- robustness tests

Double click:
run_tests.bat

---

### run_pipeline_and_tests.bat
Runs full pipeline and then all robustness tests.
Recommended for full validation.

Double click:
run_pipeline_and_tests.bat

---

### evaluation.bat
Generates:
- comparison_report.txt
- plots in outputs/figures

Double click:
evaluation.bat

---

### run_visuals.bat
Generates only the plots.
Useful if you want figures for the report without recomputing forecasts.

Double click:
run_visuals.bat

---

===============================================================================
Installing Make on Windows (optional)
===============================================================================

Using Chocolatey:
choco install make

Git Bash also includes make.

===============================================================================
Requirements
===============================================================================

All dependencies are listed in requirements.txt.

Install with:
pip install -r requirements.txt

===============================================================================
Notes
===============================================================================

- All forecasts include p10, median, p90 quantiles.
- Chronos 2 runs completely zero shot.
- Robustness tests measure whether the model truly leverages covariates.
- The pipeline is modular and can be extended easily.

