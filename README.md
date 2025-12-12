# DNLP Project – Chronos-2 Time Series Forecasting

!!!! NEEDED PYTHON 3.10.X !!!

This project implements a complete zero-shot forecasting pipeline using Chronos-2 (Amazon). The goal is to evaluate univariate forecasting, multivariate forecasting with covariates, and the robustness of the model under controlled perturbations. We use the Rossmann Store Sales dataset, which contains real daily sales values and several external factors such as promotions, holidays and store metadata.

## Project Structure
chronos-2-dnlp/
│
├── main.py                        (main pipeline)
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py        (loading, cleaning, feature engineering)
│   │   ├── train.csv              (raw input)
│   │   └── store.csv              (raw input)
│   │
│   ├── features/
│   │   └── build_features.py      (target + covariates extraction)
│   │
│   ├── models/
│       ├── predict_model.py       (Chronos-2 inference: p10, median, p90)
│       └── robustness.py          (noise, shuffle, missing future tests)
│
└── tests/
    ├── test_environment.py        (pytest environment checks)
    └── test_robustness.py         (pytest robustness tests)

Outputs are stored in: outputs/

## 1. Data Preparation
The preprocessing step:
- merges train.csv and store.csv
- removes days with store closed
- fills missing values
- creates time features such as day, month, year, week number
- fixes mixed or categorical types

The processed dataset is saved in:
src/data/processed_rossmann.csv

## 2. Feature Extraction
We extract:
- the Sales column as target series
- all other relevant numeric columns as covariates

Chronos-2 expects:
- past covariates of shape (num_covariates, T)
- future covariates of shape (num_covariates, horizon)

The pipeline reshapes everything correctly.

## 3. Chronos-2 Model
Model used:
amazon/chronos-2 https://huggingface.co/amazon/chronos-2

Capabilities:
- multivariate forecasting
- past and future covariates
- group attention
- quantile predictions (p10, median, p90)

This model works zero-shot, no training required.

## 4. Forecasting Outputs
### Univariate Forecast
Uses only the target series.
Output: outputs/univariate.csv  
Columns: p10, median, p90

### Covariate Forecast
Uses target plus all covariates.
Output: outputs/covariate.csv

## 5. Robustness Experiments
We evaluate whether Chronos-2 actually uses covariates.

### Noise Test
Adds a random noise column.  
A robust model should ignore it.  
Output: outputs/noise_test.csv

### Shuffle Test
Shuffles the Promo column.  
Breaks correlation and should degrade performance.  
Output: outputs/shuffle_test.csv

### Missing Future Test
Removes future values of SchoolHoliday.  
The model should lose accuracy.  
Output: outputs/missing_future_test.csv

## Using the Makefile
The Makefile lets you run everything with simple commands from the project root.

### Create the virtual environment
make env

### Install PyTorch with GPU support
make gpu

### Run the main forecasting pipeline
make run

### Run all tests (environment + robustness)
make test

### Run only environment tests
make envtest

### Run only robustness tests
make robustness

### Quick environment check
make check

### Full initialization workflow (env + tests + run)
make init

### Full workflow for users who already have an environment
make fullrun

This runs: environment check, forecasting pipeline and robustness tests.

## Running the Project Without Make (Windows)
If make is not available, use the PowerShell setup script.

Run:
powershell -ExecutionPolicy Bypass -File setup.ps1

This script will:
- create the virtual environment
- install dependencies
- optionally install PyTorch with GPU support
- verify Chronos-2 and PyTorch installation

After setup:
.venv\Scripts\python main.py  
To run tests manually:  
.venv\Scripts\pytest -q

## Running the Project Using Windows .bat Scripts

For users who cannot use Make or prefer a one-click execution method, two Windows batch scripts are provided.

### Setup Script (first-time installation)
Use `setup.bat` only the first time:
- creates the `.venv` virtual environment
- installs dependencies
- optionally installs PyTorch with GPU support
- verifies PyTorch and Chronos-2 installation

Run with:
    setup.bat

## Running the Project Using Windows Batch Files (.bat)

For Windows users who prefer running the project without Make or PowerShell, the repository includes several .bat files that automate the most common workflows. These scripts allow you to run the full pipeline, the tests, or both together with a simple double-click.

Each .bat script automatically:
- activates the virtual environment (.venv)
- prints clear messages about what is happening
- shows the full output of Python (pipeline, warnings, test logs)
- pauses at the end so you can read the results

When the script finishes and shows the message:
    Press any key to continue...

You can exit by pressing:
- Enter  
- Space  
- or any other key on the keyboard  

This closes the batch window safely.

---

### run_main.bat
Runs only the forecasting pipeline (`main.py`).

Useful when you want to see:
- data preparation steps
- model loading
- univariate and covariate predictions
- output file locations

Run it with a double-click:
    run_main.bat

---

### run_tests.bat
Runs all test files:
- environment tests
- robustness tests

This is helpful to validate that:
- dependencies are installed correctly
- Chronos-2 is working
- covariate robustness checks run successfully

Double-click:
    run_tests.bat

---

### run_pipeline_and_tests.bat
Runs the entire workflow:
1. the forecasting pipeline  
2. environment tests  
3. robustness tests  

This is the recommended script for group members who want to “run everything” without using Python or Make.

Run with:
    run_pipeline_and_tests.bat

The script stops automatically if the pipeline fails.

---

### run_visuals.bat (optional)
If the project contains a script to generate plots (for example `scripts/generate_plots.py`), this .bat file will run it.

Useful for generating figures for the final report.

---

### Summary of .bat Files
- `run_main.bat`: pipeline only  
- `run_tests.bat`: tests only  
- `run_pipeline_and_tests.bat`: pipeline + all tests  
- `run_visuals.bat`: generate plots (optional)

All .bat files should be run **after the environment is created**, either using:
- `setup.bat` (first time only), or  
- `make env` (recommended for developers)

---

### How to Exit a .bat Script
At the end of each script you will see:
    Press any key to continue . . .

To close the window, simply press:
- Enter  
- or Space  
- or any key  

This safely closes the terminal and ends the run.



## Installing Make on Windows (optional)
Install via Chocolatey:
choco install make

Or use Git Bash where make is already available.

## Requirements
Dependencies are listed in requirements.txt. Install with:
pip install -r requirements.txt

The project supports CPU and GPU execution.

## Notes
All forecasts include quantiles p10, median, p90.  
Chronos-2 runs entirely zero-shot.  
Robustness tests are essential for the scientific evaluation.  
The pipeline is modular and can be extended with additional metrics or visualizations.
