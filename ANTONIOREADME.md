DNLP Project – Scripts Overview
===============================

This document explains in simple terms the Python scripts added to the project,
what they do, and why they are needed for the Chronos-2 forecasting pipeline.


------------------------------------------------------------
1. prepare_rossmann.py
Location: docs/data_preparation/prepare_rossmann.py
Purpose: Prepare the Rossmann dataset for all experiments.

What it does:
- Loads train.csv and store.csv
- Merges them into a single dataset
- Removes days when the store was closed
- Handles missing values
- Generates useful time-based features (day, month, week, etc.)
- Adds a "RandomNoise" column (used for robustness experiments)
- Saves the final processed dataset as: processed_rossmann.csv

Why it is needed:
This script produces the clean dataset that all other scripts will use
during forecasting. Without it, Chronos-2 cannot correctly use covariates.

How to run:
python prepare_rossmann.py


------------------------------------------------------------
2. run_univariate.py
Location: docs/forecasting/run_univariate.py
Purpose: Run Chronos-2 in zero-shot univariate forecasting mode.

What it does:
- Loads processed_rossmann.csv
- Extracts the target series "Sales"
- Calls Chronos-2 WITHOUT any covariates
- Produces future predictions (default horizon = 30)
- Saves results as: univariate_results.csv

Why it is needed:
This script provides the baseline performance of Chronos-2.
We use this as comparison for:
- covariate forecasting
- noise tests
- shuffle tests
- masking tests

How to run:
python run_univariate.py


------------------------------------------------------------
3. run_covariates.py
Location: docs/forecasting/run_covariates.py
Purpose: Run Chronos-2 using target + covariates.

What it does:
- Loads processed_rossmann.csv
- Uses "Sales" as target
- Uses all other numeric columns as covariates
- Runs Chronos-2 with past and future covariates
- Computes metrics (MAE, MSE, RMSE)
- Saves predictions and metrics in: covariate_results/

Why it is needed:
This script shows whether Chronos-2 actually uses external information
such as:
- Promo
- SchoolHoliday
- Store features
and whether these variables improve forecasting accuracy.

How to run:
python run_covariates.py


------------------------------------------------------------
Next Step: Robustness Experiments
A separate script (run_robustness_tests.py) will later perform:
- Noise test (random covariate)
- Shuffle test (breaking correlations)
- Missing future covariate test

These experiments rely on the outputs produced by the scripts above.


------------------------------------------------------------
Summary
prepare_rossmann.py → prepares dataset
run_univariate.py   → baseline forecasting
run_covariates.py   → forecasting with covariates
robustness tests    → evaluate Chronos-2 intelligence

All scripts work without training (Chronos-2 is zero-shot).
The cleaned dataset is required for the entire pipeline.

