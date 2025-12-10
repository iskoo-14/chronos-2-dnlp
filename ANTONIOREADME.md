DNLP Project – Overview of the Scripts I Added
==============================================

Here is a quick explanation of the Python scripts I created so everyone in the group 
knows what they do and how they fit into the project.


------------------------------------------------------------
1. prepare_rossmann.py
Location: docs/data_preparation/prepare_rossmann.py

This script prepares the Rossmann dataset for all the experiments.

What it does:
- loads train.csv and store.csv
- merges the two datasets
- removes days when the store was closed
- fills missing values
- adds time-based features (year, month, day, etc.)
- adds a noise column (used later for robustness tests)
- saves the final dataset as processed_rossmann.csv

Why it’s important:
All other scripts use this cleaned dataset with covariates.


------------------------------------------------------------
2. run_univariate.py
Location: docs/forecasting/run_univariate.py

This script runs the baseline model.

What it does:
- loads processed_rossmann.csv
- extracts only the target series “Sales”
- runs Chronos-2 without any covariates
- saves the predictions as univariate_results.csv

Why it’s useful:
This gives us a baseline to compare against the covariate model and the robustness tests.


------------------------------------------------------------
3. run_covariates.py
Location: docs/forecasting/run_covariates.py

This script runs the forecasting model using covariates.

What it does:
- loads processed_rossmann.csv
- uses “Sales” as the target
- uses all other numeric columns as covariates
- runs Chronos-2 with both past and future covariates
- computes MAE, MSE, RMSE
- saves results inside the covariate_results/ folder

Why it’s useful:
It lets us check whether Chronos-2 improves when we give it extra information
like Promo, SchoolHoliday, store features, etc.


------------------------------------------------------------
4. run_robustness_tests.py
Location: docs/experiments/run_robustness_tests.py

This script runs the robustness tests that we will use as our project extension.

The three tests:

A) Noise Test  
   Adds a random noise column to see if Chronos ignores irrelevant information.

B) Shuffle Test  
   Shuffles values of “Promo” to break correlations.  
   If performance drops, Chronos was actually using Promo.

C) Missing Future Test  
   Removes future values of “SchoolHoliday” to test how much Chronos depends on
   future-known covariates.

Each test saves:
- forecast.csv
- metrics.csv

The results are saved in:
docs/experiments/noise_test/
docs/experiments/shuffle_test/
docs/experiments/missing_future_test/


------------------------------------------------------------
Quick Summary
-------------
prepare_rossmann.py → prepares the dataset  
run_univariate.py   → baseline forecasting  
run_covariates.py   → forecasting with covariates  
run_robustness_tests.py → robustness experiments

Everything depends on the processed dataset.
