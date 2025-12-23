# DNLP Project – Chronos 2 Time Series Forecasting

**Required Python version: 3.10.x**

This project implements a complete zero-shot time series forecasting pipeline using Chronos 2 by Amazon.  
The goal is to evaluate univariate forecasting, multivariate forecasting with covariates, and the robustness of the model under controlled perturbations.

The Rossmann Store Sales dataset is used, containing daily sales and external features such as promotions and holidays.

All experiments are zero-shot.  
No model training is performed.

## Project Structure

```text
chronos-2-dnlp/
│
├── main.py                        main forecasting pipeline
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py        loading, cleaning, feature engineering
│   │   ├── train.csv              raw input data
│   │   └── store.csv              raw input data
│   │
│   ├── features/
│   │   └── build_features.py      target and covariate extraction
│   │
│   ├── models/
│   │   ├── predict_model.py       Chronos 2 inference
│   │   └── robustness.py          robustness experiments
│
├── evaluation/
│   └── compare_results.py         metrics and comparison report
│
├── tests/
│   ├── test_environment.py        pytest environment checks
│   └── test_robustness.py         pytest robustness tests
│
└── outputs/
    ├── *.csv                      forecasts and ground truth
    └── figures/                  plots and visualizations
```

## 1. Data Preparation

The script `make_dataset.py` performs:
- merge of `train.csv` and `store.csv`
- removal of days when the store was closed
- handling of missing values
- creation of calendar features (day, month, year, week number)
- conversion of categorical and mixed-type features

The processed dataset is saved to:
`src/data/processed_rossmann.csv`

## 2. Feature Extraction

We extract:
- Target: Sales
- Covariates: all engineered numerical features

Covariates are formatted to match the Chronos 2 API:
- past covariates shape: num_features × sequence_length
- future covariates shape: num_features × prediction_horizon

## 3. Chronos 2 Model

Model used:
- amazon/chronos-2
- https://huggingface.co/amazon/chronos-2

Key properties:
- zero-shot forecasting
- supports multivariate inputs
- accepts past and future covariates
- produces probabilistic forecasts
- outputs quantiles: p10, median, p90
- automatic scaling

## 4. Forecasting Outputs

Univariate Forecast  
Uses only the Sales time series.  
Output: `outputs/univariate.csv`

Covariate Forecast  
Uses Sales together with all covariates.  
Output: `outputs/covariate.csv`

## 5. Robustness Experiments

Noise Test  
Adds a purely random covariate.  
Output: `outputs/noise_output.csv`

Shuffle Test  
Randomly shuffles the Promo column to break temporal correlation.  
Output: `outputs/shuffle_output.csv`

Missing Future Test  
Masks future values of SchoolHoliday.  
Output: `outputs/missing_future_output.csv`

## 6. Evaluation

Evaluation is performed using:
- temporal split (last 30 days as ground truth)
- Weighted Quantile Loss (WQL)
- comparison between univariate and covariate forecasts
- relative comparisons for robustness tests

Results are written to:
`outputs/comparison_report.txt`

## Using the Makefile

make env  
make gpu  
make run  
make envtest  
make robustness  
make test  
make fullrun  

## Running Without Make (Windows)

PowerShell setup:
`powershell -ExecutionPolicy Bypass -File setup.ps1`

After setup:
`.venv\Scripts\python main.py`

To run tests manually:
`.venv\Scripts\pytest -q`

## Notes

- All forecasts include p10, median, and p90 quantiles.
- Chronos 2 runs completely zero-shot.
- Robustness tests measure whether the model truly leverages covariates.
- The pipeline is modular and easily extensible.
