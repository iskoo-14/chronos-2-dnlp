# DNLP Project – Chronos 2 Time Series Forecasting (Rossmann)

**Required Python version: 3.10.x**

This project implements a zero-shot time series forecasting pipeline using Chronos 2 by Amazon. The goal is to evaluate univariate forecasting, multivariate forecasting with covariates, and the robustness of the model under controlled perturbations. We use the Rossmann Store Sales dataset (daily sales with promo/holiday covariates). All experiments are zero-shot: no model training or cross-store learning.

## Project Structure

```text
chronos-2-dnlp/
│
├── main.py                        main forecasting pipeline (multi-store)
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py        loading, cleaning, continuity checks, store filters
│   │   ├── train.csv              raw input data
│   │   └── store.csv              raw input data
│   │
│   ├── models/
│   │   ├── predict_model.py       Chronos 2 inference helpers
│   │   └── robustness.py          robustness experiments (noise, shuffle, etc.)
│   │
│   ├── evaluation/
│   │   ├── compare_results.py     WQL aggregation and report
│   │   └── select_best_context.py context selection summary
│   │
│   └── visualization/
│       └── generate_plots.py      sampled plots to avoid thousands of PNG
│
├── reports/                       aggregated metrics (validity, WQL, robustness)
├── outputs/                       forecasts per store/context (ctx_*)
└── auto_runs/                     .bat/.sh scripts for end-to-end runs
```

## 1. Data Preparation

`make_dataset.py`:
- merges `train.csv` and `store.csv`
- enforces daily frequency per store (reindex on full calendar, no fill on target)
- filters invalid stores: minimum consecutive run, continuous recent window, covariates not NaN, minimum observations, and “zero tail” filter (long zero runs or high zero share in the recent window)
- adds calendar feature `DayOfWeek`
- converts to Chronos format (`id`, `timestamp`, `target`, covariates)
- saves per-store processed files `src/data/processed_rossmann_store_<id>.csv`

## 2. Feature Extraction

Target: `Sales` (`target` after conversion).  
Covariates (paper-style): `Customers` (past-only), `Open`, `Promo`, `SchoolHoliday`, `StateHoliday`, `DayOfWeek`.

Additional derived covariates (feature engineering):
- `PromoEff = Promo * Open` (remove ambiguous Promo when closed)
- `ClosedRunLen` (capped) and `DaysToNextOpen` (capped) to expose long closures
- `OpenCat` (0/1 categorical) for some wrappers
All are added in `main.py` and used as known-future covariates.

## 3. Chronos 2 Model

Model: `amazon/chronos-2` (zero-shot).  
Key properties: multivariate input, known future covariates, quantile outputs (p10, median, p90), automatic scaling. No training/fine-tuning performed.

## 4. Forecasting Outputs

- Context lengths: default 512 (ablation [128, 256, 512] optional).  
- Horizon: 30.  
- Outputs per store and context in `outputs/ctx_<len>/`:
  - `univariate_store_<id>.csv`
  - `covariate_store_<id>.csv`
  - `ground_truth_store_<id>.csv`
- MAE diagnostics per store: `reports/mae_open_closed.csv` (all/open/closed splits)
- Optional system-gated covariates (forecast * Open) via `maingated.py`, saved as `covariate_gated_store_<id>.csv`

## 5. Robustness Experiments (optional)

Located in `src/models/robustness.py`. Tests include noise, strong noise, shuffle (Promo), missing future (SchoolHoliday), time shift, trend break, feature drop, partial mask, scaling, long horizon. Outputs are saved in `outputs/` as `<test>_output_store_<id>.csv`. Use `RUN_ROBUSTNESS=True` in `main.py` to enable (expensive).

## 6. Evaluation

`compare_results.py`:
- computes Weighted Quantile Loss (WQL) per store and context
- optional outlier filter on WQL
- saves:
  - `reports/wql_per_store.csv` (and `wql_per_store_all.csv` if filtered)
  - `reports/wql_by_context.csv`
  - `reports/wql_summary.csv`
  - `outputs/comparison_report.txt`
`select_best_context.py`:
- reads `reports/wql_by_context.csv` and prints the best context per mode (covariate/univariate)

## 7. Plots

`generate_plots.py` creates figures only for a small sample of stores (configurable via `PLOT_SAMPLE_STORES` / `GENERATE_PER_STORE`) to avoid thousands of PNG. Worst stores (by `mae_open_closed.csv`), a few best, and optional samples are always included. Figures are saved under `outputs/figures/ctx_*` split into `bad/`, `good/`, `other/`. Selection summary: `reports/plot_selection.csv`.

## Running

One-click:
- Windows: `auto_runs\run_all.bat`
- mac/Linux: `bash auto_runs_mac/run_all.sh`

Manual:
```
python main.py
python src/evaluation/compare_results.py
python src/evaluation/select_best_context.py   # only if multiple contexts
python src/visualization/generate_plots.py     # optional (sampled stores)
```

Key flags in `main.py`:
- `RUN_ALL_CONTEXTS` (default False): ablation [128, 256, 512]
- `RUN_ROBUSTNESS` (default False): enable robustness tests
- `SKIP_EXISTING_*`: set to False to force regeneration
- Store filters: `MIN_RUN`, `MIN_OBS`, `CHECK_RECENT_COVS`, `ZERO_TAIL_MAX`, `ZERO_TAIL_SHARE`

## Notes

- Zero-shot only; no training or cross-store learning.
- Filters (continuity + zero-tail) keep the setup close to the Chronos paper and drop degenerate stores.
- Robustness is optional; enable it only if you need stability analysis (costly).***
