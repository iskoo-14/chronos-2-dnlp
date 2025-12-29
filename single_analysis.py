import os
from chronos import Chronos2Pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd

from src.data.make_dataset import (
    load_raw_data, clean_data, add_time_features, fix_mixed_types, save_processed, temporal_split, select_important_features
)

def ensure_output_folder(path="outputs"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

print("=== DNLP PIPELINE STARTED ===")

base = os.path.dirname(os.path.abspath(__file__))
output_dir = ensure_output_folder("outputs")

print("Loading raw data...")
train_path = os.path.join(base, "src/data/train.csv")
store_path = os.path.join(base, "src/data/store.csv")

df = load_raw_data(train_path, store_path)

print("Cleaning data...")
df = clean_data(df)
df = select_important_features(df)
df = fix_mixed_types(df)

df_past, df_test = temporal_split(df, test_size=30)

FUTURE_COVS = [
    "id",
    "timestamp",
    "Open",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "DayOfWeek"
]

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2")

# Load historical target values and past values of covariates
context_df = df_past

# # (Optional) Load future values of covariates
future_df = df_test[FUTURE_COVS]

# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=30,  # Number of steps to forecast
    quantile_levels=[0.1, 0.5, 0.9],  # Quantile for probabilistic forecast
    id_column="id",  # Column identifying different time series
    timestamp_column="timestamp",  # Column with datetime information
    target="target",  # Column(s) with time series values to predict
)


pred_df_univariate = pipeline.predict_df(context_df, prediction_length=30, quantile_levels=[0.1, 0.5, 0.9], id_column="id", target="target")


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(18, 4), sharey=True)

# ===================== Multivariate =====================
ts_context = context_df.set_index("timestamp")["target"].tail(256)
ts_pred = pred_df.set_index("timestamp")
ts_ground_truth = df_test.set_index("timestamp")["target"]

ts_context.plot(ax=axes[0], label="historical data", color="xkcd:azure")
ts_ground_truth.plot(ax=axes[0], label="future data (ground truth)", color="xkcd:grass green")
ts_pred["predictions"].plot(ax=axes[0], label="forecast", color="xkcd:violet")

axes[0].fill_between(
    ts_pred.index,
    ts_pred["0.1"],
    ts_pred["0.9"],
    alpha=0.7,
    label="prediction interval",
    color="xkcd:light lavender",
)

axes[0].set_title("Multivariate Forecast")
axes[0].legend()


# ===================== Univariate =====================
ts_context = context_df.set_index("timestamp")["target"].tail(256)
ts_pred = pred_df_univariate.set_index("timestamp")
ts_ground_truth = df_test.set_index("timestamp")["target"]

ts_context.plot(ax=axes[1], label="historical data", color="xkcd:azure")
ts_ground_truth.plot(ax=axes[1], label="future data (ground truth)", color="xkcd:grass green")
ts_pred["predictions"].plot(ax=axes[1], label="forecast", color="xkcd:violet")

axes[1].fill_between(
    ts_pred.index,
    ts_pred["0.1"],
    ts_pred["0.9"],
    alpha=0.7,
    label="prediction interval",
    color="xkcd:light lavender",
)

axes[1].set_title("Univariate Forecast")
axes[1].legend()

plt.tight_layout()
plt.show()

