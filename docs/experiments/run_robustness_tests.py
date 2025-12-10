import pandas as pd
import numpy as np
import os
from shutil import copyfile
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import ChronosPipeline
import torch


def load_dataset(path):
    return pd.read_csv(path)


def save_dataset(df, path):
    df.to_csv(path, index=False)


def add_noise_column(df):
    df["NoiseFeature"] = np.random.normal(0, 1, len(df))
    return df


def shuffle_column(df, column="Promo"):
    df[column] = np.random.permutation(df[column].values)
    return df


def mask_future(df, column="SchoolHoliday", horizon=30):
    df.loc[len(df) - horizon:, column] = np.nan
    return df


def extract_target(df):
    return df["Sales"].values.tolist()


def extract_covariates(df):
    return df.drop(columns=["Sales", "Date"]).values.tolist()


def run_forecast(series, covariates):
    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    forecast = pipe.predict(
        series=series,
        prediction_length=30,
        past_covariates=covariates,
        future_covariates=covariates,
    )
    return forecast


def compute_metrics(true_vals, preds):
    mae = mean_absolute_error(true_vals, preds)
    mse = mean_squared_error(true_vals, preds)
    rmse = np.sqrt(mse)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def save_results(forecast, metrics, folder):
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"forecast": forecast}).to_csv(
        os.path.join(folder, "forecast.csv"), index=False
    )
    pd.DataFrame([metrics]).to_csv(
        os.path.join(folder, "metrics.csv"), index=False
    )


if __name__ == "__main__":

    BASE = os.path.dirname(os.path.abspath(__file__))
    DATASET = os.path.join(BASE, "../data_preparation/processed_rossmann.csv")

    df = load_dataset(DATASET)

    # Extract baseline true values
    series = extract_target(df)
    covariates = extract_covariates(df)
    true_future = series[-30:]

    # Experiment A: Noise Test
    df_noise = df.copy()
    df_noise = add_noise_column(df_noise)
    cov_noise = extract_covariates(df_noise)
    pred_noise = run_forecast(series, cov_noise)
    metrics_noise = compute_metrics(true_future, pred_noise[:30])
    save_results(pred_noise, metrics_noise, os.path.join(BASE, "noise_test"))

    # Experiment B: Shuffle Test
    df_shuffle = df.copy()
    df_shuffle = shuffle_column(df_shuffle, "Promo")
    cov_shuffle = extract_covariates(df_shuffle)
    pred_shuffle = run_forecast(series, cov_shuffle)
    metrics_shuffle = compute_metrics(true_future, pred_shuffle[:30])
    save_results(pred_shuffle, metrics_shuffle, os.path.join(BASE, "shuffle_test"))

    # Experiment C: Missing Future Covariate Test
    df_mask = df.copy()
    df_mask = mask_future(df_mask, "SchoolHoliday")
    cov_mask = extract_covariates(df_mask)
    pred_mask = run_forecast(series, cov_mask)
    metrics_mask = compute_metrics(true_future, pred_mask[:30])
    save_results(pred_mask, metrics_mask, os.path.join(BASE, "missing_future_test"))

    print("✓ All robustness experiments completed successfully.")
