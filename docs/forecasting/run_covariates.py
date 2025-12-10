import pandas as pd
import numpy as np
import torch
from transformers import ChronosPipeline
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_dataset(path: str): # Rossmann dataset
    return pd.read_csv(path)


def extract_target(df, target_col="Sales"): 
    return df[target_col].values.tolist() # sales as a list


# NUMERIC COLUMNS
# PAST AND FUTURE COVARIATES
def extract_covariates(df, exclude_cols=("Date", "Sales")):
    cov_df = df.drop(columns=list(exclude_cols), errors="ignore")
    return cov_df.values.tolist()

# Chronos
def run_covariate_forecasting(series, past_covariates, future_covariates, horizon=30):

    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    forecast = pipe.predict(
        series=series,
        prediction_length=horizon,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )

    return forecast


def compute_metrics(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


def save_results(predictions, metrics, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pd.DataFrame({"forecast": predictions}).to_csv(
        os.path.join(output_folder, "covariate_forecast.csv"), index=False
    )

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_folder, "covariate_metrics.csv"), index=False
    )

    print(f"Saved forecast and metrics to {output_folder}")


if __name__ == "__main__":

    base = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base, "../data_preparation/processed_rossmann.csv")
    output_folder = os.path.join(base, "covariate_results")

    print("Loading dataset...")
    df = load_dataset(dataset_path)

    print("Extracting series and covariates...")
    series = extract_target(df)
    covariates = extract_covariates(df)

    # Chronos requires past and future covariates explicitly
    past_covariates = covariates
    future_covariates = covariates

    print("Running Chronos-2 covariate forecasting...")
    forecast = run_covariate_forecasting(series, past_covariates, future_covariates)

    # Compute metrics (we can evaluate only if last true values exist)
    true_future = series[-len(forecast):]
    metrics = compute_metrics(true_future, forecast[:len(true_future)])

    print("Metrics:", metrics)

    print("Saving results...")
    save_results(forecast, metrics, output_folder)

    print("✓ Done.")
