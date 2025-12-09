import pandas as pd
import torch
from transformers import ChronosPipeline
import os


def load_processed_dataset(path: str):
    """Load dataset preprocessed by prepare_rossmann.py"""
    df = pd.read_csv(path)
    return df


def extract_target(df, target_col="Sales"):
    """Extract the univariate series"""
    return df[target_col].values.tolist()


def run_univariate_forecast(series, horizon=30):
    """Run Chronos-2 Zero-Shot Univariate prediction"""
    
    pipe = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    forecast = pipe.predict(
        series=series,
        prediction_length=horizon,
        past_covariates=None,
        future_covariates=None
    )

    return forecast


def save_results(forecast, output_path):
    pd.DataFrame({"predicted": forecast}).to_csv(output_path, index=False)
    print(f"Saved univariate forecast to {output_path}")


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, "../../docs/data_preparation/processed_rossmann.csv")
    output_path = os.path.join(base_path, "univariate_results.csv")

    print("Loading dataset...")
    df = load_processed_dataset(dataset_path)

    print("Extracting target series...")
    series = extract_target(df)

    print("Running univariate forecast...")
    forecast = run_univariate_forecast(series)

    print("Saving results...")
    save_results(forecast, output_path)
