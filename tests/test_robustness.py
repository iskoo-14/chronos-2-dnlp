import os
import pandas as pd
import torch

from src.data.make_dataset import (
    load_raw_data, clean_data, add_time_features, fix_mixed_types
)
from src.features.build_features import extract_target, extract_covariates
from src.models.predict_model import load_model
from src.models.robustness import (
    noise_test, shuffle_test, missing_future_test
)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "src/data")
OUTPUT_DIR = os.path.join(BASE, "outputs")


def prepare_dataframe():
    """Load and preprocess Rossmann dataset for testing."""
    train_path = os.path.join(DATA_DIR, "train.csv")
    store_path = os.path.join(DATA_DIR, "store.csv")

    df = load_raw_data(train_path, store_path)
    df = clean_data(df)
    df = add_time_features(df)
    df = fix_mixed_types(df)

    return df


def prepare_features(df):
    """Extract target and covariates."""
    target = extract_target(df)
    covariates = extract_covariates(df)
    return target, covariates


def test_noise_covariate():
    """Check that noise test returns valid forecast quantiles."""
    df = prepare_dataframe()
    target, cov = prepare_features(df)

    model = load_model("amazon/chronos-2")

    result = noise_test(model, df, target, cov)

    assert "median" in result
    assert len(result["median"]) > 0
    assert torch.is_tensor(torch.tensor(result["median"]))


def test_shuffle_covariate():
    """Check that shuffled covariates still produce valid forecasts."""
    df = prepare_dataframe()
    target, cov = prepare_features(df)

    model = load_model("amazon/chronos-2")

    result = shuffle_test(model, df, target, cov)

    assert "p10" in result
    assert "p90" in result
    assert len(result["p10"]) == len(result["p90"])


def test_missing_future_covariate():
    """Check missing future covariate behavior."""
    df = prepare_dataframe()
    target, cov = prepare_features(df)

    model = load_model("amazon/chronos-2")

    result = missing_future_test(model, df, target, cov)

    assert "median" in result
    assert len(result["median"]) > 0


def test_output_files_created():
    """Ensure robustness results are saved correctly by robustness module."""
    expected_files = [
        "noise_output.csv",
        "shuffle_output.csv",
        "missing_future_output.csv"
    ]
    for fname in expected_files:
        path = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(path), f"Missing output file: {fname}"
