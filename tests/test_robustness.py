import os

from src.data.make_dataset import (
    load_raw_data,
    clean_data,
    add_time_features,
    fix_mixed_types,
)
from src.features.build_features import extract_target, extract_covariates
from src.models.predict_model import load_model
from src.models.robustness import (
    noise_test,
    shuffle_test,
    missing_future_test,
    strong_noise_test,
    time_shift_test,
    trend_break_test,
    feature_drop_test,
    partial_mask_test,
    scaling_test,
    long_horizon_test,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "src", "data")
OUT_DIR = os.path.join(ROOT, "outputs")


def prepare_df():
    train = os.path.join(DATA_DIR, "train.csv")
    store = os.path.join(DATA_DIR, "store.csv")

    df = load_raw_data(train, store)
    df = clean_data(df)
    df = add_time_features(df)
    df = fix_mixed_types(df)
    return df


def prepare_features(df):
    target = extract_target(df)
    covariates = extract_covariates(df)  # (T, F) numpy array
    return target, covariates


def test_noise():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = noise_test(model, df, target, cov)

    assert "median" in pred
    assert len(pred["median"]) > 0
    assert os.path.exists(os.path.join(OUT_DIR, "noise_output.csv"))


def test_shuffle():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = shuffle_test(model, df, target, cov)

    assert "p10" in pred
    assert "p90" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "shuffle_output.csv"))


def test_missing_future():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = missing_future_test(model, df, target, cov)

    assert "median" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "missing_future_output.csv"))


def test_strong_noise():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = strong_noise_test(model, df, target, cov)

    assert "median" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "strong_noise_output.csv"))


def test_time_shift():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = time_shift_test(model, df, target, cov)

    assert "median" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "time_shift_output.csv"))


def test_trend_break():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = trend_break_test(model, df, target, cov)

    assert "median" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "trend_break_output.csv"))


def test_feature_drop():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = feature_drop_test(model, df, target, cov)

    assert "median" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "feature_drop_output.csv"))


def test_partial_mask():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = partial_mask_test(model, df, target, cov)

    assert "median" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "partial_mask_output.csv"))


def test_scaling():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = scaling_test(model, df, target, cov)

    assert "median" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "scaling_output.csv"))


def test_long_horizon():
    df = prepare_df()
    target, cov = prepare_features(df)
    model = load_model("amazon/chronos-2")

    pred = long_horizon_test(model, df, target, cov)

    assert "median" in pred
    assert os.path.exists(os.path.join(OUT_DIR, "long_horizon_output.csv"))
