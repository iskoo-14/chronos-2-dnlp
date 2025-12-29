# =========================
# FILE: tests/test_robustness.py
# (updated to new pipeline: df-based, predict_df-based)
# =========================
import os
import pandas as pd

from src.data.make_dataset import (
    load_raw_data,
    clean_data,
    add_time_features,
    select_important_features,
    fix_mixed_types,
    to_chronos_df,
)
from src.models.predict_model import load_model
from src.models.robustness import (
    noise_test,
    strong_noise_test,
    shuffle_test,
    missing_future_test,
    time_shift_test,
    trend_break_test,
    feature_drop_test,
    partial_mask_test,
    scaling_test,
    long_horizon_test,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")


def prepare_df():
    train_path = os.path.join(ROOT, "src/data/train.csv")
    store_path = os.path.join(ROOT, "src/data/store.csv")

    df = load_raw_data(train_path, store_path, store_id=1)
    df = clean_data(df, keep_closed_days=True)
    df = add_time_features(df)
    df = select_important_features(df)
    df = fix_mixed_types(df)
    df = to_chronos_df(df)
    return df


def test_noise():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = noise_test(model, df, horizon=30)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "noise_output.csv"))


def test_strong_noise():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = strong_noise_test(model, df, horizon=30)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "strong_noise_output.csv"))


def test_shuffle():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = shuffle_test(model, df, horizon=30)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "shuffle_output.csv"))


def test_missing_future():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = missing_future_test(model, df, horizon=30)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "missing_future_output.csv"))


def test_time_shift():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = time_shift_test(model, df, horizon=30, shift=7)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "time_shift_output.csv"))


def test_trend_break():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = trend_break_test(model, df, horizon=30, jump=1.0)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "trend_break_output.csv"))


def test_feature_drop():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = feature_drop_test(model, df, horizon=30, drop_feature="Promo")
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "feature_drop_output.csv"))


def test_partial_mask():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = partial_mask_test(model, df, horizon=30, frac=0.3)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "partial_mask_output.csv"))


def test_scaling():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = scaling_test(model, df, horizon=30, scale=10.0)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "scaling_output.csv"))


def test_long_horizon():
    df = prepare_df()
    model = load_model("amazon/chronos-2")
    pred = long_horizon_test(model, df, horizon=90)
    assert pred is not None
    assert os.path.exists(os.path.join(OUT_DIR, "long_horizon_output.csv"))
