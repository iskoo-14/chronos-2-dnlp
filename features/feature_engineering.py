import os
import glob
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Constants
TIMESTAMP_COL = "timestamp"
TARGET_COL = "target"
ID_COL = "id"

BASE_COVARIATES = ["Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday", "DayOfWeek"]

LAGS_DAYS = [1, 2, 4, 8, 12, 16, 24, 52]

WEEK_PERIOD = 53.0
MONTH_PERIOD = 12.0

# utils
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_timestamp_sorted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


# adding features related to date and time
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[TIMESTAMP_COL])

    iso_week = ts.dt.isocalendar().week.astype(int)
    month = ts.dt.month.astype(int)
    quarter = ts.dt.quarter.astype(int)

    df["week_sin"] = np.sin(2.0 * np.pi * (iso_week / WEEK_PERIOD)).astype(np.float32)
    df["week_cos"] = np.cos(2.0 * np.pi * (iso_week / WEEK_PERIOD)).astype(np.float32)

    df["month_sin"] = np.sin(2.0 * np.pi * (month / MONTH_PERIOD)).astype(np.float32)
    df["month_cos"] = np.cos(2.0 * np.pi * (month / MONTH_PERIOD)).astype(np.float32)

    df["quarter"] = quarter.astype(np.int8)

    df["is_month_start"] = ts.dt.is_month_start.astype(np.int8)
    df["is_month_end"] = ts.dt.is_month_end.astype(np.int8)

    return df


def add_target_lags_with_known(df: pd.DataFrame, lags: List[int] = LAGS_DAYS) -> pd.DataFrame:
    df = df.copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    for k in lags:
        col = f"lag_{k}"
        known = f"{col}_known"

        shifted = y.shift(k)  # we shift so we take only the past
        df[known] = (~shifted.isna()).astype(np.int8)
        df[col] = shifted.fillna(0.0).astype(np.float32)

    return df


# since these are covariates that are made out of the target variable, we use only the past so we prevent data leakage
def add_past_only_target_stats(
    df: pd.DataFrame,
    include_ema: bool = True,
    include_chg: bool = True,
    include_rolling: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    y_prev = y.shift(1)

    cols_to_fill = []

    # exponential moving avg
    if include_ema:
        df["ema_4"] = y_prev.ewm(span=4, adjust=False).mean()
        df["ema_8"] = y_prev.ewm(span=8, adjust=False).mean()
        cols_to_fill += ["ema_4", "ema_8"]

    # last change in the target variable
    if include_chg:
        df["chg_1"] = y_prev - y_prev.shift(1)
        cols_to_fill += ["chg_1"]

    # mean in the next 3 days
    if include_rolling:
        df["rolling_mean_3"] = y_prev.rolling(window=3, min_periods=1).mean()
        df["rolling_std_3"] = y_prev.rolling(window=3, min_periods=1).std()
        cols_to_fill += ["rolling_mean_3", "rolling_std_3"]

    # fill safely for the missing ones
    for c in cols_to_fill:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    return df



# EXTENSION 1 PIPELINE
def apply_extension1(
    df: pd.DataFrame,
    include_ema: bool = False,
    include_chg: bool = False,
    include_rolling: bool = False,
):
    required = {ID_COL, TIMESTAMP_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = ensure_timestamp_sorted(df)

    df = add_calendar_features(df)
    df = add_target_lags_with_known(df, LAGS_DAYS)

    if include_ema or include_chg or include_rolling:
        df = add_past_only_target_stats(
            df,
            include_ema=include_ema,
            include_chg=include_chg,
            include_rolling=include_rolling,
        )

    # fill NaNs
    fill_float0 = [
        "week_sin","week_cos","month_sin","month_cos",
        *[f"lag_{k}" for k in LAGS_DAYS],
    ]
    fill_int0 = ["quarter","is_month_start","is_month_end", *[f"lag_{k}_known" for k in LAGS_DAYS]]

    # add covariates
    if include_ema:
        fill_float0 += ["ema_4", "ema_8"]
    if include_chg:
        fill_float0 += ["chg_1"]
    if include_rolling:
        fill_float0 += ["rolling_mean_3", "rolling_std_3"]

    for c in fill_float0:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    for c in fill_int0:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)

    return df


def build_extension1_files(
    processed_dir: str = os.path.join("data", "processed"),
    output_dir: str = os.path.join("data", "extension1"),
    store_ids: Optional[Iterable[int]] = None,
    pattern: str = "processed_store_*.csv",
    include_ema: bool = False,
    include_chg: bool = False,
    include_rolling: bool = False,
) -> str:
    ensure_dir(output_dir)

    if store_ids is None:
        files = sorted(glob.glob(os.path.join(processed_dir, pattern)))
    else:
        files = [os.path.join(processed_dir, f"processed_store_{sid}.csv") for sid in store_ids]
        files = [p for p in files if os.path.exists(p)]

    if not files:
        raise RuntimeError(f"No processed store files found in {processed_dir} (pattern={pattern})")

    print(f"[INFO] Building extension1 for {len(files)} store file(s)")

    strict_check_cols = [
        "week_sin", "week_cos", "month_sin", "month_cos", "quarter",
        "is_month_start", "is_month_end",
        *[f"lag_{k}" for k in LAGS_DAYS],
        *[f"lag_{k}_known" for k in LAGS_DAYS],
    ]

    if include_ema:
        strict_check_cols += ["ema_4", "ema_8"]
    if include_chg:
        strict_check_cols += ["chg_1"]
    if include_rolling:
        strict_check_cols += ["rolling_mean_3", "rolling_std_3"]

    for p in files:
        df = pd.read_csv(p)
        df2 = apply_extension1(
            df,
            include_ema=include_ema,
            include_chg=include_chg,
            include_rolling=include_rolling,
        )

        bad = [c for c in strict_check_cols if c in df2.columns and df2[c].isna().any()]
        if bad:
            raise RuntimeError(f"NaNs detected after feature engineering: {bad} in file {os.path.basename(p)}")

        out_path = os.path.join(output_dir, os.path.basename(p))
        df2.to_csv(out_path, index=False)

    print(f"[INFO] Saved extension1 dataset to: {output_dir}")
    return output_dir



def extension1_covariate_sets(
    include_ema: bool = False,
    include_chg: bool = False,
    include_rolling: bool = False,
):
    future_known = BASE_COVARIATES + [
        "week_sin", "week_cos", "month_sin", "month_cos",
        "quarter", "is_month_start", "is_month_end",
    ]

    past_only = [f"lag_{k}" for k in LAGS_DAYS] + [f"lag_{k}_known" for k in LAGS_DAYS]

    if include_ema:
        past_only += ["ema_4", "ema_8"]
    if include_chg:
        past_only += ["chg_1"]
    if include_rolling:
        past_only += ["rolling_mean_3", "rolling_std_3"]

    return past_only, future_known
