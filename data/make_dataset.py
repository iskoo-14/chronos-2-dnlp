import pandas as pd
import os
import numpy as np

def load_raw_data(train_path: str, store_path: str, store_id: int | None = None) -> pd.DataFrame:
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path, low_memory=False)

    df = train.merge(store, on="Store", how="left")

    if store_id is not None:
        df = df[df["Store"] == store_id].copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame, keep_closed_days: bool = True) -> pd.DataFrame:
    df = df.copy()

    if not keep_closed_days and "Open" in df.columns:
        df = df[df["Open"] == 1].copy()

    target_col = "Sales" if "Sales" in df.columns else None
    target_series = df[target_col] if target_col else None

    static_cols = {
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
    }
    fill_cols = [c for c in df.columns if c in static_cols]

    if fill_cols:
        df[fill_cols] = (
            df.groupby("Store")[fill_cols].ffill().bfill().infer_objects(copy=False)
        )

    if target_col:
        df[target_col] = target_series

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

    return df


def enforce_daily_frequency_store(df_store: pd.DataFrame, date_col: str = "Date", store_col: str = "Store") -> pd.DataFrame:
    g = df_store.copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    full_idx = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
    g = g.set_index(date_col).reindex(full_idx)
    g.index.name = date_col
    g = g.reset_index()

    g[store_col] = df_store[store_col].iloc[0]
    return g


def enforce_daily_frequency_all_stores(df: pd.DataFrame, store_col: str = "Store", date_col: str = "Date") -> pd.DataFrame:
    return pd.concat(
        [enforce_daily_frequency_store(g, date_col=date_col, store_col=store_col) for _, g in df.groupby(store_col)],
        ignore_index=True,
    )


def to_chronos_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"Store": "id", "Date": "timestamp", "Sales": "target"})
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return out

def select_important_features(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "id",
        "timestamp",
        "target",
        "Customers", # past-only
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "DayOfWeek",
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[keep_cols].copy()

def fix_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target_cols = {"Sales", "target"}

    # deterministic mapping for StateHoliday
    if "StateHoliday" in df.columns:
        mapping = {"0": 0, "a": 1, "b": 2, "c": 3}
        df["StateHoliday"] = (
            df["StateHoliday"].astype(str).str.lower().map(mapping).fillna(0).astype(int)
        )

    for col in df.columns:
        if df[col].dtype == "object":
            if col == "StateHoliday":
                continue
            df[col] = df[col].astype("category").cat.codes

    # Fill NaNs for numeric covariates except Open
    for col in df.columns:
        if col in target_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if col == "Open":
                continue
            df[col] = df[col].fillna(0)

    return df


def save_processed(df, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def save_processed_per_store(df_chronos, out_dir: str, id_col="id"):
    os.makedirs(out_dir, exist_ok=True)
    for sid, g in df_chronos.groupby(id_col):
        out_path = os.path.join(out_dir, f"processed_store_{int(sid)}.csv")
        g.sort_values("timestamp").to_csv(out_path, index=False)
        
def temporal_split(df: pd.DataFrame, test_size: int = 30):
    g = df.copy()

    if "timestamp" in g.columns:
        g["timestamp"] = pd.to_datetime(g["timestamp"])
        g = g.sort_values("timestamp").reset_index(drop=True)
    elif "Date" in g.columns:
        g["Date"] = pd.to_datetime(g["Date"])
        g = g.sort_values("Date").reset_index(drop=True)

    if len(g) <= test_size:
        raise ValueError("Dataset too small for temporal split")

    df_past = g.iloc[:-test_size].reset_index(drop=True)
    df_test = g.iloc[-test_size:].reset_index(drop=True)
    return df_past, df_test



#New methods
def aggregate_shop(df, shop_id):
    """
    Aggregate time-series data of a single shop into a fixed-length
    feature vector suitable for shop-level clustering.

    Each feature captures a different aspect of shop behaviour:
    level, volatility, trend dynamics, seasonality, promotions
    and operational patterns.
    """
    out = {}
    out["shop_id"] = shop_id

    target = df["target"]

    # ============================================================
    # BLOCK A — Target level and variability
    # Capture the general sales level and stability of the shop
    # ============================================================
    out["mean_target"] = target.mean()                     # Average sales level
    out["std_target"] = target.std()                       # Sales volatility
    out["cv_target"] = out["std_target"] / (out["mean_target"] + 1e-6)  # Relative variability
    out["iqr_target"] = (
        target.quantile(0.95) - target.quantile(0.05)
    )                                                       # Robust spread (outlier-resistant)

    # ============================================================
    # BLOCK B — Short-term dynamics and spikes
    # Characterize day-to-day changes and abrupt movements
    # ============================================================
    chg = df["chg_1"].dropna()                              # First-order differences
    out["mean_abs_chg"] = np.abs(chg).mean()                # Average magnitude of changes
    out["std_chg"] = chg.std()                              # Variability of changes

    spike_thr = chg.quantile(0.9)                           # Extreme-change threshold
    out["spike_rate"] = (np.abs(chg) > spike_thr).mean()    # Frequency of large spikes

    # ============================================================
    # BLOCK C — Temporal dependence and trend smoothness
    # Measure persistence and long-term memory in the series
    # ============================================================
    for lag in [1, 4, 12, 52]:
        col = f"lag_{lag}"
        if col in df:
            out[f"corr_lag{lag}"] = (
                df[["target", col]].corr().iloc[0, 1]
            )                                               # Autocorrelation at given lag
        else:
            out[f"corr_lag{lag}"] = np.nan

    out["ema_gap"] = (
        np.abs(df["ema_4"] - df["ema_8"]).mean()
    )                                                       # Short vs long trend divergence

    # ============================================================
    # BLOCK D — Seasonality strength
    # Quantify weekly, monthly and yearly seasonal effects
    # ============================================================
    out["weekly_strength"] = np.var(
        df["target"] * df["week_sin"]
    )                                                       # Weekly seasonality intensity

    out["monthly_strength"] = np.var(
        df["target"] * df["month_sin"]
    )                                                       # Monthly seasonality intensity

    out["yearly_strength"] = abs(
        out["corr_lag52"]
    )                                                       # Annual pattern persistence

    # ============================================================
    # BLOCK E — Promotion and holiday effects
    # Estimate sensitivity to external demand drivers
    # ============================================================
    if "Promo" in df:
        out["promo_rate"] = df["Promo"].mean()              # Fraction of promo days
        out["promo_lift"] = (
            df.loc[df["Promo"] == 1, "target"].mean()
            - df.loc[df["Promo"] == 0, "target"].mean()
        )                                                   # Average promo impact
    else:
        out["promo_rate"] = np.nan
        out["promo_lift"] = np.nan

    if "SchoolHoliday" in df:
        out["holiday_lift"] = (
            df.loc[df["SchoolHoliday"] == 1, "target"].mean()
            - df.loc[df["SchoolHoliday"] == 0, "target"].mean()
        )                                                   # Holiday effect on sales
    else:
        out["holiday_lift"] = np.nan

    # ============================================================
    # BLOCK F — Store operation patterns
    # Capture opening behaviour and sales when operational
    # ============================================================
    out["closed_rate"] = (df["Open"] == 0).mean()           # Fraction of closed days
    out["mean_target_open"] = (
        df.loc[df["Open"] == 1, "target"].mean()
    )                                                       # Average sales when open

    return out

