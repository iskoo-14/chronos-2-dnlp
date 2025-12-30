# =========================
# FILE: src/data/make_dataset.py
# =========================
import os
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)


def select_important_features(df):
    """
    Select covariates after conversion to Chronos format.
    Expected columns:
    - id, timestamp, target
    - covariates used in the Chronos-2 paper
    """
    keep_cols = [
        "id",
        "timestamp",
        "target",
        "Customers",      # past-only covariate
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "DayOfWeek",
    ]

    missing = [c for c in keep_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")

    return df[keep_cols].copy()


def temporal_split(df, test_size=30):
    if len(df) <= test_size:
        raise ValueError("Dataset too small for temporal split")
    df_past = df.iloc[:-test_size].reset_index(drop=True)
    df_test = df.iloc[-test_size:].reset_index(drop=True)
    return df_past, df_test

import pandas as pd

def enforce_daily_frequency_store(df_store,date_col = "Date",store_col = "Store"):
    g = df_store.copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col)

    full_idx = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
    g = g.set_index(date_col).reindex(full_idx)
    g.index.name = date_col
    g = g.reset_index()

    store_id = df_store[store_col].iloc[0]
    g[store_col] = store_id

    return g


def has_continuous_history(df_store ,date_col = "Date",target_col = "Sales", min_len = 256):
    g = df_store.copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col)

    observed = g[target_col].notna()

    if observed.sum() < min_len:
        return False

    d = g.loc[observed, date_col].drop_duplicates().sort_values()
    if len(d) == 0:
        return False

    diffs = d.diff().dt.days
    segments = diffs.ne(1).cumsum()
    max_run = d.groupby(segments).size().max()

    return int(max_run) >= int(min_len)

def has_continuous_tail(df_store ,date_col = "Date",target_col = "Sales",context_length = 256,horizon = 30):
    g = df_store.copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col)

    # prendiamo l'ultimo giorno disponibile come "fine"
    end_date = g[date_col].max()
    ctx_end = end_date - pd.Timedelta(days=horizon)
    ctx_start = ctx_end - pd.Timedelta(days=context_length - 1)

    window = g[(g[date_col] >= ctx_start) & (g[date_col] <= ctx_end)].copy()
    if len(window) != context_length:
        return False

    # tutte le date consecutive e target osservato
    window = window.sort_values(date_col)
    diffs = window[date_col].diff().dt.days.iloc[1:]
    if not (diffs == 1).all():
        return False

    return window[target_col].notna().all()



def load_raw_data(train_path, store_path, store_id=None):
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path, low_memory=False)

    df = train.merge(store, on="Store", how="left")

    # optional single-store filtering
    if store_id is not None:
        df = df[df["Store"] == store_id].copy()

    # chronological order
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    return df



def clean_data(df, keep_closed_days=True):
    df = df.copy()

    # NOTE: keep closed days if requested (PDF)
    if not keep_closed_days and "Open" in df.columns:
        df = df[df["Open"] == 1].copy()

    df = df.dropna(subset=["Sales"])

    # ffill/bfill like in your PDF
    df = df.ffill().bfill().infer_objects(copy=False)


    return df


def add_time_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # IMPORTANT: keep DayOfWeek because it is used as covariate in paper-style selection
    # Rossmann has DayOfWeek already, but we standardize it
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    return df


def fix_mixed_types(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes
    return df.fillna(0)


def to_chronos_df(df):
    # Align naming with Chronos quickstart style
    out = df.copy()
    out = out.rename(
        columns={
            "Store": "id",
            "Date": "timestamp",
            "Sales": "target",
        }
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def save_processed(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Processed dataset saved to {output_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base, "train.csv")
    store_path = os.path.join(base, "store.csv")

    out_path = os.path.join(base, "processed_rossmann_single.csv")

    print("[STEP] Loading raw data (single store)...")
    df = load_raw_data(train_path, store_path, store_id=1)

    print("[STEP] Cleaning (keep closed days)...")
    df = clean_data(df, keep_closed_days=True)

    print("[STEP] Adding time features...")
    df = add_time_features(df)

    print("[STEP] Selecting important features (paper-style)...")
    df = select_important_features(df)

    print("[STEP] Fixing mixed types...")
    df = fix_mixed_types(df)

    print("[STEP] Converting to Chronos format...")
    df = to_chronos_df(df)

    print("[STEP] Saving processed dataset...")
    save_processed(df, out_path)
