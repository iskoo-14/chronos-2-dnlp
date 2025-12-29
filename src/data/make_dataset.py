# =========================
# FILE: src/data/make_dataset.py
# =========================
import os
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)


def temporal_split(df, test_size=30):
    if len(df) <= test_size:
        raise ValueError("Dataset too small for temporal split")
    df_past = df.iloc[:-test_size].reset_index(drop=True)
    df_test = df.iloc[-test_size:].reset_index(drop=True)
    return df_past, df_test


def load_raw_data(train_path, store_path, store_id=1):
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path, low_memory=False)

    df = train.merge(store, on="Store", how="left")

    # single-store (as in your PDF)
    df = df[df["Store"] == store_id].copy()

    # chronological order
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

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


def select_important_features(df):
    # Paper-like minimal set (plus target + timestamp + id)
    keep_cols = [
        "Store",
        "Date",
        "Sales",
        "Customers",      # past-only covariate
        "Open",           # known-future covariate
        "Promo",          # known-future covariate
        "StateHoliday",   # known-future covariate (encoded later)
        "SchoolHoliday",  # known-future covariate
        "DayOfWeek",      # calendar feature
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[keep_cols].copy()


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
