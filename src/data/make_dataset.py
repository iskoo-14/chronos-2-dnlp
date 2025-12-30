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


# ---------------------------------------------------------------------------#
# DATA REGULARITY UTILITIES
# ---------------------------------------------------------------------------#

def enforce_daily_frequency_store(df_store, date_col="Date", store_col="Store"):
    """Reindex a single-store frame on the full daily range [min, max]."""
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


def max_consecutive_daily_run(dates: pd.Series) -> int:
    """Compute the longest streak of consecutive calendar days."""
    if dates is None:
        return 0
    d = pd.to_datetime(dates).dropna().drop_duplicates().sort_values()
    if len(d) == 0:
        return 0

    diffs = d.diff().dt.days
    segments = diffs.ne(1).cumsum()
    return int(d.groupby(segments).size().max())


def has_continuous_history(
    df_store, date_col="Date", target_col="Sales", min_len=256
):
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


def has_continuous_tail(
    df_store, date_col="Date", target_col="Sales", context_length=256, horizon=30
):
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


def has_continuous_recent_window(
    df_store, date_col="Date", target_col="Sales", window_length=256
):
    """
    Check that the last `window_length` days ending at max(date) are consecutive and observed.
    """
    if len(df_store) == 0:
        return False

    g = df_store.copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col)

    end_date = g[date_col].max()
    start_date = end_date - pd.Timedelta(days=window_length - 1)
    expected_range = pd.date_range(start_date, end_date, freq="D")

    g_recent = (
        g.drop_duplicates(subset=date_col)
        .set_index(date_col)
        .reindex(expected_range)
        .reset_index()
        .rename(columns={"index": date_col})
    )

    if len(g_recent) != window_length:
        return False

    return g_recent[target_col].notna().all()


def build_store_validity_report(
    df,
    store_col="Store",
    date_col="Date",
    target_col="Sales",
    min_run=256,
    recent_window_length=None,
):
    """Return per-store continuity stats and a validity flag."""
    if recent_window_length is None:
        recent_window_length = min_run

    records = []
    for store_id, g in df.groupby(store_col):
        g = g.copy()
        g[date_col] = pd.to_datetime(g[date_col])
        g = g.sort_values(date_col)

        observed = g[g[target_col].notna()]
        n_obs = len(observed)
        start_date = observed[date_col].min() if n_obs > 0 else pd.NaT
        end_date = observed[date_col].max() if n_obs > 0 else pd.NaT
        max_run = max_consecutive_daily_run(observed[date_col])

        is_valid = True
        reasons = []

        if n_obs == 0:
            is_valid = False
            reasons.append("no_target")

        if max_run < min_run:
            is_valid = False
            reasons.append(f"max_run<{min_run}")

        if recent_window_length is not None:
            if not has_continuous_recent_window(
                g,
                date_col=date_col,
                target_col=target_col,
                window_length=recent_window_length,
            ):
                is_valid = False
                reasons.append("recent_gap")

        records.append(
            {
                "store_id": store_id,
                "n_obs": n_obs,
                "start_date": start_date,
                "end_date": end_date,
                "max_consecutive_daily_run": max_run,
                "is_valid": bool(is_valid),
                "reasons": ";".join(reasons) if reasons else "",
            }
        )

    return pd.DataFrame(records).sort_values("store_id").reset_index(drop=True)


def filter_valid_stores(
    df,
    store_col="Store",
    date_col="Date",
    target_col="Sales",
    min_run=256,
    recent_window_length=None,
):
    """Filter dataframe to valid stores and return df, report, and id list."""
    report_df = build_store_validity_report(
        df,
        store_col=store_col,
        date_col=date_col,
        target_col=target_col,
        min_run=min_run,
        recent_window_length=recent_window_length,
    )
    valid_ids = report_df.loc[report_df["is_valid"], "store_id"].tolist()
    df_filtered = df[df[store_col].isin(valid_ids)].copy()
    return df_filtered, report_df, valid_ids


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

    # keep target as-is (no ffill/bfill to avoid inventing sales)
    target_col = "Sales" if "Sales" in df.columns else None
    target_series = df[target_col] if target_col else None

    non_target_cols = [c for c in df.columns if c != target_col]
    df_non_target = df[non_target_cols].ffill().bfill().infer_objects(copy=False)

    if target_col:
        df = pd.concat([df_non_target, target_series], axis=1)
    else:
        df = df_non_target

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
    target_cols = {"Sales", "target"}

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    for col in df.columns:
        if col in target_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)

    return df


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


def save_processed(df, output_path, verbose=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    if verbose:
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
