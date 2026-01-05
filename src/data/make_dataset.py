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
    # ensure temporal order to avoid misaligned split
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
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
    g = g.drop_duplicates(subset=[date_col], keep="last")

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
    # Legacy utility (unused in main pipeline); kept for reference only.
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
    # Legacy utility (unused in main pipeline); kept for reference only.
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
    g = g.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

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
    min_obs=None,
    min_mean_target=None,
    covariate_cols=None,
    check_recent_covariates=False,
    zero_tail_open_max=None,
    zero_tail_open_share=None,
    check_future_covariates=None,
):
    """Return per-store continuity stats and a validity flag."""
    if covariate_cols is None:
        covariate_cols = []

    records = []
    for store_id, g in df.groupby(store_col):
        g = g.copy()
        g[date_col] = pd.to_datetime(g[date_col])
        g = g.sort_values(date_col)

        observed = g[g[target_col].notna()]
        n_obs = len(observed)
        start_date = observed[date_col].min() if n_obs > 0 else pd.NaT
        start_date_all = g[date_col].min()
        end_date_observed = observed[date_col].max() if n_obs > 0 else pd.NaT
        end_date_all = g[date_col].max()
        max_run = max_consecutive_daily_run(observed[date_col])

        is_valid = True
        reasons = []
        recent_ok = True
        cov_ok = True
        zero_tail_open_ok = True
        zero_open_count = 0
        open_count = 0
        zero_open_share = 0.0
        max_zero_open_run = 0
        future_na_cols = []

        if n_obs == 0:
            is_valid = False
            reasons.append("no_target")

        # Treat max_run as gate only if recent window check is disabled.
        if recent_window_length is None and max_run < min_run:
            is_valid = False
            reasons.append(f"max_run<{min_run}")

        if min_obs is not None and n_obs < min_obs:
            is_valid = False
            reasons.append(f"n_obs<{min_obs}")

        if min_mean_target is not None:
            mean_target = observed[target_col].mean()
            if pd.isna(mean_target) or mean_target < min_mean_target:
                is_valid = False
                reasons.append(f"mean_target<{min_mean_target}")

        if recent_window_length is not None:
            recent_ok = has_continuous_recent_window(
                g,
                date_col=date_col,
                target_col=target_col,
                window_length=recent_window_length,
            )
            if not recent_ok:
                is_valid = False
                reasons.append("recent_gap")

        if check_recent_covariates and covariate_cols and recent_window_length is not None:
            end_date_cov = g[date_col].max()
            start_date_cov = end_date_cov - pd.Timedelta(days=recent_window_length - 1)
            expected_range_cov = pd.date_range(start_date_cov, end_date_cov, freq="D")
            g_recent_cov = (
                g.drop_duplicates(subset=date_col)
                .set_index(date_col)
                .reindex(expected_range_cov)
            )
            cov_na = g_recent_cov[covariate_cols].isna().any(axis=1).any()
            cov_ok = not cov_na
            if cov_na:
                is_valid = False
                reasons.append("recent_cov_na")

        # Known future covariates check (Open/Promo/etc.) limited to recent window
        if check_future_covariates and recent_window_length is not None:
            future_cols = check_future_covariates
            end_date_future = g[date_col].max()
            start_date_future = end_date_future - pd.Timedelta(days=recent_window_length - 1)
            expected_range_future = pd.date_range(start_date_future, end_date_future, freq="D")
            g_recent_future = (
                g.drop_duplicates(subset=date_col)
                .set_index(date_col)
                .reindex(expected_range_future)
            )
            na_mask = g_recent_future[future_cols].isna()
            future_na_cols = [c for c in future_cols if na_mask[c].any()]
            future_na = len(future_na_cols) > 0
            if future_na:
                is_valid = False
                reasons.append(f"known_future_na[{','.join(future_na_cols)}]")

        # Zero-tail on open days only (exclude real closures Open=0)
        if recent_window_length is not None and "Open" in g.columns:
            end_date_tail = g[date_col].max()
            start_date_tail = end_date_tail - pd.Timedelta(days=recent_window_length - 1)
            expected_range_tail = pd.date_range(start_date_tail, end_date_tail, freq="D")
            g_recent_tail = (
                g.drop_duplicates(subset=date_col)
                .set_index(date_col)
                .reindex(expected_range_tail)
            )
            open_vals = pd.to_numeric(g_recent_tail["Open"], errors="coerce")
            open_mask = open_vals == 1
            zeros_open = (g_recent_tail[target_col] == 0) & open_mask
            open_count = int(open_mask.sum())
            zero_open_count = int(zeros_open.sum())
            zero_open_share = float(zero_open_count / open_count) if open_count > 0 else 0.0
            segments = zeros_open.ne(zeros_open.shift()).cumsum()
            max_zero_open_run = int(zeros_open.groupby(segments).sum().max() or 0)
            if zero_tail_open_max is not None:
                if open_count > 0 and max_zero_open_run > zero_tail_open_max:
                    zero_tail_open_ok = False
                    is_valid = False
                    reasons.append(f"inconsistent_zero_open_run>{zero_tail_open_max}")
            if zero_tail_open_share is not None:
                if open_count > 0 and zero_open_share > zero_tail_open_share:
                    zero_tail_open_ok = False
                    is_valid = False
                    reasons.append(f"inconsistent_zero_open_share>{zero_tail_open_share}")

        records.append(
            {
                "store_id": store_id,
                "n_obs": n_obs,
                "start_date": start_date,
                "start_date_all": start_date_all,
                "end_date_observed": end_date_observed,
                "end_date_all": end_date_all,
                "max_consecutive_daily_run": max_run,
                "recent_window_ok": bool(recent_ok),
                "recent_cov_ok": bool(cov_ok),
                "zero_tail_open_ok": bool(zero_tail_open_ok),
                "zero_open_count": int(zero_open_count),
                "open_count": int(open_count),
                "zero_open_share": float(zero_open_share),
                "max_zero_open_run": int(max_zero_open_run),
                # n_total_days/missing_target_days refer to calendar span (meaningful after daily reindex in main)
                "n_total_days": int((end_date_all - start_date_all).days + 1) if pd.notna(end_date_all) and pd.notna(start_date_all) else None,
                "missing_target_days": int((end_date_all - start_date_all).days + 1 - n_obs) if pd.notna(end_date_all) and pd.notna(start_date_all) else None,
                "future_na_cols": ",".join(future_na_cols) if future_na_cols else "",
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
    min_obs=None,
    min_mean_target=None,
    covariate_cols=None,
    check_recent_covariates=False,
    zero_tail_open_max=None,
    zero_tail_open_share=None,
    check_future_covariates=None,
):
    """Filter dataframe to valid stores and return df, report, and id list."""
    report_df = build_store_validity_report(
        df,
        store_col=store_col,
        date_col=date_col,
        target_col=target_col,
        min_run=min_run,
        recent_window_length=recent_window_length,
        min_obs=min_obs,
        min_mean_target=min_mean_target,
        covariate_cols=covariate_cols,
        check_recent_covariates=check_recent_covariates,
        zero_tail_open_max=zero_tail_open_max,
        zero_tail_open_share=zero_tail_open_share,
        check_future_covariates=check_future_covariates,
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

    # fill only static store-level metadata; leave time-varying covariates untouched
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
        if "Store" in df.columns:
            df[fill_cols] = (
                df.groupby("Store")[fill_cols].ffill().bfill().infer_objects(copy=False)
            )
        else:
            df[fill_cols] = df[fill_cols].ffill().bfill().infer_objects(copy=False)

    if target_col:
        df[target_col] = target_series

    return df


def add_time_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Keep original DayOfWeek if present (Rossmann usa 1-7); otherwise derive 1-7
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

    return df


def fix_mixed_types(df):
    df = df.copy()
    target_cols = {"Sales", "target"}
    # Global, deterministic mapping for StateHoliday to avoid per-store category codes
    if "StateHoliday" in df.columns:
        mapping = {"0": 0, "a": 1, "b": 2, "c": 3}
        df["StateHoliday"] = (
            df["StateHoliday"].astype(str).str.lower().map(mapping).fillna(0).astype(int)
        )

    for col in df.columns:
        if df[col].dtype == "object":
            # NOTE: Avoid cat.codes for model features; use explicit global mappings instead.
            if col == "StateHoliday":
                continue
            df[col] = df[col].astype("category").cat.codes

    for col in df.columns:
        if col in target_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if col == "Open":
                # Do not invent closures: leave NaN to be caught by continuity checks
                continue
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

    print("[STEP] Converting to Chronos format...")
    df = to_chronos_df(df)

    print("[STEP] Selecting important features (paper-style)...")
    df = select_important_features(df)

    print("[STEP] Fixing mixed types...")
    df = fix_mixed_types(df)

    print("[STEP] Saving processed dataset...")
    save_processed(df, out_path)
