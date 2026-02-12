import os
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    TRAIN_PATH, STORE_PATH, PROCESSED_DIR,
    KEEP_CLOSED_DAYS, ENFORCE_DAILY_FREQUENCY, SKIP_EXISTING_PROCESSED
)

from data.make_dataset import (
    load_raw_data,
    clean_data,
    enforce_daily_frequency_all_stores,
    add_time_features,
    to_chronos_df,
    save_processed,
    fix_mixed_types,
)

from features.baseline_features import select_baseline_features
from data.store_selection import filter_valid_stores, reasons_summary


def _ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    if "DayOfWeek" not in df.columns or df["DayOfWeek"].isna().any():
        df = df.copy()
        df["DayOfWeek"] = pd.to_datetime(df["timestamp"]).dt.dayofweek + 1
    return df


if __name__ == "__main__":
    print("[1] Load raw...")
    df_all = load_raw_data(TRAIN_PATH, STORE_PATH)

    if ENFORCE_DAILY_FREQUENCY:
        print("[2] Enforce daily frequency (all stores)...")
        df_all = enforce_daily_frequency_all_stores(df_all)

    HORIZON = 30
    CONTEXT_LENGTHS = [128, 256, 512]
    MIN_RUN = max(CONTEXT_LENGTHS) + HORIZON

    REQUIRE_RECENT_WINDOW = True
    CHECK_RECENT_COVS = False
    ZERO_TAIL_OPEN_MAX = 14
    ZERO_TAIL_OPEN_SHARE = 0.5

    CHECK_FUTURE_COVS = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]
    COVARIATE_COLS_FOR_RECENT_CHECK = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "Customers", "DayOfWeek"]

    print("[3] Store selection: validity report + filter...")
    df_filtered, report_df, valid_store_ids = filter_valid_stores(
        df_all,
        store_col="Store",
        date_col="Date",
        target_col="Sales",
        min_run=MIN_RUN,
        recent_window_length=MIN_RUN if REQUIRE_RECENT_WINDOW else None,
        min_obs=None,
        covariate_cols=COVARIATE_COLS_FOR_RECENT_CHECK,
        check_recent_covariates=CHECK_RECENT_COVS,
        zero_tail_open_max=ZERO_TAIL_OPEN_MAX,
        zero_tail_open_share=ZERO_TAIL_OPEN_SHARE,
        check_future_covariates=CHECK_FUTURE_COVS if REQUIRE_RECENT_WINDOW else None,
    )

    # save reports
    os.makedirs("reports", exist_ok=True)
    report_df.to_csv(os.path.join("reports", "store_validity.csv"), index=False)
    reasons_summary(report_df).to_csv(os.path.join("reports", "store_validity_summary.csv"), index=False)

    with open(os.path.join("reports", "valid_store_ids.txt"), "w", encoding="utf-8") as f:
        for sid in valid_store_ids:
            f.write(f"{sid}\n")

    print(f"[3.1] Valid stores: {len(valid_store_ids)}/{len(report_df)}")
    if len(valid_store_ids) == 0:
        raise SystemExit("[ERROR] No valid stores found after filtering.")

    # pre-process per store
    print("[4] Per-store preprocess + save processed_store_<id>.csv")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    kept_frames = []

    for sid in valid_store_ids:
        out_store_path = os.path.join(PROCESSED_DIR, f"processed_store_{sid}.csv")

        if SKIP_EXISTING_PROCESSED and os.path.exists(out_store_path):
            store_df = pd.read_csv(out_store_path)
            if "timestamp" in store_df.columns:
                store_df["timestamp"] = pd.to_datetime(store_df["timestamp"])
                store_df = store_df.sort_values("timestamp").reset_index(drop=True)
            kept_frames.append(store_df)
            continue

        store_df = df_filtered.loc[df_filtered["Store"] == sid].copy()

        # just in case
        if store_df.empty:
            print(f"[WARN] Store {sid}: empty after filtering. Skipping.")
            continue

        # sort by date
        if "Date" in store_df.columns:
            store_df["Date"] = pd.to_datetime(store_df["Date"])
            store_df = store_df.sort_values("Date").reset_index(drop=True)

        store_df = clean_data(store_df, keep_closed_days=KEEP_CLOSED_DAYS)

        store_df = fix_mixed_types(store_df)
        store_df = add_time_features(store_df)
        store_df = to_chronos_df(store_df)
        store_df = _ensure_dayofweek(store_df)
        store_df = select_baseline_features(store_df)

        if "timestamp" in store_df.columns:
            store_df["timestamp"] = pd.to_datetime(store_df["timestamp"])
            store_df = store_df.sort_values("timestamp").reset_index(drop=True)

        save_processed(store_df, out_store_path)
        kept_frames.append(store_df)

    if not kept_frames:
        raise RuntimeError(
            "No per-store frames were produced. "
            "Check that PROCESSED_DIR is writable, valid_store_ids is non-empty, "
            "and clean_data/to_chronos_df are not dropping all rows."
        )

    all_processed = pd.concat(kept_frames, ignore_index=True)
    out_path = os.path.join(PROCESSED_DIR, "rossmann_allstores_processed.csv")
    save_processed(all_processed, out_path)

    print("[DONE] Saved per-store processed CSVs to:", PROCESSED_DIR)
    print("[DONE] Also saved:", out_path)
    print(all_processed.head(3))
