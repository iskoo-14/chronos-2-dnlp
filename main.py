import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd

from src.data.make_dataset import (
    load_raw_data,
    clean_data,
    add_time_features,
    select_important_features,
    fix_mixed_types,
    to_chronos_df,
    temporal_split,
    save_processed,
    enforce_daily_frequency_store,
    has_continuous_recent_window,
)
from src.models.predict_model import (
    load_model,
    predict_df_univariate,
    predict_df_covariates,
    save_quantiles_csv,
)

def ensure_output_folder(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    print("===================================================")
    print("=== DNLP PIPELINE STARTED (Chronos quickstart aligned) ===")
    print("===================================================")

    base = os.path.dirname(os.path.abspath(__file__))
    output_dir = ensure_output_folder(os.path.join(base, "outputs"))

    train_path = os.path.join(base, "src/data/train.csv")
    store_path = os.path.join(base, "src/data/store.csv")

    # MULTISTORE CONFIG
    STORES = []

    HORIZON = 30
    CONTEXT_LEN = 256

    # Data regularity and eligibility checks
    ENFORCE_DAILY_FREQUENCY = True  # reindex each store to a daily calendar
    REQUIRE_RECENT_WINDOW = True    # require last (CONTEXT_LEN + HORIZON) days to be continuous and observed

    # DETERMINE STORE LIST 
    if len(STORES) == 0:
        print("[INFO] Detecting all stores...")
        df_all = load_raw_data(train_path, store_path, store_id=None)
        store_list = sorted(df_all["Store"].unique())
    else:
        store_list = STORES

    print(f"[INFO] Running pipeline for {len(store_list)} store(s)")

    # LOAD MODEL ONCE 
    print("[STEP] Loading Chronos-2 model...")
    pipeline = load_model("amazon/chronos-2")

    # MULTISTORE LOOP 
    for STORE_ID in store_list:

        print("\n---------------------------------------------------")
        print(f"[STORE {STORE_ID}] Starting pipeline")
        print("---------------------------------------------------")

        print("[STEP] Loading raw data (single store)...")
        df = load_raw_data(train_path, store_path, store_id=STORE_ID)

        if ENFORCE_DAILY_FREQUENCY:
            print("[STEP] Enforcing daily frequency (calendar reindex)...")
            df = enforce_daily_frequency_store(df, date_col="Date", store_col="Store")

        print("[STEP] Cleaning (keep closed days)...")
        df = clean_data(df, keep_closed_days=True)

        print("[STEP] Adding time features...")
        df = add_time_features(df)

        print("[STEP] Fixing mixed types...")
        df = fix_mixed_types(df)

        print("[STEP] Converting to Chronos dataframe format...")
        df = to_chronos_df(df)

        if REQUIRE_RECENT_WINDOW:
            needed = CONTEXT_LEN + HORIZON
            if not has_continuous_recent_window(
                df, window_length=needed, date_col="timestamp", target_col="target"
            ):
                print(f"[SKIP] Store {STORE_ID}: last {needed} days are not continuous daily data with observed targets")
                continue

        print("[STEP] Selecting paper-style covariates...")
        df = select_important_features(df)

        processed_path = os.path.join(
            base, "src/data", f"processed_rossmann_store_{STORE_ID}.csv"
        )
        save_processed(df, processed_path)

        print("[STEP] Temporal split: last 30 days as ground truth...")
        df_past, df_test = temporal_split(df, test_size=HORIZON)

        print(f"[INFO] Past length: {len(df_past)} | Test length: {len(df_test)}")

        # keep last CONTEXT_LEN as context
        if len(df_past) > CONTEXT_LEN:
            df_past = df_past.iloc[-CONTEXT_LEN:].reset_index(drop=True)
            print(f"[INFO] Context truncated to last {CONTEXT_LEN} rows")

        gt_path = os.path.join(output_dir, f"ground_truth_store_{STORE_ID}.csv")
        pd.DataFrame({"y_true": df_test["target"].values}).to_csv(gt_path, index=False)
        print(f"[INFO] Saved ground truth: {gt_path}")

        # covariates sets (paper-style)
        PAST_ONLY_COVS = ["Customers"]
        KNOWN_FUTURE_COVS = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]

        # UNIVARIATE
        print("[STEP] Running univariate forecast...")
        context_uni = df_past[["id", "timestamp", "target"]].copy()
        pred_uni = predict_df_univariate(pipeline, context_uni, horizon=HORIZON)
        save_quantiles_csv(
            pred_uni,
            os.path.join(output_dir, f"univariate_store_{STORE_ID}.csv"),
        )

        # COVARIATE (ICL) FORECAST
        print("[STEP] Running covariate forecast (predict_df)...")
        context_cov = df_past[
            ["id", "timestamp", "target"] + PAST_ONLY_COVS + KNOWN_FUTURE_COVS
        ].copy()
        future_cov = df_test[["id", "timestamp"] + KNOWN_FUTURE_COVS].copy()

        pred_cov = predict_df_covariates(
            pipeline, context_cov, future_cov, horizon=HORIZON
        )
        save_quantiles_csv(
            pred_cov,
            os.path.join(output_dir, f"covariate_store_{STORE_ID}.csv"),
        )

        print(f"[STORE {STORE_ID}] Completed")

    # BACKWARD COMPATIBILITY (SINGLE STORE)
    if len(store_list) == 1:
        STORE_ID = store_list[0]
        pd.read_csv(
            os.path.join(output_dir, f"univariate_store_{STORE_ID}.csv")
        ).to_csv(os.path.join(output_dir, "univariate.csv"), index=False)

        pd.read_csv(
            os.path.join(output_dir, f"covariate_store_{STORE_ID}.csv")
        ).to_csv(os.path.join(output_dir, "covariate.csv"), index=False)

        pd.read_csv(
            os.path.join(output_dir, f"ground_truth_store_{STORE_ID}.csv")
        ).to_csv(os.path.join(output_dir, "ground_truth.csv"), index=False)

    print("===================================================")
    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("===================================================")
