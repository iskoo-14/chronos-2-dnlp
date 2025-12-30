import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
from tqdm import tqdm

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
    filter_valid_stores,
)
from src.models.predict_model import (
    load_model,
    predict_df_univariate,
    predict_df_covariates,
    save_quantiles_csv,
)
from src.models.robustness import run_all_robustness_tests

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

    # MULTISTORE CONFIG (lightweight by default; flip flags for full ablation/robustness)
    STORES = []  # empty = all valid stores
    HORIZON = 30

    BEST_CONTEXT = 512
    RUN_ALL_CONTEXTS = False  # set True to run [128, 256, 512] ablation
    CONTEXT_LENGTHS = [128, 256, 512] if RUN_ALL_CONTEXTS else [BEST_CONTEXT]
    MIN_RUN = max(CONTEXT_LENGTHS) + HORIZON

    # Light defaults: skip heavy recomputation/robustness unless explicitly requested
    RUN_ROBUSTNESS = False  # set True to produce robustness CSVs per store (expensive)
    SKIP_EXISTING_PROCESSED = True
    SKIP_EXISTING_FORECASTS = True
    SKIP_EXISTING_ROBUSTNESS = True

    MIN_OBS = 600  # drop stores with fewer observed targets
    CHECK_RECENT_COVS = True  # drop stores with NaN covariates in recent window
    ZERO_TAIL_MAX = 30  # max consecutive zeros allowed in recent window
    ZERO_TAIL_SHARE = 0.4  # max share of zeros in recent window

    # Data regularity and eligibility checks
    ENFORCE_DAILY_FREQUENCY = True  # reindex each store to a daily calendar
    REQUIRE_RECENT_WINDOW = True    # require last (MIN_RUN) days to be continuous and observed

    reports_dir = os.path.join(base, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # DETERMINE STORE LIST 
    print("[INFO] Loading full dataset to determine store set...")
    df_all = load_raw_data(train_path, store_path, store_id=None)

    if ENFORCE_DAILY_FREQUENCY:
        print("[STEP] Enforcing daily frequency for all stores (calendar reindex)...")
        df_all = pd.concat(
            [
                enforce_daily_frequency_store(store_df, date_col="Date", store_col="Store")
                for _, store_df in df_all.groupby("Store")
            ],
            ignore_index=True,
        )

    df_all_filtered, report_df, valid_store_ids = filter_valid_stores(
        df_all,
        store_col="Store",
        date_col="Date",
        target_col="Sales",
        min_run=MIN_RUN,
        recent_window_length=MIN_RUN if REQUIRE_RECENT_WINDOW else None,
        min_obs=MIN_OBS,
        covariate_cols=["Open", "Promo", "SchoolHoliday", "StateHoliday", "Customers", "DayOfWeek"],
        check_recent_covariates=CHECK_RECENT_COVS,
        zero_tail_max=ZERO_TAIL_MAX,
        zero_tail_share=ZERO_TAIL_SHARE,
    )

    report_path = os.path.join(reports_dir, "store_validity.csv")
    report_df.to_csv(report_path, index=False)
    print(f"[INFO] Store validity report saved to {report_path}")

    if len(STORES) == 0:
        store_list = valid_store_ids
    else:
        store_list = [s for s in STORES if s in valid_store_ids]
        skipped = sorted(set(STORES) - set(store_list))
        if skipped:
            print(f"[WARN] Requested stores excluded due to validity filters: {skipped}")

    reason_counts = {}
    for reason_str in report_df.loc[~report_df["is_valid"], "reasons"]:
        for reason in [r for r in reason_str.split(";") if r]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    n_valid = len(store_list)
    n_total = len(report_df)
    print(f"[INFO] Valid stores: {n_valid}/{n_total}")
    for reason, count in reason_counts.items():
        print(f"[INFO] Discarded ({reason}): {count}")

    if len(store_list) == 0:
        print("[ERROR] No valid stores found after continuity checks. Exiting.")
        raise SystemExit(1)

    # LOAD MODEL ONCE 
    print("[STEP] Loading Chronos-2 model...")
    pipeline = load_model("amazon/chronos-2")

    # cache processed dfs per store to avoid re-running preprocessing across context ablations
    processed_by_store = {}

    print("[STEP] Preprocessing stores (clean + features)...")
    for STORE_ID in tqdm(store_list, desc="Preprocess stores", unit="store"):
        store_df = df_all_filtered[df_all_filtered["Store"] == STORE_ID].copy()

        processed_path = os.path.join(
            base, "src/data", f"processed_rossmann_store_{STORE_ID}.csv"
        )
        if SKIP_EXISTING_PROCESSED and os.path.exists(processed_path):
            store_df = pd.read_csv(processed_path)
            # ensure recent window continuity if required
            if REQUIRE_RECENT_WINDOW and not has_continuous_recent_window(
                store_df, window_length=MIN_RUN, date_col="timestamp", target_col="target"
            ):
                tqdm.write(
                    f"[SKIP] Store {STORE_ID}: existing processed file fails recent continuity check"
                )
                continue
            processed_by_store[STORE_ID] = store_df
            continue

        store_df = clean_data(store_df, keep_closed_days=True)
        store_df = add_time_features(store_df)
        store_df = to_chronos_df(store_df)

        store_df = select_important_features(store_df)
        store_df = fix_mixed_types(store_df)

        save_processed(store_df, processed_path, verbose=False)

        processed_by_store[STORE_ID] = store_df

        if RUN_ROBUSTNESS:
            robustness_files = [
                f"{name}_output_store_{STORE_ID}.csv"
                for name in [
                    "noise",
                    "strong_noise",
                    "shuffle",
                    "missing_future",
                    "time_shift",
                    "trend_break",
                    "feature_drop",
                    "partial_mask",
                    "scaling",
                    "long_horizon",
                ]
            ]
            root_out = output_dir
            if SKIP_EXISTING_ROBUSTNESS and all(
                os.path.exists(os.path.join(root_out, fname))
                for fname in robustness_files
            ):
                continue
            run_all_robustness_tests(pipeline, store_df, store_id=STORE_ID, verbose=False)

    store_list = list(processed_by_store.keys())
    if len(store_list) == 0:
        print("[ERROR] No stores left after preprocessing. Exiting.")
        raise SystemExit(1)

    # MULTISTORE LOOP 
    for CONTEXT_LEN in CONTEXT_LENGTHS:
        ctx_output_dir = os.path.join(output_dir, f"ctx_{CONTEXT_LEN}")
        ensure_output_folder(ctx_output_dir)
        print(f"[CTX {CONTEXT_LEN}] Forecasts for {len(store_list)} store(s)")

        for STORE_ID in tqdm(store_list, desc=f"CTX {CONTEXT_LEN} forecasts", unit="store"):

            df = processed_by_store[STORE_ID]

            df_past, df_test = temporal_split(df, test_size=HORIZON)

            if len(df_past) > CONTEXT_LEN:
                df_past = df_past.iloc[-CONTEXT_LEN:].reset_index(drop=True)

            gt_path = os.path.join(ctx_output_dir, f"ground_truth_store_{STORE_ID}.csv")
            pd.DataFrame({"y_true": df_test["target"].values}).to_csv(gt_path, index=False)

            # covariates sets (paper-style)
            PAST_ONLY_COVS = ["Customers"]
            KNOWN_FUTURE_COVS = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]

            uni_path = os.path.join(ctx_output_dir, f"univariate_store_{STORE_ID}.csv")
            cov_path = os.path.join(ctx_output_dir, f"covariate_store_{STORE_ID}.csv")
            gt_path = os.path.join(ctx_output_dir, f"ground_truth_store_{STORE_ID}.csv")

            if SKIP_EXISTING_FORECASTS and all(
                os.path.exists(p) for p in [uni_path, cov_path, gt_path]
            ):
                continue

            context_uni = df_past[["id", "timestamp", "target"]].copy()
            pred_uni = predict_df_univariate(pipeline, context_uni, horizon=HORIZON)
            save_quantiles_csv(
                pred_uni,
                uni_path,
                verbose=False,
            )

            context_cov = df_past[
                ["id", "timestamp", "target"] + PAST_ONLY_COVS + KNOWN_FUTURE_COVS
            ].copy()
            future_cov = df_test[["id", "timestamp"] + KNOWN_FUTURE_COVS].copy()

            pred_cov = predict_df_covariates(
                pipeline, context_cov, future_cov, horizon=HORIZON
            )
            save_quantiles_csv(
                pred_cov,
                cov_path,
                verbose=False,
            )

        # BACKWARD COMPATIBILITY (SINGLE STORE)
        if len(store_list) == 1:
            STORE_ID = store_list[0]
            pd.read_csv(
                os.path.join(ctx_output_dir, f"univariate_store_{STORE_ID}.csv")
            ).to_csv(os.path.join(ctx_output_dir, "univariate.csv"), index=False)

            pd.read_csv(
                os.path.join(ctx_output_dir, f"covariate_store_{STORE_ID}.csv")
            ).to_csv(os.path.join(ctx_output_dir, "covariate.csv"), index=False)

            pd.read_csv(
                os.path.join(ctx_output_dir, f"ground_truth_store_{STORE_ID}.csv")
            ).to_csv(os.path.join(ctx_output_dir, "ground_truth.csv"), index=False)

    print("===================================================")
    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("===================================================")
