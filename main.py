import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
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
    pred_df_to_quantiles,
)
from src.models.robustness import run_all_robustness_tests


def ensure_output_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DayOfWeek exists and has no NaN, based on timestamp."""
    if "timestamp" not in df.columns:
        return df
    if "DayOfWeek" not in df.columns or df["DayOfWeek"].isna().any():
        df = df.copy()
        df["DayOfWeek"] = pd.to_datetime(df["timestamp"]).dt.dayofweek + 1
    return df


def _numeric_series(s: pd.Series, default: float) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _num(s: pd.Series, default: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)


def _closed_run_len(open_series: pd.Series) -> pd.Series:
    closed = (_num(open_series, 1) == 0).astype(int)
    return closed.groupby((closed == 0).cumsum()).cumsum()


def _days_to_next_open(open_series: pd.Series, cap: int) -> pd.Series:
    o = _num(open_series, 1)
    days = np.full(len(o), np.nan)
    next_open = None
    for i in range(len(o) - 1, -1, -1):
        if o.iloc[i] == 1:
            next_open = 0
            days[i] = 0
        elif next_open is not None:
            next_open += 1
            days[i] = next_open
    return (
        pd.Series(days, index=o.index)
        .fillna(cap)
        .clip(upper=cap)
        .astype(int)
    )


def add_open_derived_features(context_cov, future_cov, horizon_cap: int):
    cap_val = max(1, int(horizon_cap))
    o_c = _num(context_cov["Open"], 1)
    o_f = _num(future_cov["Open"], 1)

    # Promo effettiva (promo solo se il negozio e aperto)
    if "Promo" in future_cov.columns:
        p_c = _num(context_cov["Promo"], 0)
        p_f = _num(future_cov["Promo"], 0)
        context_cov["PromoEff"] = p_c * o_c
        future_cov["PromoEff"] = p_f * o_f
        context_cov = context_cov.drop(columns=["Promo"])
        future_cov = future_cov.drop(columns=["Promo"])

    # Closed run length continua fra passato e futuro (cap per stabilizzare la scala)
    joined_open = pd.concat([o_c, o_f], ignore_index=True)
    joined_run = _closed_run_len(joined_open).clip(upper=cap_val)
    context_cov["ClosedRunLen"] = joined_run.iloc[: len(o_c)].values
    future_cov["ClosedRunLen"] = joined_run.iloc[len(o_c) :].values

    # Giorni alla prossima riapertura (solo futuro, no leakage; cap se non riapre)
    future_cov["DaysToNextOpen"] = _days_to_next_open(o_f, cap=cap_val)
    context_cov["DaysToNextOpen"] = 0

    # Open come categorica esplicita (stabile numerica)
    context_cov["OpenCat"] = (o_c == 1).astype(int)
    future_cov["OpenCat"] = (o_f == 1).astype(int)

    return context_cov, future_cov


if __name__ == "__main__":
    print("===================================================")
    print("=== DNLP PIPELINE STARTED (Chronos-2 Rossmann) ===")
    print("===================================================")

    base = os.path.dirname(os.path.abspath(__file__))
    output_dir = ensure_output_folder(os.path.join(base, "outputs"))
    forecast_root = ensure_output_folder(os.path.join(output_dir, "forecasts"))
    robustness_root = ensure_output_folder(os.path.join(output_dir, "robustness"))
    reports_dir = ensure_output_folder(os.path.join(base, "reports"))

    train_path = os.path.join(base, "src/data/train.csv")
    store_path = os.path.join(base, "src/data/store.csv")

    # ------------------------
    # CONFIG
    # ------------------------
    STORES = []  # empty = all valid stores
    HORIZON = 30

    BEST_CONTEXT = 512
    RUN_ALL_CONTEXTS = False
    CONTEXT_LENGTHS = [128, 256, 512] if RUN_ALL_CONTEXTS else [BEST_CONTEXT]
    MIN_RUN = max(CONTEXT_LENGTHS) + HORIZON

    RUN_ROBUSTNESS = False
    SKIP_EXISTING_PROCESSED = True
    SKIP_EXISTING_FORECASTS = False
    SKIP_EXISTING_ROBUSTNESS = True

    ENFORCE_DAILY_FREQUENCY = True
    REQUIRE_RECENT_WINDOW = True

    MIN_OBS = None
    CHECK_RECENT_COVS = False

    ZERO_TAIL_OPEN_MAX = 14
    ZERO_TAIL_OPEN_SHARE = 0.5

    CHECK_FUTURE_COVS = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]

    # Debug and coherence
    SAVE_FUTURE_DEBUG = True
    SANITIZE_PROMO_WHEN_CLOSED = True
    FORCE_ZERO_WHEN_CLOSED = False
    DIAG_COMPARE_FEATURES = False  # A/B covariate full vs base (per capire se le feature derivate entrano)
    DIAG_STORE_ID = None  # None = all

    # Covariate sets
    PAST_ONLY_COVS = ["Customers"]
    FUTURE_BASE_COVS = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]
    KNOWN_FUTURE_COVS = [
        "Open",
        "PromoEff",
        "SchoolHoliday",
        "StateHoliday",
        "DayOfWeek",
        "ClosedRunLen",
        "DaysToNextOpen",
        "OpenCat",
    ]
    CLOSED_RUN_CAP = 60

    # ------------------------
    # LOAD + VALIDATE STORES
    # ------------------------
    print("[INFO] Loading raw dataset...")
    df_all = load_raw_data(train_path, store_path, store_id=None)

    if ENFORCE_DAILY_FREQUENCY:
        print("[STEP] Enforcing daily frequency for all stores...")
        df_all = pd.concat(
            [
                enforce_daily_frequency_store(
                    store_df, date_col="Date", store_col="Store"
                )
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
        covariate_cols=[
            "Open",
            "Promo",
            "SchoolHoliday",
            "StateHoliday",
            "Customers",
            "DayOfWeek",
        ],
        check_recent_covariates=CHECK_RECENT_COVS,
        zero_tail_open_max=ZERO_TAIL_OPEN_MAX,
        zero_tail_open_share=ZERO_TAIL_OPEN_SHARE,
        check_future_covariates=CHECK_FUTURE_COVS if REQUIRE_RECENT_WINDOW else None,
    )

    report_path = os.path.join(reports_dir, "store_validity.csv")
    report_df.to_csv(report_path, index=False)
    print(f"[INFO] Store validity report saved to {report_path}")

    # store list
    if len(STORES) == 0:
        store_list = valid_store_ids
    else:
        store_list = [s for s in STORES if s in valid_store_ids]
        skipped = sorted(set(STORES) - set(store_list))
        if skipped:
            print(f"[WARN] Requested stores excluded by filters: {skipped}")

    # reasons summary
    reason_counts: dict[str, int] = {}
    for reason_str in report_df.loc[~report_df["is_valid"], "reasons"].fillna(""):
        for reason in [r for r in str(reason_str).split(";") if r]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    summary_path = os.path.join(reports_dir, "store_validity_summary.csv")
    pd.DataFrame(
        [{"reason": r, "count": c} for r, c in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))]
    ).to_csv(summary_path, index=False)

    print(f"[INFO] Valid stores: {len(store_list)}/{len(report_df)}")
    print(f"[INFO] Validity summary saved to {summary_path}")

    if len(store_list) == 0:
        raise SystemExit("[ERROR] No valid stores found after filtering.")

    # ------------------------
    # LOAD MODEL
    # ------------------------
    print("[STEP] Loading Chronos-2 model...")
    pipeline = load_model("amazon/chronos-2")

    # ------------------------
    # PREPROCESS PER STORE
    # ------------------------
    processed_by_store: dict[int, pd.DataFrame] = {}

    print("[STEP] Preprocessing stores...")
    for STORE_ID in tqdm(store_list, desc="Preprocess stores", unit="store"):
        store_df = df_all_filtered[df_all_filtered["Store"] == STORE_ID].copy()

        processed_path = os.path.join(base, "src/data", f"processed_rossmann_store_{STORE_ID}.csv")

        if SKIP_EXISTING_PROCESSED and os.path.exists(processed_path):
            dfp = pd.read_csv(processed_path)
            if "timestamp" in dfp.columns:
                dfp["timestamp"] = pd.to_datetime(dfp["timestamp"])
                dfp = dfp.sort_values("timestamp").reset_index(drop=True)

            dfp = select_important_features(dfp)
            dfp = fix_mixed_types(dfp)
            dfp = _ensure_dayofweek(dfp)

            if REQUIRE_RECENT_WINDOW and not has_continuous_recent_window(
                dfp, window_length=MIN_RUN, date_col="timestamp", target_col="target"
            ):
                tqdm.write(f"[SKIP] Store {STORE_ID}: processed fails recent continuity check")
                continue

            processed_by_store[STORE_ID] = dfp

            # Even when reusing processed data, optionally run robustness
            if RUN_ROBUSTNESS:
                robustness_ctx_dir = ensure_output_folder(os.path.join(robustness_root, f"ctx_{BEST_CONTEXT}"))
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
                if SKIP_EXISTING_ROBUSTNESS and all(
                    os.path.exists(os.path.join(robustness_ctx_dir, f)) for f in robustness_files
                ):
                    continue

                run_all_robustness_tests(
                    pipeline,
                    dfp,
                    store_id=STORE_ID,
                    context_len=BEST_CONTEXT,
                    output_root=robustness_ctx_dir,
                    verbose=False,
                )

            continue

        store_df = clean_data(store_df, keep_closed_days=True)
        store_df = add_time_features(store_df)
        store_df = to_chronos_df(store_df)
        store_df = _ensure_dayofweek(store_df)

        store_df = select_important_features(store_df)
        store_df = fix_mixed_types(store_df)

        save_processed(store_df, processed_path, verbose=False)
        processed_by_store[STORE_ID] = store_df

        if RUN_ROBUSTNESS:
            robustness_ctx_dir = ensure_output_folder(os.path.join(robustness_root, f"ctx_{BEST_CONTEXT}"))
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
            if SKIP_EXISTING_ROBUSTNESS and all(os.path.exists(os.path.join(robustness_ctx_dir, f)) for f in robustness_files):
                continue

            run_all_robustness_tests(
                pipeline,
                store_df,
                store_id=STORE_ID,
                context_len=BEST_CONTEXT,
                output_root=robustness_ctx_dir,
                verbose=False,
            )

    store_list = list(processed_by_store.keys())
    if len(store_list) == 0:
        raise SystemExit("[ERROR] No stores left after preprocessing.")

    # ------------------------
    # FORECASTING
    # ------------------------
    mae_rows: list[dict] = []  # track MAE per store (all/open/closed)
    for CONTEXT_LEN in CONTEXT_LENGTHS:
        ctx_output_dir = ensure_output_folder(os.path.join(forecast_root, f"ctx_{CONTEXT_LEN}"))
        print(f"[CTX {CONTEXT_LEN}] Forecasts for {len(store_list)} store(s)")

        for STORE_ID in tqdm(store_list, desc=f"CTX {CONTEXT_LEN} forecasts", unit="store"):
            df = processed_by_store[STORE_ID]

            df_past, df_test = temporal_split(df, test_size=HORIZON)

            if len(df_past) > CONTEXT_LEN:
                df_past = df_past.iloc[-CONTEXT_LEN:].reset_index(drop=True)

            uni_path = os.path.join(ctx_output_dir, f"univariate_store_{STORE_ID}.csv")
            cov_path = os.path.join(ctx_output_dir, f"covariate_store_{STORE_ID}.csv")
            gt_path = os.path.join(ctx_output_dir, f"ground_truth_store_{STORE_ID}.csv")
            dbg_path = os.path.join(ctx_output_dir, f"future_debug_store_{STORE_ID}.csv")

            if SKIP_EXISTING_FORECASTS and all(os.path.exists(p) for p in [uni_path, cov_path, gt_path]):
                continue

            # ------------------------
            # UNIVARIATE
            # ------------------------
            context_uni = df_past[["id", "timestamp", "target"]].copy()

            if context_uni["target"].isna().any():
                tqdm.write(f"[SKIP] Store {STORE_ID}: NaN in context target")
                continue

            if (not context_uni["timestamp"].is_monotonic_increasing) or context_uni["timestamp"].duplicated().any():
                tqdm.write(f"[SKIP] Store {STORE_ID}: non-monotonic or duplicate context timestamps")
                continue

            pred_uni = predict_df_univariate(pipeline, context_uni, horizon=HORIZON)
            if "timestamp" not in pred_uni.columns and len(pred_uni) == len(df_test):
                pred_uni = pred_uni.copy()
                pred_uni["timestamp"] = df_test["timestamp"].values

            save_quantiles_csv(pred_uni, uni_path, verbose=False)

            # Save GT after at least one forecast exists
            gt_cols = {"timestamp": df_test["timestamp"], "y_true": df_test["target"]}
            if "Open" in df_test.columns:
                gt_cols["Open"] = df_test["Open"]
            pd.DataFrame(gt_cols).to_csv(gt_path, index=False)

            # ------------------------
            # COVARIATES
            # ------------------------
            context_cov = df_past[["id", "timestamp", "target"] + PAST_ONLY_COVS + FUTURE_BASE_COVS].copy()
            future_cov = df_test[["id", "timestamp"] + FUTURE_BASE_COVS].copy()

            context_cov = _ensure_dayofweek(context_cov)
            future_cov = _ensure_dayofweek(future_cov)

            # Save debug future window
            if SAVE_FUTURE_DEBUG:
                dbg_cols = [c for c in ["timestamp", "target", "Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek", "Customers"] if c in df_test.columns]
                dbg = df_test[dbg_cols].copy()
                if "target" in dbg.columns:
                    dbg = dbg.rename(columns={"target": "y_true"})
                if "Open" in dbg.columns:
                    dbg["is_closed"] = (_numeric_series(dbg["Open"], 1).eq(0)).astype(int)
                if "Open" in dbg.columns and "Promo" in dbg.columns:
                    dbg["promo_when_closed"] = (
                        _numeric_series(dbg["Open"], 1).eq(0) & _numeric_series(dbg["Promo"], 0).eq(1)
                    ).astype(int)
                dbg.to_csv(dbg_path, index=False)

            # Coherence: if Open==0 then Promo must be 0 (both context and future)
            if SANITIZE_PROMO_WHEN_CLOSED and ("Open" in future_cov.columns) and ("Promo" in future_cov.columns):
                fut_open = _numeric_series(future_cov["Open"], 1)
                fut_closed = fut_open.eq(0)
                if fut_closed.any():
                    future_cov.loc[fut_closed, "Promo"] = 0

                if ("Open" in context_cov.columns) and ("Promo" in context_cov.columns):
                    ctx_open = _numeric_series(context_cov["Open"], 1)
                    ctx_closed = ctx_open.eq(0)
                    if ctx_closed.any():
                        context_cov.loc[ctx_closed, "Promo"] = 0

            # Keep a copy of base covariates (before adding derived features) for A/B diagnostics.
            base_context_cov = context_cov.copy()
            base_future_cov = future_cov.copy()

            # Derived covariates to make closures/promos more salient
            context_cov, future_cov = add_open_derived_features(
                context_cov,
                future_cov,
                horizon_cap=max(HORIZON, CLOSED_RUN_CAP),
            )

            # Optional A/B test: confronto forecast con/ senza feature derivate
            if DIAG_COMPARE_FEATURES and (DIAG_STORE_ID is None or STORE_ID == DIAG_STORE_ID):
                pred_cov_full = predict_df_covariates(pipeline, context_cov, future_cov, horizon=HORIZON)

                base_cols_ctx = ["id", "timestamp", "target"] + PAST_ONLY_COVS + FUTURE_BASE_COVS
                base_cols_fut = ["id", "timestamp"] + FUTURE_BASE_COVS

                pred_cov_base = predict_df_covariates(
                    pipeline,
                    base_context_cov[base_cols_ctx],
                    base_future_cov[base_cols_fut],
                    horizon=HORIZON,
                )

                from src.models.predict_model import pred_df_to_quantiles

                _, med_full, _ = pred_df_to_quantiles(pred_cov_full)
                _, med_base, _ = pred_df_to_quantiles(pred_cov_base)

                diff_all = float(np.max(np.abs(med_full - med_base)))

                if "Open" in df_test.columns:
                    closed_mask = _numeric_series(df_test["Open"], 1).eq(0).values
                    if closed_mask.any():
                        diff_closed = float(np.max(np.abs((med_full - med_base)[closed_mask])))
                    else:
                        diff_closed = 0.0
                else:
                    diff_closed = 0.0

                tqdm.write(
                    f"[DIAG] Store {STORE_ID} ctx={CONTEXT_LEN} max_abs_diff_median all={diff_all:.6f} closed={diff_closed:.6f}"
                )

            # Optional debug of derived covariates
            if SAVE_FUTURE_DEBUG:
                dbg2_cols = [
                    c
                    for c in [
                        "timestamp",
                        "Open",
                        "PromoEff",
                        "ClosedRunLen",
                        "DaysToNextOpen",
                        "OpenCat",
                    ]
                    if c in future_cov.columns
                ]
                dbg2_path = os.path.join(ctx_output_dir, f"future_debug_derived_store_{STORE_ID}.csv")
                future_cov[dbg2_cols].to_csv(dbg2_path, index=False)

            # Sanity checks
            if context_cov[PAST_ONLY_COVS + KNOWN_FUTURE_COVS].isna().any().any():
                tqdm.write(f"[SKIP] Store {STORE_ID}: NaN in context covariates")
                continue
            if future_cov[KNOWN_FUTURE_COVS].isna().any().any():
                tqdm.write(f"[SKIP] Store {STORE_ID}: NaN in future covariates")
                continue
            if (not future_cov["timestamp"].is_monotonic_increasing) or future_cov["timestamp"].duplicated().any():
                tqdm.write(f"[SKIP] Store {STORE_ID}: non-monotonic or duplicate future timestamps")
                continue

            pred_cov = predict_df_covariates(pipeline, context_cov, future_cov, horizon=HORIZON)
            if "timestamp" not in pred_cov.columns and len(pred_cov) == len(df_test):
                pred_cov = pred_cov.copy()
                pred_cov["timestamp"] = df_test["timestamp"].values

            # Optional hard rule: Open==0 => forecast==0
            if FORCE_ZERO_WHEN_CLOSED and ("Open" in df_test.columns) and (len(pred_cov) == len(df_test)):
                closed_mask = _numeric_series(df_test["Open"], 1).eq(0).values
                if closed_mask.any():
                    pred_cov = pred_cov.copy()
                    for col in ["p10", "p50", "p90", "median", "0.1", "0.5", "0.9", "predictions"]:
                        if col in pred_cov.columns:
                            pred_cov.loc[closed_mask, col] = 0.0

            save_quantiles_csv(pred_cov, cov_path, verbose=False)

            # MAE diagnostics per store (all / open / closed)
            median_col = next((c for c in ["median", "0.5", "p50"] if c in pred_cov.columns), None)
            if median_col is not None and "target" in df_test.columns:
                y_true = pd.to_numeric(df_test["target"], errors="coerce").to_numpy(dtype=float)
                y_hat = pd.to_numeric(pred_cov[median_col], errors="coerce").to_numpy(dtype=float)
                open_series = _numeric_series(df_test["Open"], 1) if "Open" in df_test.columns else pd.Series(np.ones_like(y_true))
                open_mask = open_series.eq(1).to_numpy()
                closed_mask = open_series.eq(0).to_numpy()

                mae_all = float(np.mean(np.abs(y_true - y_hat)))
                mae_open = float(np.mean(np.abs(y_true[open_mask] - y_hat[open_mask]))) if open_mask.any() else None
                mae_closed = float(np.mean(np.abs(y_true[closed_mask] - y_hat[closed_mask]))) if closed_mask.any() else None

                mae_rows.append(
                    {
                        "store_id": STORE_ID,
                        "context_len": CONTEXT_LEN,
                        "mae_all": mae_all,
                        "mae_open": mae_open,
                        "mae_closed": mae_closed,
                        "n_open": int(open_mask.sum()),
                        "n_closed": int(closed_mask.sum()),
                    }
                )

        # Single-store convenience copy
        if len(store_list) == 1:
            sid = store_list[0]
            mapping = [
                (os.path.join(ctx_output_dir, f"univariate_store_{sid}.csv"), os.path.join(ctx_output_dir, "univariate.csv")),
                (os.path.join(ctx_output_dir, f"covariate_store_{sid}.csv"), os.path.join(ctx_output_dir, "covariate.csv")),
                (os.path.join(ctx_output_dir, f"ground_truth_store_{sid}.csv"), os.path.join(ctx_output_dir, "ground_truth.csv")),
            ]
            for src, dst in mapping:
                if os.path.exists(src):
                    pd.read_csv(src).to_csv(dst, index=False)

    if mae_rows:
        mae_path = os.path.join(reports_dir, "mae_open_closed.csv")
        pd.DataFrame(mae_rows).to_csv(mae_path, index=False)
        print(f"[INFO] MAE open/closed report saved to {mae_path}")

    print("===================================================")
    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("===================================================")
