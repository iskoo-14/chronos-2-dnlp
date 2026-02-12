import os
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PROCESSED_DIR, HORIZON, CONTEXT_LENGTHS, SAVE_FUTURE_DEBUG, PAST_ONLY_COVS, FUTURE_KNOWN_COVS, SKIP_EXISTING_FORECASTS, SKIP_EXISTING_GT_DEBUG
from data.make_dataset import temporal_split
from models.chronos import load_model
from models.univariate import predict_df_univariate, save_quantiles_csv
from models.covariate import predict_df_covariates

# helper functions 

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def make_ctx_dirs(project_root: str, context_len: int) -> dict:
    base = ensure_dir(os.path.join(project_root, "outputs", "baseline", "forecasts", f"ctx_{context_len}"))
    return {
        "univariate_predictions": ensure_dir(os.path.join(base, "univariate", "predictions")),
        "covariate_predictions": ensure_dir(os.path.join(base, "covariate", "predictions")),
    }

def make_global_dirs(project_root: str) -> dict:
    base = ensure_dir(os.path.join(project_root, "outputs", "baseline"))
    return {
        "gt_dir": ensure_dir(os.path.join(base, "ground_truth")),
        "dbg_dir": ensure_dir(os.path.join(base, "debug")),
    }

def read_valid_store_ids(path: str = os.path.join("reports", "valid_store_ids.txt")) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run run_preprocessing_0.py first to generate store selection outputs."
        )
    ids: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(line))
    return ids


def read_processed_store(store_id: int) -> pd.DataFrame:
    p = os.path.join(PROCESSED_DIR, f"processed_store_{store_id}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing processed file: {p}. Run run_preprocessing_0.py first.")
    df = pd.read_csv(p)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def build_future_debug(df_test: pd.DataFrame) -> pd.DataFrame:
    dbg_cols = [
        c for c in [
            "timestamp",
            "target",
            "Open",
            "Promo",
            "SchoolHoliday",
            "StateHoliday",
            "DayOfWeek",
            "Customers",
        ]
        if c in df_test.columns
    ]
    dbg = df_test[dbg_cols].copy()
    if "target" in dbg.columns:
        dbg = dbg.rename(columns={"target": "y_true"})
    return dbg

def ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns and ("DayOfWeek" not in df.columns or df["DayOfWeek"].isna().any()):
        df = df.copy()
        df["DayOfWeek"] = pd.to_datetime(df["timestamp"]).dt.dayofweek + 1
    return df


if __name__ == "__main__":
    print("===================================================")
    print("=== BASELINE RUN (UNIVARIATE + COVARIATE) ===")
    print("===================================================")

    project_root = os.path.dirname(os.path.abspath(__file__))
    global_dirs = make_global_dirs(project_root)
    gt_dir = global_dirs["gt_dir"]
    dbg_dir = global_dirs["dbg_dir"]

    # Load store list (from preprocessing step)
    store_ids = read_valid_store_ids()
    print(f"[INFO] Stores to forecast: {len(store_ids)}")

    # Load model
    pipeline = load_model("amazon/chronos-2")
    
    for ctx_len in CONTEXT_LENGTHS:
        dirs = make_ctx_dirs(project_root, ctx_len)

        uni_pred_dir = dirs["univariate_predictions"]
        cov_pred_dir = dirs["covariate_predictions"]
        

        print(f"\n[CTX {ctx_len}] Forecasting {len(store_ids)} store(s)")
        print(f"[INFO] Saving to: {os.path.join(project_root, 'outputs', 'baseline', 'forecasts', f'ctx_{ctx_len}')}")

        for sid in store_ids:
            uni_pred_path = os.path.join(uni_pred_dir, f"forecast_store_{sid}.csv")
            cov_pred_path = os.path.join(cov_pred_dir, f"forecast_store_{sid}.csv")

            # skip existing forecasts to speed up the pipeline
            if SKIP_EXISTING_FORECASTS and os.path.exists(uni_pred_path) and os.path.exists(cov_pred_path):
                continue

            df = read_processed_store(sid)
            df = ensure_dayofweek(df)

            df_past, df_test = temporal_split(df, test_size=HORIZON)
            
            if len(df_past) > ctx_len:
                df_past = df_past.iloc[-ctx_len:].reset_index(drop=True)

            df_past = ensure_dayofweek(df_past)
            df_test = ensure_dayofweek(df_test)

            # save GT
            gt_path = os.path.join(gt_dir, f"ground_truth_store_{sid}.csv")
            if not SKIP_EXISTING_GT_DEBUG or not os.path.exists(gt_path):
                gt_cols = {"timestamp": df_test["timestamp"], "y_true": df_test["target"]}
                if "Open" in df_test.columns:
                    gt_cols["Open"] = df_test["Open"]
                pd.DataFrame(gt_cols).to_csv(gt_path, index=False)

            # save debug future window
            if SAVE_FUTURE_DEBUG:
                dbg_path = os.path.join(dbg_dir, f"future_debug_store_{sid}.csv")
                if SAVE_FUTURE_DEBUG and (not SKIP_EXISTING_GT_DEBUG or not os.path.exists(dbg_path)):
                    dbg = build_future_debug(df_test)
                    dbg.to_csv(dbg_path, index=False)

            # UNIVARIATE
            if not (SKIP_EXISTING_FORECASTS and os.path.exists(uni_pred_path)):
                context_uni = df_past[["id", "timestamp", "target"]].copy()

                # sanity checks
                if context_uni["target"].isna().any():
                    print(f"[SKIP] Store {sid}: NaN in context target (univariate)")
                elif (not context_uni["timestamp"].is_monotonic_increasing) or context_uni["timestamp"].duplicated().any():
                    print(f"[SKIP] Store {sid}: bad context timestamps (univariate)")
                else:
                    pred_uni = predict_df_univariate(pipeline, context_uni, horizon=HORIZON)

                    # if pred has no timestamp, align with df_test
                    if "timestamp" not in pred_uni.columns and len(pred_uni) == len(df_test):
                        pred_uni = pred_uni.copy()
                        pred_uni["timestamp"] = df_test["timestamp"].values

                    save_quantiles_csv(pred_uni, uni_pred_path, verbose=False)
            else:
                print(f"[SKIP] Store {sid} ctx={ctx_len}: univariate forecast exists")

            # COVARIATE
            if not (SKIP_EXISTING_FORECASTS and os.path.exists(cov_pred_path)):
                needed_ctx = ["id", "timestamp", "target"] + PAST_ONLY_COVS + FUTURE_KNOWN_COVS
                needed_fut = ["id", "timestamp"] + FUTURE_KNOWN_COVS

                missing_ctx = [c for c in needed_ctx if c not in df_past.columns]
                missing_fut = [c for c in needed_fut if c not in df_test.columns]
                if missing_ctx or missing_fut:
                    print(f"[SKIP] Store {sid}: missing cov columns ctx={missing_ctx} fut={missing_fut}")
                else:
                    context_cov = df_past[needed_ctx].copy()
                    future_cov = df_test[needed_fut].copy()

                    # sanity: timestamps
                    if (not context_cov["timestamp"].is_monotonic_increasing) or context_cov["timestamp"].duplicated().any():
                        print(f"[SKIP] Store {sid}: bad context timestamps (covariate)")
                    elif (not future_cov["timestamp"].is_monotonic_increasing) or future_cov["timestamp"].duplicated().any():
                        print(f"[SKIP] Store {sid}: bad future timestamps (covariate)")
                    else:
                        if "Open" in future_cov.columns and "Promo" in future_cov.columns:
                            fut_open = pd.to_numeric(future_cov["Open"], errors="coerce").fillna(1)
                            future_cov.loc[fut_open.eq(0), "Promo"] = 0
                        if "Open" in context_cov.columns and "Promo" in context_cov.columns:
                            ctx_open = pd.to_numeric(context_cov["Open"], errors="coerce").fillna(1)
                            context_cov.loc[ctx_open.eq(0), "Promo"] = 0

                        # NaN check
                        if context_cov.isna().any().any():
                            print(f"[SKIP] Store {sid}: NaN in cov context")
                        elif future_cov.isna().any().any():
                            print(f"[SKIP] Store {sid}: NaN in cov future")
                        else:
                            pred_cov = predict_df_covariates(pipeline, context_cov, future_cov, horizon=HORIZON)

                            # align timestamps if needed
                            if "timestamp" not in pred_cov.columns and len(pred_cov) == len(df_test):
                                pred_cov = pred_cov.copy()
                                pred_cov["timestamp"] = df_test["timestamp"].values

                            save_quantiles_csv(pred_cov, cov_pred_path, verbose=False)
                            
            else:
                print(f"[SKIP] Store {sid} ctx={ctx_len}: covariate forecast exists")

        print(f"[INFO] Done ctx={ctx_len}.")

    print("\n===================================================")
    print("=== BASELINE UNIVARIATE + COVARIATE COMPLETED ===")
    print("===================================================")