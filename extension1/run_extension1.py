# ext1_run.py
import os
import argparse
import pandas as pd
import re
import shutil

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import HORIZON, SKIP_EXISTING_FORECASTS
from data.make_dataset import temporal_split
from models.chronos import load_model
from models.univariate import save_quantiles_csv
from models.covariate import predict_df_covariates

from extension1.feature_utilities import extension1_covariate_sets

# evaluation imports
from evaluation.io import ensure_dir as eval_ensure_dir
from evaluation.compare_results import compute_wql_per_store, summarize_wql, write_comparison_report
from evaluation.select_best_context import select_best_context, print_best_context
from evaluation.metrics import compute_mae_open_closed


EXPERIMENT = "extension1"
CTX_LEN = 512
PROCESSED_DIR_EXT1 = os.path.join("data", "extension1")
TEST_IDS_PATH = os.path.join("data", "extension3", "splits", "test_store_ids.txt")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_ctx_dirs(project_root: str, context_len: int) -> dict:
    base = ensure_dir(os.path.join(project_root, "outputs", EXPERIMENT, "forecasts", f"ctx_{context_len}"))
    cov_dir = ensure_dir(os.path.join(base, "covariate"))
    return {"covariate_dir": cov_dir}


def read_processed_store(store_id: int, processed_dir: str) -> pd.DataFrame:
    p = os.path.join(processed_dir, f"processed_store_{store_id}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing processed file: {p}")
    df = pd.read_csv(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns and ("DayOfWeek" not in df.columns or df["DayOfWeek"].isna().any()):
        df = df.copy()
        df["DayOfWeek"] = pd.to_datetime(df["timestamp"]).dt.dayofweek + 1
    return df


def normalize_pred_columns(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    if "0.1" in pred.columns and "0.5" in pred.columns and "0.9" in pred.columns:
        pred = pred.rename(columns={"0.1": "p10", "0.5": "p50", "0.9": "p90"})
    elif "q0.1" in pred.columns and "q0.5" in pred.columns and "q0.9" in pred.columns:
        pred = pred.rename(columns={"q0.1": "p10", "q0.5": "p50", "q0.9": "p90"})
    elif "p10" in pred.columns and "p50" in pred.columns and "p90" in pred.columns:
        pass
    elif "p10" in pred.columns and "median" in pred.columns and "p90" in pred.columns:
        pred["p50"] = pred["median"]
    else:
        raise ValueError(f"Unknown prediction quantile format. Columns={list(pred.columns)}")

    if "median" not in pred.columns and "p50" in pred.columns:
        pred["median"] = pred["p50"]

    keep = [c for c in ["timestamp", "p10", "p50", "p90", "median"] if c in pred.columns]
    return pred[keep].copy()


def read_ids_txt(path: str) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ids file: {path}")
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(int(s))
    return sorted(set(ids))


def materialize_test_only_outputs(experiment: str, test_ids: list[int]) -> str:
    src_root = os.path.join("outputs", experiment)
    src_forecasts = os.path.join(src_root, "forecasts")
    src_gt = os.path.join(src_root, "ground_truth")

    test_experiment = f"{experiment}__test_only"
    dst_root = ensure_dir(os.path.join("outputs", test_experiment))
    dst_forecasts = ensure_dir(os.path.join(dst_root, "forecasts"))
    dst_gt = ensure_dir(os.path.join(dst_root, "ground_truth"))

    test_set = set(test_ids)

    for sid in test_ids:
        src = os.path.join(src_gt, f"ground_truth_store_{sid}.csv")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dst_gt, f"ground_truth_store_{sid}.csv"))

    for root, _, files in os.walk(src_forecasts):
        rel = os.path.relpath(root, src_forecasts)
        dst_dir = dst_forecasts if rel == "." else ensure_dir(os.path.join(dst_forecasts, rel))
        for fn in files:
            m = re.search(r"forecast_store_(\d+)\.csv$", fn)
            if not m:
                continue
            sid = int(m.group(1))
            if sid not in test_set:
                continue
            shutil.copyfile(os.path.join(root, fn), os.path.join(dst_dir, fn))

    return test_experiment


def run_evaluation(experiment: str, outlier_threshold: float, no_outlier_filter: bool, show_per_store_lines: bool):
    forecasts_root = os.path.join("outputs", experiment, "forecasts")
    gt_dir = os.path.join("outputs", experiment, "ground_truth")
    reports_dir = eval_ensure_dir(os.path.join("reports", experiment))

    records, text_report = compute_wql_per_store(
        forecasts_root=forecasts_root,
        gt_dir=gt_dir,
        include_store_lines=show_per_store_lines,
    )

    apply_filter = (not no_outlier_filter)

    try:
        per_store_path, by_ctx_path, summary_path, grouped_df, filter_note = summarize_wql(
            records=records,
            reports_dir=reports_dir,
            apply_outlier_filter=apply_filter,
            outlier_threshold=outlier_threshold,
        )
    except UnboundLocalError as e:
        print(f"[WARN] summarize_wql crashed: {e}")
        grouped_df, filter_note = None, "(no summary)"

    if grouped_df is not None and not grouped_df.empty:
        text_report.append(f"=== Context summary {filter_note} ===")
        for _, row in grouped_df.sort_values(["context_length", "mode"]).iterrows():
            std = 0.0 if pd.isna(row["std_wql"]) else float(row["std_wql"])
            text_report.append(
                f"CTX {row['context_length']} {row['mode']}: "
                f"mean_wql={row['mean_wql']:.4f} std_wql={std:.4f} "
                f"mean_mae={row['mean_mae']:.2f} mean_rmse={row['mean_rmse']:.2f} "
                f"p10_under={row['mean_p10_under']:.2f} p90_over={row['mean_p90_over']:.2f} "
                f"n_stores={int(row['n_stores'])}"
            )

    best_summary = select_best_context(reports_dir=reports_dir)
    print_best_context(best_summary)

    out_txt = write_comparison_report(reports_dir=reports_dir, text_lines=text_report)
    print(f"[INFO] Saved {out_txt}")

    compute_mae_open_closed(forecasts_root=forecasts_root, gt_dir=gt_dir, reports_dir=reports_dir)


def main(args):
    project_root = os.path.dirname(os.path.abspath(__file__)) #promjeniti ovo na nivo iznad
    print(project_root)
    dirs = make_ctx_dirs(project_root, CTX_LEN)
    cov_dir = dirs["covariate_dir"]

    PAST_ONLY_COVS, FUTURE_KNOWN_COVS = extension1_covariate_sets(
        include_ema=args.include_ema,
        include_chg=args.include_chg,
        include_rolling=args.include_rolling,
    )

    pipeline = load_model("amazon/chronos-2")

    store_ids = read_ids_txt(args.store_ids_path)

    for sid in store_ids:
        cov_pred_path = os.path.join(cov_dir, f"forecast_store_{sid}.csv")
        if SKIP_EXISTING_FORECASTS and os.path.exists(cov_pred_path):
            continue

        df = read_processed_store(sid, processed_dir=PROCESSED_DIR_EXT1)
        df = ensure_dayofweek(df)

        df_past, df_test = temporal_split(df, test_size=HORIZON)
        if len(df_past) > CTX_LEN:
            df_past = df_past.iloc[-CTX_LEN:].reset_index(drop=True)

        df_past = ensure_dayofweek(df_past)
        df_test = ensure_dayofweek(df_test)

        needed_ctx = ["id", "timestamp", "target"] + PAST_ONLY_COVS + FUTURE_KNOWN_COVS
        needed_fut = ["id", "timestamp"] + FUTURE_KNOWN_COVS

        missing_ctx = [c for c in needed_ctx if c not in df_past.columns]
        missing_fut = [c for c in needed_fut if c not in df_test.columns]
        if missing_ctx or missing_fut:
            print(f"[SKIP] Store {sid}: missing cov columns ctx={missing_ctx} fut={missing_fut}")
            continue

        context_cov = df_past[needed_ctx].copy()
        future_cov  = df_test[needed_fut].copy()

        if "Open" in future_cov.columns and "Promo" in future_cov.columns:
            fut_open = pd.to_numeric(future_cov["Open"], errors="coerce").fillna(1)
            future_cov.loc[fut_open.eq(0), "Promo"] = 0
        if "Open" in context_cov.columns and "Promo" in context_cov.columns:
            ctx_open = pd.to_numeric(context_cov["Open"], errors="coerce").fillna(1)
            context_cov.loc[ctx_open.eq(0), "Promo"] = 0

        if context_cov.isna().any().any() or future_cov.isna().any().any():
            print(f"[SKIP] Store {sid}: NaN in covariates")
            continue

        pred_cov = predict_df_covariates(pipeline, context_cov, future_cov, horizon=HORIZON)

        if "timestamp" not in pred_cov.columns and len(pred_cov) == len(df_test):
            pred_cov = pred_cov.copy()
            pred_cov["timestamp"] = df_test["timestamp"].values

        pred_cov = normalize_pred_columns(pred_cov)
        save_quantiles_csv(pred_cov, cov_pred_path, verbose=False)

    # eval full
    run_evaluation(EXPERIMENT, args.outlier_threshold, args.no_outlier_filter, args.show_per_store_lines)

    # eval test-only
    test_ids = read_ids_txt(TEST_IDS_PATH)
    exp_test = materialize_test_only_outputs(EXPERIMENT, test_ids)
    run_evaluation(exp_test, args.outlier_threshold, args.no_outlier_filter, args.show_per_store_lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store_ids_path", type=str, default=os.path.join("reports", "valid_store_ids.txt"))
    ap.add_argument("--include_ema", action="store_true")
    ap.add_argument("--include_chg", action="store_true")
    ap.add_argument("--include_rolling", action="store_true")
    ap.add_argument("--outlier_threshold", type=float, default=0.5)
    ap.add_argument("--no_outlier_filter", action="store_true")
    ap.add_argument("--show_per_store_lines", action="store_true")
    args = ap.parse_args()
    main(args)
