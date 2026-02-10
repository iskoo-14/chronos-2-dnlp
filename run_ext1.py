import os
import argparse
import pandas as pd
import re
import shutil


from config import (
    HORIZON,
    SAVE_FUTURE_DEBUG,
    SKIP_EXISTING_FORECASTS,
    SKIP_EXISTING_GT_DEBUG,
)

from data.make_dataset import temporal_split
from models.chronos import load_model
from models.univariate import save_quantiles_csv
from models.covariate import predict_df_covariates

from features.feature_engineering import build_extension1_files, extension1_covariate_sets

# evaluation imports
from evaluation.io import ensure_dir as eval_ensure_dir
from evaluation.compare_results import (
    compute_wql_per_store,
    summarize_wql,
    write_comparison_report,
)
from evaluation.select_best_context import select_best_context, print_best_context
from evaluation.metrics import compute_mae_open_closed


EXPERIMENT = "extension1"
CTX_LEN = 512

PROCESSED_DIR_BASELINE = os.path.join("data", "processed")
PROCESSED_DIR_EXT1 = os.path.join("data", "extension1")

# for experimenting
INCLUDE_EMA = True
INCLUDE_CHG = True
INCLUDE_ROLLING = False

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_ctx_dirs(project_root: str, context_len: int) -> dict:
    base = ensure_dir(os.path.join(project_root, "outputs", EXPERIMENT, "forecasts", f"ctx_{context_len}"))
    cov_dir = ensure_dir(os.path.join(base, "covariate"))

    return {"covariate_dir": cov_dir}


def make_global_dirs(project_root: str) -> dict:
    base = ensure_dir(os.path.join(project_root, "outputs", EXPERIMENT))
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
            "week_sin", "week_cos", "month_sin", "month_cos", "quarter",
        ]
        if c in df_test.columns
    ]
    dbg = df_test[dbg_cols].copy()
    if "target" in dbg.columns:
        dbg = dbg.rename(columns={"target": "y_true"})
    return dbg


def normalize_pred_columns(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()

    # map quantile columns coming from your predictor
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

    # keep median for plotting
    if "median" not in pred.columns and "p50" in pred.columns:
        pred["median"] = pred["p50"]

    # keep only what evaluation/plots need
    keep = [c for c in ["timestamp", "p10", "p50", "p90", "median"] if c in pred.columns]
    pred = pred[keep].copy()

    return pred

TEST_IDS_PATH = os.path.join("data", "extension3", "splits", "test_store_ids.txt")


def read_ids_txt(path: str) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing test ids file: {path}")
    ids: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(int(s))
    return sorted(set(ids))


def materialize_test_only_outputs(experiment: str, test_ids: list[int]) -> str:
    """
    Kreira outputs/<experiment>__test_only/{forecasts,ground_truth} sa fajlovima samo za test_ids.
    Ne menja evaluator funkcije, samo pravi test-only view na disku.
    """
    src_root = os.path.join("outputs", experiment)
    src_forecasts = os.path.join(src_root, "forecasts")
    src_gt = os.path.join(src_root, "ground_truth")

    if not os.path.exists(src_forecasts):
        raise FileNotFoundError(f"Missing forecasts_root: {src_forecasts}")
    if not os.path.exists(src_gt):
        raise FileNotFoundError(f"Missing gt_dir: {src_gt}")

    test_experiment = f"{experiment}__test_only"
    dst_root = ensure_dir(os.path.join("outputs", test_experiment))
    dst_forecasts = ensure_dir(os.path.join(dst_root, "forecasts"))
    dst_gt = ensure_dir(os.path.join(dst_root, "ground_truth"))

    test_set = set(test_ids)

    # copy GT (flat)
    copied_gt = 0
    for sid in test_ids:
        src = os.path.join(src_gt, f"ground_truth_store_{sid}.csv")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dst_gt, f"ground_truth_store_{sid}.csv"))
            copied_gt += 1

    # copy forecasts (keep folder structure)
    copied_fc = 0
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

            src_f = os.path.join(root, fn)
            dst_f = os.path.join(dst_dir, fn)
            shutil.copyfile(src_f, dst_f)
            copied_fc += 1

    return test_experiment


# run evaluating - MAYBE I CHANGE THIS SO WE ALWAYS RUN THE run_evaluations.py script for all evaluations
def run_evaluation(
    experiment: str,
    outlier_threshold: float = 0.5,
    no_outlier_filter: bool = False,
    show_per_store_lines: bool = False,
):
    forecasts_root = os.path.join("outputs", experiment, "forecasts")
    gt_dir = os.path.join("outputs", experiment, "ground_truth")
    reports_dir = eval_ensure_dir(os.path.join("reports", experiment))

    print("===================================================")
    print(f"=== EVALUATION — experiment: {experiment} ===")
    print("===================================================")
    print(f"[INFO] forecasts_root: {forecasts_root}")
    print(f"[INFO] gt_dir:        {gt_dir}")
    print(f"[INFO] reports_dir:   {reports_dir}")

    records, text_report = compute_wql_per_store(
        forecasts_root=forecasts_root,
        gt_dir=gt_dir,
        include_store_lines=show_per_store_lines,
    )

    apply_filter = (not no_outlier_filter)

    # -----------------------------
    # Try summarize_wql; if it crashes (filter_note bug), do inline fallback
    # -----------------------------
    try:
        per_store_path, by_ctx_path, summary_path, grouped_df, filter_note = summarize_wql(
            records=records,
            reports_dir=reports_dir,
            apply_outlier_filter=apply_filter,
            outlier_threshold=outlier_threshold,
        )
    except UnboundLocalError as e:
        print(f"[WARN] summarize_wql crashed (known filter_note bug). Using inline fallback. Error: {e}")

        if not records:
            print("[WARN] No records returned from compute_wql_per_store. Cannot summarize.")
            out_txt = write_comparison_report(
                reports_dir=reports_dir,
                text_lines=text_report + ["[WARN] Empty records: no forecasts/GT matched for evaluation."],
            )
            print(f"[INFO] Saved {out_txt}")
            return

        df = pd.DataFrame(records)

        # robust col pick
        def _pick_col(cands):
            for c in cands:
                if c in df.columns:
                    return c
            return None

        store_col = _pick_col(["store_id", "shop_id", "sid", "id"])
        wql_col   = _pick_col(["wql", "pinball", "mean_wql"])
        mae_col   = _pick_col(["mae", "mean_mae"])
        rmse_col  = _pick_col(["rmse", "mean_rmse"])
        ctx_col   = _pick_col(["context_length", "ctx", "context"])
        mode_col  = _pick_col(["mode", "setting"])
        p10u_col  = _pick_col(["p10_under", "mean_p10_under"])
        p90o_col  = _pick_col(["p90_over", "mean_p90_over"])

        missing_cols = [("store", store_col), ("wql", wql_col), ("ctx", ctx_col), ("mode", mode_col)]
        missing_cols = [name for name, col in missing_cols if col is None]
        if missing_cols:
            raise RuntimeError(
                f"Fallback summarize failed: missing expected cols {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        per_store_path = os.path.join(reports_dir, "wql_per_store.csv")
        by_ctx_path = os.path.join(reports_dir, "wql_by_context.csv")
        summary_path = os.path.join(reports_dir, "wql_summary.csv")

        df.to_csv(per_store_path, index=False)

        if apply_filter:
            thr = float(outlier_threshold)
            df_f = df[df[wql_col] <= thr].copy()
            filter_note = f"(filtered <= {thr})"
        else:
            df_f = df.copy()
            filter_note = "(no filter)"

        grouped_df = (
            df_f.groupby([ctx_col, mode_col], dropna=False)
               .agg(
                    mean_wql=(wql_col, "mean"),
                    std_wql=(wql_col, "std"),
                    mean_mae=(mae_col, "mean") if mae_col else (wql_col, "mean"),
                    mean_rmse=(rmse_col, "mean") if rmse_col else (wql_col, "mean"),
                    mean_p10_under=(p10u_col, "mean") if p10u_col else (wql_col, "mean"),
                    mean_p90_over=(p90o_col, "mean") if p90o_col else (wql_col, "mean"),
                    n_stores=(store_col, "nunique"),
               )
               .reset_index()
               .rename(columns={ctx_col: "context_length", mode_col: "mode"})
               .sort_values(["context_length", "mode"])
        )

        grouped_df.to_csv(by_ctx_path, index=False)
        grouped_df.to_csv(summary_path, index=False)

    # -----------------------------
    # Continue exactly like before
    # -----------------------------
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

    cov_best = [b for b in best_summary if b["mode"] == "covariate"]
    if cov_best:
        b = cov_best[0]
        best_line = (
            f"Best context (covariate): {b['best_context']} "
            f"mean_wql={b['mean_wql']:.4f} std={b['std_wql']:.4f} n_stores={b['n_stores']}"
        )
        text_report.append(best_line)
        print(best_line)

    out_txt = write_comparison_report(reports_dir=reports_dir, text_lines=text_report)
    print(f"[INFO] Saved {out_txt}")

    mae_path = compute_mae_open_closed(
        forecasts_root=forecasts_root,
        gt_dir=gt_dir,
        reports_dir=reports_dir,
    )
    if mae_path:
        print(f"[INFO] MAE open/closed report saved to {mae_path}")
    else:
        print("[WARN] MAE open/closed report not produced (missing forecasts or GT).")

    if per_store_path:
        print(f"[INFO] WQL per store saved to {per_store_path}")
    if by_ctx_path:
        print(f"[INFO] WQL by context saved to {by_ctx_path}")
    if summary_path:
        print(f"[INFO] WQL summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--outlier_threshold", type=float, default=0.5)
    parser.add_argument("--no_outlier_filter", action="store_true")
    parser.add_argument("--show_per_store_lines", action="store_true")
    args = parser.parse_args()

    print("===================================================")
    print(f"=== EXTENSION1 RUN (COVARIATE ONLY, CTX={CTX_LEN}) ===")
    print("===================================================")

    project_root = os.path.dirname(os.path.abspath(__file__))

    if not args.skip_training:
        store_ids = read_valid_store_ids()

        build_extension1_files(
            processed_dir=PROCESSED_DIR_BASELINE,
            output_dir=PROCESSED_DIR_EXT1,
            store_ids=store_ids,
            include_ema=INCLUDE_EMA,
            include_chg=INCLUDE_CHG,
            include_rolling=INCLUDE_ROLLING,
        )

        PAST_ONLY_COVS, FUTURE_KNOWN_COVS = extension1_covariate_sets(
            include_ema=INCLUDE_EMA,
            include_chg=INCLUDE_CHG,
            include_rolling=INCLUDE_ROLLING,
        )

        # define gt_dir and dbg_dir
        global_dirs = make_global_dirs(project_root)
        gt_dir = global_dirs["gt_dir"]
        dbg_dir = global_dirs["dbg_dir"]

        dirs = make_ctx_dirs(project_root, CTX_LEN)
        cov_dir = dirs["covariate_dir"]

        pipeline = load_model("amazon/chronos-2")

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

            gt_path = os.path.join(gt_dir, f"ground_truth_store_{sid}.csv")
            if not SKIP_EXISTING_GT_DEBUG or not os.path.exists(gt_path):
                pd.DataFrame({
                    "timestamp": df_test["timestamp"],
                    "y_true": df_test["target"],
                    **({"Open": df_test["Open"]} if "Open" in df_test.columns else {}),
                }).to_csv(gt_path, index=False)

            if SAVE_FUTURE_DEBUG:
                dbg_path = os.path.join(dbg_dir, f"future_debug_store_{sid}.csv")
                if not SKIP_EXISTING_GT_DEBUG or not os.path.exists(dbg_path):
                    build_future_debug(df_test).to_csv(dbg_path, index=False)

            needed_ctx = ["id", "timestamp", "target"] + PAST_ONLY_COVS + FUTURE_KNOWN_COVS
            needed_fut = ["id", "timestamp"] + FUTURE_KNOWN_COVS

            missing_ctx = [c for c in needed_ctx if c not in df_past.columns]
            missing_fut = [c for c in needed_fut if c not in df_test.columns]
            if missing_ctx or missing_fut:
                print(f"[SKIP] Store {sid}: missing cov columns ctx={missing_ctx} fut={missing_fut}")
                continue

            context_cov = df_past[needed_ctx].copy()
            future_cov = df_test[needed_fut].copy()

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

    run_evaluation(
        experiment=EXPERIMENT,
        outlier_threshold=args.outlier_threshold,
        no_outlier_filter=args.no_outlier_filter,
        show_per_store_lines=args.show_per_store_lines,
    )
    
    test_ids = read_ids_txt(TEST_IDS_PATH)
    exp_test = materialize_test_only_outputs(EXPERIMENT, test_ids)

    run_evaluation(
        experiment=exp_test,
        outlier_threshold=args.outlier_threshold,
        no_outlier_filter=args.no_outlier_filter,
        show_per_store_lines=args.show_per_store_lines,
    )