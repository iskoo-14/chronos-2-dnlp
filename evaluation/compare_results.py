import os
import numpy as np
import pandas as pd

from .io import ensure_dir, load_csv, list_context_dirs, detect_store_ids
from .metrics import mae, rmse, weighted_quantile_loss

def _extract_quantiles(pred_df: pd.DataFrame) -> dict[float, np.ndarray]:
    cols = set(pred_df.columns)

    # support different formats
    if {"p10", "median", "p90"}.issubset(cols):
        q10, q50, q90 = "p10", "median", "p90"
    elif {"p10", "p50", "p90"}.issubset(cols):
        q10, q50, q90 = "p10", "p50", "p90"
    elif {"0.1", "0.5", "0.9"}.issubset(cols):
        q10, q50, q90 = "0.1", "0.5", "0.9"
    else:
        raise ValueError(f"Missing quantile cols in forecast: columns={list(pred_df.columns)}")

    return {
        0.1: pd.to_numeric(pred_df[q10], errors="coerce").to_numpy(dtype=float),
        0.5: pd.to_numeric(pred_df[q50], errors="coerce").to_numpy(dtype=float),
        0.9: pd.to_numeric(pred_df[q90], errors="coerce").to_numpy(dtype=float),
    }

def _candidate_pred_paths(ctx_dir: str, mode: str, sid: int) -> list[str]:
    fname = f"forecast_store_{sid}.csv"
    return [
        os.path.join(ctx_dir, mode, "predictions", fname),
        os.path.join(ctx_dir, mode, fname),
        os.path.join(ctx_dir, fname),
    ]


def compute_wql_per_store(
    forecasts_root: str,
    gt_dir: str,
    include_store_lines: bool = False,
) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    report_lines = ["=== FORECAST COMPARISON REPORT (MULTICTX) ===", ""]

    context_dirs = list_context_dirs(forecasts_root)
    if not context_dirs:
        report_lines.append(f"[WARN] No ctx_* dirs in {forecasts_root}")
        return records, report_lines

    for ctx_len, ctx_dir in context_dirs:
        store_ids = detect_store_ids(ctx_dir)
        if not store_ids:
            report_lines.append(f"[WARN] No stores detected in {ctx_dir}")
            continue

        for sid in store_ids:
            gt_path = os.path.join(gt_dir, f"ground_truth_store_{sid}.csv")
            gt = load_csv(gt_path)
            if gt is None or "y_true" not in gt.columns:
                continue
            y_true = pd.to_numeric(gt["y_true"], errors="coerce").to_numpy(dtype=float)

            for mode in ["univariate", "covariate"]:
                pred = None

                for pth in _candidate_pred_paths(ctx_dir, mode, sid):
                    pred = load_csv(pth)
                    if pred is not None:
                        break

                if pred is None:
                    continue

                try:
                    q = _extract_quantiles(pred)
                except Exception:
                    continue

                n = min(len(y_true), len(q[0.1]), len(q[0.5]), len(q[0.9]))
                if n <= 0:
                    continue

                yt = y_true[:n]
                p10, p50, p90 = q[0.1][:n], q[0.5][:n], q[0.9][:n]

                wql = weighted_quantile_loss(yt, {0.1: p10, 0.5: p50, 0.9: p90})

                records.append(
                    {
                        "context_length": ctx_len,
                        "store_id": sid,
                        "mode": mode,
                        "wql": float(wql),
                        "mae": float(mae(yt, p50)),
                        "rmse": float(rmse(yt, p50)),
                        "p10_under": float((yt < p10).mean()),
                        "p90_over": float((yt > p90).mean()),
                    }
                )

            if include_store_lines:
                report_lines.append(f"[CTX {ctx_len}] Store {sid}")
                last = [r for r in records if r["context_length"] == ctx_len and r["store_id"] == sid][-2:]
                for r in last:
                    report_lines.append(f"{r['mode']:<10} WQL: {r['wql']:.4f}")
                report_lines.append("")

    return records, report_lines


def summarize_wql(
    records: list[dict],
    reports_dir: str,
    apply_outlier_filter: bool = True,
    outlier_threshold: float = 0.5,
) -> tuple[str | None, str | None, str | None, pd.DataFrame | None, str]:
    if not records:
        return None, None, None, None, "(empty)"

    ensure_dir(reports_dir)
    df_all = pd.DataFrame(records)

    if apply_outlier_filter:
        df_used = df_all[df_all["wql"] <= outlier_threshold].copy()
        filter_note = f"(filtered <= {outlier_threshold})"
        df_all.to_csv(os.path.join(reports_dir, "wql_per_store_all.csv"), index=False)
    else:
        df_used = df_all.copy()
        filter_note = "(unfiltered)"

    per_store_path = os.path.join(reports_dir, "wql_per_store.csv")
    df_used.to_csv(per_store_path, index=False)

    grouped = (
        df_used.groupby(["context_length", "mode"])
        .agg(
            mean_wql=("wql", "mean"),
            std_wql=("wql", "std"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_p10_under=("p10_under", "mean"),
            mean_p90_over=("p90_over", "mean"),
            n_stores=("store_id", "nunique"),
        )
        .reset_index()
    )
    by_ctx_path = os.path.join(reports_dir, "wql_by_context.csv")
    grouped.to_csv(by_ctx_path, index=False)

    summary = (
        df_used.groupby("mode")["wql"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
        )
        .reset_index()
    )
    summary_path = os.path.join(reports_dir, "wql_summary.csv")
    summary.to_csv(summary_path, index=False)

    return per_store_path, by_ctx_path, summary_path, grouped, filter_note


def write_comparison_report(
    reports_dir: str,
    text_lines: list[str],
) -> str:
    ensure_dir(reports_dir)
    out_path = os.path.join(reports_dir, "comparison_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    return out_path
