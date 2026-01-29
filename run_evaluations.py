import os
import argparse

from evaluation.io import ensure_dir
from evaluation.compare_results import (
    compute_wql_per_store,
    summarize_wql,
    write_comparison_report,
)
from evaluation.select_best_context import select_best_context, print_best_context
from evaluation.metrics import compute_mae_open_closed

if __name__ == "__main__":
    # something i'm trying for extension, not done yet
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--outlier_threshold", type=float, default=0.5)
    parser.add_argument("--no_outlier_filter", action="store_true")
    parser.add_argument("--show_per_store_lines", action="store_true")
    args = parser.parse_args()

    forecasts_root = os.path.join("outputs", args.experiment, "forecasts")
    gt_dir = os.path.join("outputs", args.experiment, "ground_truth")
    reports_dir = ensure_dir(os.path.join("reports", args.experiment))

    print("===================================================")
    print(f"=== EVALUATION — experiment: {args.experiment} ===")
    print("===================================================")

    # WQL compare
    records, text_report = compute_wql_per_store(
        forecasts_root=forecasts_root,
        gt_dir=gt_dir,
        include_store_lines=args.show_per_store_lines,
    )

    per_store_path, by_ctx_path, summary_path, grouped_df, filter_note = summarize_wql(
        records=records,
        reports_dir=reports_dir,
        apply_outlier_filter=(not args.no_outlier_filter),
        outlier_threshold=args.outlier_threshold,
    )

    # add context summary lines
    if grouped_df is not None and not grouped_df.empty:
        text_report.append(f"=== Context summary {filter_note} ===")
        for _, row in grouped_df.sort_values(["context_length", "mode"]).iterrows():
            text_report.append(
                f"CTX {row['context_length']} {row['mode']}: "
                f"mean_wql={row['mean_wql']:.4f} std_wql={row['std_wql']:.4f} "
                f"mean_mae={row['mean_mae']:.2f} mean_rmse={row['mean_rmse']:.2f} "
                f"p10_under={row['mean_p10_under']:.2f} p90_over={row['mean_p90_over']:.2f} "
                f"n_stores={int(row['n_stores'])}"
            )

    # select best context
    best_summary = select_best_context(reports_dir=reports_dir)
    print_best_context(best_summary)

    # add best cov line to report
    cov_best = [b for b in best_summary if b["mode"] == "covariate"]
    if cov_best:
        b = cov_best[0]
        best_line = (
            f"Best context (covariate): {b['best_context']} "
            f"mean_wql={b['mean_wql']:.4f} std={b['std_wql']:.4f} n_stores={b['n_stores']}"
        )
        text_report.append(best_line)
        print(best_line)
        
    # save text report
    out_txt = write_comparison_report(reports_dir=reports_dir, text_lines=text_report)
    print(f"[INFO] Saved {out_txt}")

    # checks to see if everything is working
    mae_path = compute_mae_open_closed(
        forecasts_root=forecasts_root,
        gt_dir=gt_dir,
        reports_dir=reports_dir,
    )
    if mae_path:
        print(f"[INFO] MAE open/closed report saved to {mae_path}")
    else:
        print("[WARN] MAE open/closed report not produced (missing forecasts or GT).")

    # info prints
    if per_store_path:
        print(f"[INFO] WQL per store saved to {per_store_path}")
    if by_ctx_path:
        print(f"[INFO] WQL by context saved to {by_ctx_path}")
    if summary_path:
        print(f"[INFO] WQL summary saved to {summary_path}")
