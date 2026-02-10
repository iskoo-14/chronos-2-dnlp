import os
import re
import shutil
import pandas as pd


from evaluation.io import ensure_dir
from evaluation.compare_results import (
    compute_wql_per_store,
    summarize_wql,
    write_comparison_report,
)
from evaluation.select_best_context import select_best_context, print_best_context
from evaluation.metrics import compute_mae_open_closed


# -----------------------------
# CONFIG (edit here)
# -----------------------------
EXPERIMENT = "baseline"  # outputs/<EXPERIMENT>/...
TEST_IDS_PATH = os.path.join("data", "extension3", "splits", "test_store_ids.txt")

OUTLIER_THRESHOLD = 0.5
APPLY_OUTLIER_FILTER = True
SHOW_PER_STORE_LINES = False


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
    Kreira outputs/<experiment>__test_only/forecasts i ground_truth
    samo sa fajlovima koji pripadaju test_ids, zadržava folder strukturu ispod forecasts/.
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

    # ---- copy GT (flat) ----
    copied_gt = 0
    for sid in test_ids:
        src = os.path.join(src_gt, f"ground_truth_store_{sid}.csv")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dst_gt, f"ground_truth_store_{sid}.csv"))
            copied_gt += 1

    # ---- copy forecasts (keep structure) ----
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

    print(f"[INFO] Materialized TEST-only outputs -> outputs/{test_experiment}")
    print(f"[INFO] Copied GT files: {copied_gt} | Copied forecast files: {copied_fc}")
    return test_experiment


if __name__ == "__main__":
    print("===================================================")
    print(f"=== EVALUATION — experiment: {EXPERIMENT} (TEST ONLY) ===")
    print("===================================================")

    # 1) load test ids
    test_ids = read_ids_txt(TEST_IDS_PATH)
    print(f"[INFO] Loaded test store ids: {len(test_ids)} from {TEST_IDS_PATH}")

    # 2) make test-only view
    exp_test = materialize_test_only_outputs(EXPERIMENT, test_ids)

    # 3) run SAME evaluation as before, but on test-only view
    forecasts_root = os.path.join("outputs", exp_test, "forecasts")
    gt_dir = os.path.join("outputs", exp_test, "ground_truth")
    reports_dir = ensure_dir(os.path.join("reports", exp_test))

    print("===================================================")
    print(f"=== EVALUATION — experiment: {exp_test} ===")
    print("===================================================")

    records, text_report = compute_wql_per_store(
        forecasts_root=forecasts_root,
        gt_dir=gt_dir,
        include_store_lines=SHOW_PER_STORE_LINES,
    )

        # WQL compare
    records, text_report = compute_wql_per_store(
        forecasts_root=forecasts_root,
        gt_dir=gt_dir,
        include_store_lines=SHOW_PER_STORE_LINES,
    )

    # -----------------------------
    # INLINE summarization fallback
    # (avoids summarize_wql bug: filter_note unbound)
    # -----------------------------
    reports_dir = reports_dir  # already defined above
    per_store_path = os.path.join(reports_dir, "wql_per_store.csv")
    by_ctx_path = os.path.join(reports_dir, "wql_by_context.csv")
    summary_path = os.path.join(reports_dir, "wql_summary.csv")

    if not records:
        print("[WARN] No records returned from compute_wql_per_store. Nothing to summarize.")
        # still write a minimal report so the run doesn't crash
        out_txt = write_comparison_report(
            reports_dir=reports_dir,
            text_lines=text_report + ["[WARN] Empty records: no forecasts/GT matched for evaluation."],
        )
        print(f"[INFO] Saved {out_txt}")
    else:
        df = pd.DataFrame(records)

        # try to detect column names robustly
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
                f"Cannot summarize records: missing expected cols {missing_cols}. "
                f"Available columns: {df.columns.tolist()}"
            )

        # save per-store
        df.to_csv(per_store_path, index=False)

        # apply outlier filter (same idea as summarize_wql)
        if APPLY_OUTLIER_FILTER:
            thr = float(OUTLIER_THRESHOLD)
            df_f = df[df[wql_col] <= thr].copy()
            filter_note = f"(filtered <= {thr})"
        else:
            df_f = df.copy()
            filter_note = "(no filter)"

        # group by context/mode
        # note: n_stores should count unique stores in that group
        grouped = (
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

        # save by-context and summary (keep same filenames your other scripts expect)
        grouped.to_csv(by_ctx_path, index=False)
        grouped.to_csv(summary_path, index=False)

        # add context summary lines into text report (same format as original)
        text_report.append(f"=== Context summary {filter_note} ===")
        for _, row in grouped.iterrows():
            text_report.append(
                f"CTX {int(row['context_length'])} {row['mode']}: "
                f"mean_wql={row['mean_wql']:.4f} std_wql={0.0 if pd.isna(row['std_wql']) else row['std_wql']:.4f} "
                f"mean_mae={row['mean_mae']:.2f} mean_rmse={row['mean_rmse']:.2f} "
                f"p10_under={row['mean_p10_under']:.2f} p90_over={row['mean_p90_over']:.2f} "
                f"n_stores={int(row['n_stores'])}"
            )

        # select best context (uses summary csv in reports_dir)
        best_summary = select_best_context(reports_dir=reports_dir)
        print_best_context(best_summary)

        # add best cov line to report
        cov_best = [b for b in best_summary if b.get("mode") == "covariate"]
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

        # MAE open/closed (unchanged)
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
        print(f"[INFO] WQL per store saved to {per_store_path}")
        print(f"[INFO] WQL by context saved to {by_ctx_path}")
        print(f"[INFO] WQL summary saved to {summary_path}")
