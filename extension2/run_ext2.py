import os
import random

import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.io import ensure_dir
from evaluation.compare_results import compute_wql_per_store, summarize_wql, write_comparison_report
from evaluation.metrics import compute_mae_open_closed
from config import HORIZON


# ---------------- PATHS ----------------
OUT_DIR   = r"outputs/extension2"
CORR_DIR  = os.path.join(OUT_DIR, "forecasts", "ctx_512", "covariate")

GT_DIR    = r"outputs/extension1/ground_truth"

TEST_SHOP_EMB_CSV  = r"data/extension2/shop_embeddings/test_with_clusters.csv"

REPORTS_DIR = ensure_dir(os.path.join("reports", "extension2_test_only"))

TEST_FORECASTS_ROOT = ensure_dir(os.path.join(OUT_DIR, "forecasts_test_only", "ctx_512", "covariate"))
TEST_GT_DIR         = ensure_dir(os.path.join(OUT_DIR, "gt_test_only"))

VIS_DIR = os.path.join("visualization", "extension2")
os.makedirs(VIS_DIR, exist_ok=True)


def plot_shop_forecast(sid: int, forecasts_path: str, gt_path: str, out_path: str, horizon: int):
    df_f = pd.read_csv(forecasts_path)
    df_g = pd.read_csv(gt_path)

    df_f["timestamp"] = pd.to_datetime(df_f["timestamp"])
    df_g["timestamp"] = pd.to_datetime(df_g["timestamp"])

    df = (
        df_f.merge(df_g[["timestamp", "y_true"]], on="timestamp", how="inner")
            .sort_values("timestamp")
            .iloc[:horizon]
    )

    if df.empty:
        print(f"[SKIP] Store {sid}: empty merged df")
        return

    plt.figure(figsize=(12, 4.5))
    plt.title("Covariate Forecast")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Sales")

    plt.plot(df["timestamp"], df["median"], label="Median")
    plt.fill_between(df["timestamp"], df["p10"], df["p90"], alpha=0.25, label="Confidence interval")
    plt.plot(df["timestamp"], df["y_true"], linestyle="--", linewidth=1.5, label="Ground truth")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


def main():
    # 1) load test ids
    df_test_shop = pd.read_csv(TEST_SHOP_EMB_CSV)
    test_ids = sorted(df_test_shop["shop_id"].astype(int).unique().tolist())
    print(f"[INFO] n_test={len(test_ids)}")

    # 2) copy final forecasts into forecasts_test_only structure
    for sid in test_ids:
        src = os.path.join(CORR_DIR, f"forecast_store_{sid}.csv")
        if os.path.exists(src):
            pd.read_csv(src).to_csv(os.path.join(TEST_FORECASTS_ROOT, f"forecast_store_{sid}.csv"), index=False)

    # 3) copy gt into gt_test_only
    for sid in test_ids:
        gp = os.path.join(GT_DIR, f"ground_truth_store_{sid}.csv")
        if os.path.exists(gp):
            pd.read_csv(gp).to_csv(os.path.join(TEST_GT_DIR, f"ground_truth_store_{sid}.csv"), index=False)

    # 4) evaluation (test only)
    records, text_report = compute_wql_per_store(
        forecasts_root=os.path.join(OUT_DIR, "forecasts_test_only"),
        gt_dir=TEST_GT_DIR,
        include_store_lines=False,
    )

    per_store_path, _, summary_path, grouped_df, filter_note = summarize_wql(
        records=records,
        reports_dir=REPORTS_DIR,
        apply_outlier_filter=True,
        outlier_threshold=0.5,
    )

    if grouped_df is not None and not grouped_df.empty:
        text_report.append(f"=== Summary {filter_note} ===")
        row = grouped_df.iloc[0]
        text_report.append(
            f"CTX {row['context_length']} {row['mode']}: "
            f"mean_wql={row['mean_wql']:.4f} std_wql={row['std_wql']:.4f} "
            f"mean_mae={row['mean_mae']:.2f} mean_rmse={row['mean_rmse']:.2f} "
            f"p10_under={row['mean_p10_under']:.2f} p90_over={row['mean_p90_over']:.2f} "
            f"n_stores={int(row['n_stores'])}"
        )

    write_comparison_report(reports_dir=REPORTS_DIR, text_lines=text_report)

    compute_mae_open_closed(
        forecasts_root=os.path.join(OUT_DIR, "forecasts_test_only"),
        gt_dir=TEST_GT_DIR,
        reports_dir=REPORTS_DIR,
    )

    print("[OK] Evaluation done.")

    # 5) plots (5 random test shops that exist)
    available = []
    for sid in test_ids:
        f_path = os.path.join(CORR_DIR, f"forecast_store_{sid}.csv")
        g_path = os.path.join(GT_DIR, f"ground_truth_store_{sid}.csv")
        if os.path.exists(f_path) and os.path.exists(g_path):
            available.append(sid)

    if not available:
        print("[WARN] No available shops found for plotting.")
        return

    random.seed(42)
    chosen = available[:5] if len(available) < 5 else random.sample(available, 5)
    print("[INFO] Plotting shops:", chosen)

    for sid in chosen:
        f_path = os.path.join(CORR_DIR, f"forecast_store_{sid}.csv")
        g_path = os.path.join(GT_DIR, f"ground_truth_store_{sid}.csv")
        out_path = os.path.join(VIS_DIR, f"forecast_store_{sid}.png")
        plot_shop_forecast(sid, f_path, g_path, out_path, HORIZON)


if __name__ == "__main__":
    main()
