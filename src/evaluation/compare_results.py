# =========================
# (WQL only vs ground truth; robustness = relative)
# =========================
import os
import numpy as np
import pandas as pd

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs",
)
REPORTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports",
)
os.makedirs(REPORTS, exist_ok=True)
OUTLIER_THRESHOLD = 0.5  # filter to drop extreme WQL outliers
APPLY_OUTLIER_FILTER = True  # apply threshold to all reported WQL files
SHOW_PER_STORE_LINES = False  # set True to include per-store lines in comparison_report


# ------------------------------------------------------------
# UTILITIES 
# ------------------------------------------------------------

def load_csv(base_dir, name):
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        print(f"[WARN] Missing {name} in {base_dir}")
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def interval_width(df):
    return float(np.mean(df["p90"] - df["p10"]))


def pct_diff(a, b):
    return float(100 * np.mean((b - a) / (np.abs(a) + 1e-9)))


def weighted_quantile_loss(y_true, preds):
    quantiles = [0.1, 0.5, 0.9]
    denom = np.sum(np.abs(y_true)) + 1e-9
    total = 0.0
    for q in quantiles:
        diff = y_true - preds[q]
        loss = np.maximum(q * diff, (q - 1) * diff)
        total += np.sum(loss)
    return float(2 * total / denom / len(quantiles))


def compare_forecasts(base, test, label_base, label_test):
    m1 = base["median"].values
    m2 = test["median"].values
    if len(m1) != len(m2):
        raise ValueError(
            f"Cannot compare different horizons: {len(m1)} vs {len(m2)}"
        )
    return {
        "MAE": mae(m1, m2),
        "RMSE": rmse(m1, m2),
        "PercentDifference": pct_diff(m1, m2),
        f"Interval_{label_base}": interval_width(base),
        f"Interval_{label_test}": interval_width(test),
    }


def list_context_dirs():
    ctx_dirs = []
    for name in os.listdir(OUT):
        path = os.path.join(OUT, name)
        if os.path.isdir(path) and name.startswith("ctx_"):
            try:
                ctx_len = int(name.replace("ctx_", ""))
                ctx_dirs.append((ctx_len, path))
            except ValueError:
                continue
    if len(ctx_dirs) == 0:
        ctx_dirs.append((None, OUT))
    return sorted(ctx_dirs, key=lambda x: (x[0] is None, x[0]))


def detect_stores(ctx_dir):
    store_files = [
        f
        for f in os.listdir(ctx_dir)
        if f.startswith("univariate_store_") and f.endswith(".csv")
    ]
    if len(store_files) == 0:
        if os.path.exists(os.path.join(ctx_dir, "univariate.csv")):
            return [None]
        return []
    return [
        f.replace("univariate_store_", "").replace(".csv", "")
        for f in store_files
    ]


def compute_wql_per_store(context_dirs, include_store_lines=True):
    records = []
    report_lines = ["=== DNLP FORECAST COMPARISON REPORT (MULTICTX) ===", ""]

    for ctx_len, ctx_dir in context_dirs:
        store_ids = detect_stores(ctx_dir)
        if len(store_ids) == 0:
            print(f"[WARN] No stores detected in {ctx_dir}")
            continue

        for store_id in store_ids:
            suffix = "" if store_id is None else f"_store_{store_id}"

            uni = load_csv(ctx_dir, f"univariate{suffix}.csv")
            cov = load_csv(ctx_dir, f"covariate{suffix}.csv")
            gt = load_csv(ctx_dir, f"ground_truth{suffix}.csv")

            if uni is None or cov is None or gt is None:
                continue

            y_true = gt["y_true"].values

            wql_uni = weighted_quantile_loss(
                y_true,
                {
                    0.1: uni["p10"].values,
                    0.5: uni["median"].values,
                    0.9: uni["p90"].values,
                },
            )
            wql_cov = weighted_quantile_loss(
                y_true,
                {
                    0.1: cov["p10"].values,
                    0.5: cov["median"].values,
                    0.9: cov["p90"].values,
                },
            )

            records.append(
                {
                    "context_length": ctx_len,
                    "store_id": store_id if store_id is not None else "single",
                    "mode": "univariate",
                    "wql": wql_uni,
                }
            )
            records.append(
                {
                    "context_length": ctx_len,
                    "store_id": store_id if store_id is not None else "single",
                    "mode": "covariate",
                    "wql": wql_cov,
                }
            )

            if include_store_lines:
                report_lines.append(
                    f"[CTX {ctx_len}] Store {store_id if store_id is not None else 'single'}"
                )
                report_lines.append(f"Univariate WQL: {wql_uni:.4f}")
                report_lines.append(f"Covariate  WQL: {wql_cov:.4f}")
                report_lines.append("")

    return records, report_lines


def summarize_wql(records, apply_filter=False, threshold=OUTLIER_THRESHOLD):
    if len(records) == 0:
        return None, None, None, None
    df_all = pd.DataFrame(records)

    if apply_filter:
        df_used = df_all[df_all["wql"] <= threshold].copy()
        filter_note = f"(filtered <= {threshold})"
    else:
        df_used = df_all.copy()
        filter_note = "(unfiltered)"

    per_store_path = os.path.join(REPORTS, "wql_per_store.csv")
    df_used.to_csv(per_store_path, index=False)
    if apply_filter:
        # keep full data for reference
        df_all.to_csv(os.path.join(REPORTS, "wql_per_store_all.csv"), index=False)

    grouped = (
        df_used.groupby(["context_length", "mode"])
        .agg(mean_wql=("wql", "mean"), std_wql=("wql", "std"), n_stores=("store_id", "nunique"))
        .reset_index()
    )
    by_ctx_path = os.path.join(REPORTS, "wql_by_context.csv")
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
    summary_path = os.path.join(REPORTS, "wql_summary.csv")
    summary.to_csv(summary_path, index=False)
    return per_store_path, by_ctx_path, summary_path, grouped, df_used, filter_note


def collect_robustness_summary(baseline_ctx_dir, base_context_length):
    robustness_map = {
        "Noise": "noise_output",
        "StrongNoise": "strong_noise_output",
        "Shuffle": "shuffle_output",
        "MissingFuture": "missing_future_output",
        "TimeShift": "time_shift_output",
        "TrendBreak": "trend_break_output",
        "FeatureDrop": "feature_drop_output",
        "PartialMask": "partial_mask_output",
        "Scaling": "scaling_output",
    }

    store_ids = detect_stores(baseline_ctx_dir)
    records = []

    for store_id in store_ids:
        suffix = "" if store_id is None else f"_store_{store_id}"
        cov = load_csv(baseline_ctx_dir, f"covariate{suffix}.csv")
        if cov is None:
            continue

        for test_name, fname in robustness_map.items():
            test_df = load_csv(OUT, f"{fname}{suffix}.csv")
            if test_df is None:
                continue

            stats = compare_forecasts(cov, test_df, "Covariates", test_name)
            record = {
                "context_length": base_context_length,
                "store_id": store_id if store_id is not None else "single",
                "test_name": test_name,
            }
            record.update(stats)
            records.append(record)

    if len(records) == 0:
        return None, None

    df = pd.DataFrame(records)
    per_store_path = os.path.join(REPORTS, "robustness_per_store_merged.csv")
    df.to_csv(per_store_path, index=False)

    numeric_cols = [c for c in df.columns if c not in ["store_id", "test_name", "context_length"]]
    summary = (
        df.groupby("test_name")[numeric_cols]
        .agg(["mean", "std", "median"])
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary_path = os.path.join(REPORTS, "robustness_summary.csv")
    summary.to_csv(summary_path, index=False)
    return per_store_path, summary_path


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    context_dirs = list_context_dirs()
    wql_records, text_report = compute_wql_per_store(context_dirs, include_store_lines=SHOW_PER_STORE_LINES)
    per_store_path, by_ctx_path, summary_path, grouped_df, df_wql, filter_note = summarize_wql(
        wql_records, apply_filter=APPLY_OUTLIER_FILTER, threshold=OUTLIER_THRESHOLD
    )

    baseline_ctx = next((c for c in context_dirs if c[0] == 256), context_dirs[0])
    robustness_paths = collect_robustness_summary(
        baseline_ctx_dir=baseline_ctx[1], base_context_length=baseline_ctx[0]
    )

    best_line = None
    if grouped_df is not None:
        best_cov = (
            grouped_df[grouped_df["mode"] == "covariate"]
            .sort_values("mean_wql", ascending=True)
            .head(1)
        )
        if len(best_cov) > 0:
            row = best_cov.iloc[0]
            best_line = (
                f"Best context (covariate): {row['context_length']} "
                f"mean_wql={row['mean_wql']:.4f} std={row['std_wql']:.4f} n_stores={int(row['n_stores'])}"
            )
            text_report.append(best_line)

        # add per-context summary lines
        text_report.append(f"=== Context summary {filter_note} ===")
        for _, row in grouped_df.sort_values(["context_length", "mode"]).iterrows():
            text_report.append(
                f"CTX {row['context_length']} {row['mode']}: "
                f"mean_wql={row['mean_wql']:.4f} std={row['std_wql']:.4f} n_stores={int(row['n_stores'])}"
            )

    if per_store_path:
        text_report.append(f"[INFO] WQL per store saved to {per_store_path}")
    if by_ctx_path:
        text_report.append(f"[INFO] WQL by context saved to {by_ctx_path}")
    if summary_path:
        text_report.append(f"[INFO] WQL summary saved to {summary_path}")

    # Filtered summaries (drop outliers above OUTLIER_THRESHOLD)
    filt_summary_path, filt_by_ctx_path = summarize_wql_filtered(df_wql, threshold=OUTLIER_THRESHOLD)
    if filt_summary_path:
        text_report.append(f"[INFO] WQL summary (filtered <= {OUTLIER_THRESHOLD}) saved to {filt_summary_path}")
    if filt_by_ctx_path:
        text_report.append(f"[INFO] WQL by context (filtered <= {OUTLIER_THRESHOLD}) saved to {filt_by_ctx_path}")

    if robustness_paths and robustness_paths[0]:
        text_report.append(f"[INFO] Robustness merged saved to {robustness_paths[0]}")
    if robustness_paths and robustness_paths[1]:
        text_report.append(f"[INFO] Robustness summary saved to {robustness_paths[1]}")

    # LONG HORIZON (optional, legacy)
    long_horizon = load_csv(OUT, "long_horizon_output.csv")
    if long_horizon is not None:
        text_report.append("Long horizon (90-step) interval width:")
        text_report.append(f"{interval_width(long_horizon):.4f}")

    out_path = os.path.join(OUT, "comparison_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(text_report))

    print(f"[INFO] Saved {out_path}")
    if best_line:
        print(best_line)
