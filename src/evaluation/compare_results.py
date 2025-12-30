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


# ------------------------------------------------------------
# UTILITIES 
# ------------------------------------------------------------

def load_csv(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        print(f"[WARN] Missing {name}")
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


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    report = []
    report.append("=== DNLP FORECAST COMPARISON REPORT (MULTISTORE) ===\n")

    # --------------------------------------------------------
    # DETECT STORES 
    # --------------------------------------------------------

    store_files = [
        f
        for f in os.listdir(OUT)
        if f.startswith("univariate_store_") and f.endswith(".csv")
    ]

    # SINGLE-STORE FALLBACK 
    if len(store_files) == 0:
        store_ids = [None]
    else:
        store_ids = [
            f.replace("univariate_store_", "").replace(".csv", "")
            for f in store_files
        ]

    # --------------------------------------------------------
    # ACCUMULATORS
    # --------------------------------------------------------

    wql_uni_all = []
    wql_cov_all = []

    # Robustness accumulators (relative)
    robustness_stats = {
        "Noise": [],
        "StrongNoise": [],
        "Shuffle": [],
        "MissingFuture": [],
        "TimeShift": [],
        "TrendBreak": [],
        "FeatureDrop": [],
        "PartialMask": [],
        "Scaling": [],
    }

    for store_id in store_ids:

        suffix = "" if store_id is None else f"_store_{store_id}"

        uni = load_csv(f"univariate{suffix}.csv")
        cov = load_csv(f"covariate{suffix}.csv")
        gt = load_csv(f"ground_truth{suffix}.csv")

        if uni is None or cov is None or gt is None:
            continue

        y_true = gt["y_true"].values

        # --------------------
        # BASELINE COMPARISON 
        # --------------------
        report.append(f"> Store {store_id if store_id is not None else 'single'}")

        report.append(">> Covariates vs Univariate (relative)")
        stats = compare_forecasts(uni, cov, "Univariate", "Covariates")
        for k, v in stats.items():
            report.append(f"{k}: {v:.4f}")
        report.append("")

        # --------------------
        # WQL vs GROUND TRUTH 

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

        report.append(">> Univariate vs Ground Truth")
        report.append(f"WQL: {wql_uni:.4f}")
        report.append("")

        report.append(">> Covariates vs Ground Truth")
        report.append(f"WQL: {wql_cov:.4f}")
        report.append("")

        wql_uni_all.append(wql_uni)
        wql_cov_all.append(wql_cov)

        # --------------------
        # ROBUSTNESS 
        # --------------------

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

        for label, fname in robustness_map.items():
            test_df = load_csv(f"{fname}{suffix}.csv")
            if test_df is None:
                continue

            stats = compare_forecasts(cov, test_df, "Covariates", label)
            robustness_stats[label].append(stats["MAE"])

    # --------------------------------------------------------
    # FINAL AVERAGE SUMMARY 
    # --------------------------------------------------------

    report.append("> AVERAGE OVER STORES")
    report.append(
        f"Univariate WQL: {np.mean(wql_uni_all):.4f} ± {np.std(wql_uni_all):.4f}"
    )
    report.append(
        f"Covariate  WQL: {np.mean(wql_cov_all):.4f} ± {np.std(wql_cov_all):.4f}"
    )
    report.append("")

    report.append("> AVERAGE ROBUSTNESS (MAE vs Covariates)")
    for label, values in robustness_stats.items():
        if len(values) > 0:
            report.append(f"{label}: {np.mean(values):.4f}")
    report.append("")

    # --------------------------------------------------------
    # LONG HORIZON 
    # --------------------------------------------------------

    long_horizon = load_csv("long_horizon_output.csv")
    if long_horizon is not None:
        report.append("> Long Horizon Test")
        report.append(
            "This test evaluates model stability on extended forecast horizons (90 steps)."
        )
        report.append(
            "It is not directly comparable with 30-step forecasts and is analyzed separately."
        )
        report.append(
            f"Average prediction interval width: {interval_width(long_horizon):.4f}"
        )
        report.append("")

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    out_path = os.path.join(OUT, "comparison_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report))

    print(f"[INFO] Saved {out_path}")
