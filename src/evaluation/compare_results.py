import os
import numpy as np
import pandas as pd

OUT = "outputs"


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def quantile_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.maximum(q * diff, (q - 1) * diff)


def weighted_quantile_loss(y_true, preds, quantiles):
    denom = np.sum(np.abs(y_true)) + 1e-9
    total = 0.0

    for q in quantiles:
        diff = y_true - preds[q]
        loss = np.maximum(q * diff, (q - 1) * diff)
        total += np.sum(loss)

    return float(2 * total / denom / len(quantiles))

def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        print(f"[WARN] Missing {name}")
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def rmse(a, b):
    return float(np.sqrt(mse(a, b)))


def interval_width(df):
    return float(np.mean(df["p90"] - df["p10"]))


def pct_diff(a, b):
    return float(100 * np.mean((b - a) / (np.abs(a) + 1e-9)))


# ------------------------------------------------------------
# CORE COMPARISON
# ------------------------------------------------------------

def compare(base, test, label_base, label_test):

    m1 = base["median"].values
    m2 = test["median"].values

    if len(m1) != len(m2):
        raise ValueError(
            f"Cannot compare forecasts with different horizons: "
            f"{len(m1)} vs {len(m2)}"
        )

    return {
        "MAE": mae(m1, m2),
        "RMSE": rmse(m1, m2),
        "PercentDifference": pct_diff(m1, m2),
        f"Interval_{label_base}": interval_width(base),
        f"Interval_{label_test}": interval_width(test),
    }


def add_section(report, title, base, test, label):
    report.append(f"> {title}")
    stats = compare(base, test, "Covariates", label)
    for k, v in stats.items():
        report.append(f"{k}: {v:.4f}")
    report.append("")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    # Baselines
    uni = load("univariate.csv")
    cov = load("covariate.csv")
    gt = load("ground_truth.csv")
    y_true = gt["y_true"].values

    # Robustness tests (same horizon = 30)
    noise = load("noise_output.csv")
    shuffle = load("shuffle_output.csv")
    missing = load("missing_future_output.csv")
    strong_noise = load("strong_noise_output.csv")
    time_shift = load("time_shift_output.csv")
    trend_break = load("trend_break_output.csv")
    feature_drop = load("feature_drop_output.csv")
    partial_mask = load("partial_mask_output.csv")
    scaling = load("scaling_output.csv")

    # Long horizon (different horizon, DO NOT compare numerically)
    long_horizon = load("long_horizon_output.csv")

    report = []
    report.append("=== DNLP FORECAST COMPARISON REPORT ===\n")

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    if uni is not None and cov is not None:
        report.append("> Covariates vs Univariate")
        stats = compare(uni, cov, "Univariate", "Covariates")
        for k, v in stats.items():
            report.append(f"{k}: {v:.4f}")
        report.append("")

    quantiles = [0.1, 0.5, 0.9]

    if uni is not None and gt is not None:
        wql_uni = weighted_quantile_loss(
            y_true,
            {
                0.1: uni["p10"].values,
                0.5: uni["median"].values,
                0.9: uni["p90"].values,
            },
            quantiles
        )
        report.append("> Univariate vs Ground Truth")
        report.append(f"WQL: {wql_uni:.4f}")
        report.append("")

    if cov is not None and gt is not None:
        wql_cov = weighted_quantile_loss(
            y_true,
            {
                0.1: cov["p10"].values,
                0.5: cov["median"].values,
                0.9: cov["p90"].values,
            },
            quantiles
        )
        report.append("> Covariates vs Ground Truth")
        report.append(f"WQL: {wql_cov:.4f}")
        report.append("")

    # --------------------------------------------------------
    # ROBUSTNESS COMPARISONS (VALID)
    # --------------------------------------------------------

    if cov is not None and noise is not None:
        add_section(report, "Noise vs Covariates", cov, noise, "Noise")

    if cov is not None and strong_noise is not None:
        add_section(report, "Strong Noise vs Covariates", cov, strong_noise, "StrongNoise")

    if cov is not None and shuffle is not None:
        add_section(report, "Shuffle vs Covariates", cov, shuffle, "Shuffle")

    if cov is not None and missing is not None:
        add_section(report, "Missing Future vs Covariates", cov, missing, "MissingFuture")

    if cov is not None and time_shift is not None:
        add_section(report, "Time Shift vs Covariates", cov, time_shift, "TimeShift")

    if cov is not None and trend_break is not None:
        add_section(report, "Trend Break vs Covariates", cov, trend_break, "TrendBreak")

    if cov is not None and feature_drop is not None:
        add_section(report, "Feature Drop vs Covariates", cov, feature_drop, "FeatureDrop")

    if cov is not None and partial_mask is not None:
        add_section(report, "Partial Mask vs Covariates", cov, partial_mask, "PartialMask")

    if cov is not None and scaling is not None:
        add_section(report, "Scaling vs Covariates", cov, scaling, "Scaling")

    # --------------------------------------------------------
    # LONG HORIZON (DESCRIPTIVE ONLY)
    # --------------------------------------------------------

    if long_horizon is not None:
        report.append("> Long Horizon Test")
        report.append(
            "This test evaluates model stability on extended forecast horizons (90 steps)."
        )
        report.append(
            "It is not directly comparable with 30-step forecasts and is analyzed separately."
        )
        report.append(
            f"Average prediction interval width: "
            f"{interval_width(long_horizon):.4f}"
        )
        report.append("")

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    out_path = os.path.join(OUT, "comparison_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report))

    print(f"[INFO] Saved {out_path}")
