# =========================
# FILE: src/evaluation/compare_results.py
# (WQL only vs ground truth; robustness = relative)
# =========================
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")


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
        raise ValueError(f"Cannot compare different horizons: {len(m1)} vs {len(m2)}")
    return {
        "MAE": mae(m1, m2),
        "RMSE": rmse(m1, m2),
        "PercentDifference": pct_diff(m1, m2),
        f"Interval_{label_base}": interval_width(base),
        f"Interval_{label_test}": interval_width(test),
    }


if __name__ == "__main__":
    uni = load_csv("univariate.csv")
    cov = load_csv("covariate.csv")
    gt = load_csv("ground_truth.csv")

    noise = load_csv("noise_output.csv")
    strong_noise = load_csv("strong_noise_output.csv")
    shuffle = load_csv("shuffle_output.csv")
    missing = load_csv("missing_future_output.csv")
    time_shift = load_csv("time_shift_output.csv")
    trend_break = load_csv("trend_break_output.csv")
    feature_drop = load_csv("feature_drop_output.csv")
    partial_mask = load_csv("partial_mask_output.csv")
    scaling = load_csv("scaling_output.csv")
    long_horizon = load_csv("long_horizon_output.csv")

    report = []
    report.append("=== DNLP FORECAST COMPARISON REPORT ===\n")

    # Baseline comparison (relative)
    if uni is not None and cov is not None:
        report.append("> Covariates vs Univariate (relative)")
        stats = compare_forecasts(uni, cov, "Univariate", "Covariates")
        for k, v in stats.items():
            report.append(f"{k}: {v:.4f}")
        report.append("")

    # WQL vs ground truth ONLY
    if gt is not None:
        y_true = gt["y_true"].values

        if uni is not None:
            wql_uni = weighted_quantile_loss(
                y_true,
                {0.1: uni["p10"].values, 0.5: uni["median"].values, 0.9: uni["p90"].values},
            )
            report.append("> Univariate vs Ground Truth")
            report.append(f"WQL: {wql_uni:.4f}")
            report.append("")

        if cov is not None:
            wql_cov = weighted_quantile_loss(
                y_true,
                {0.1: cov["p10"].values, 0.5: cov["median"].values, 0.9: cov["p90"].values},
            )
            report.append("> Covariates vs Ground Truth")
            report.append(f"WQL: {wql_cov:.4f}")
            report.append("")

    # Robustness comparisons (valid relative comparisons)
    def add_section(title, test_df, label):
        if cov is None or test_df is None:
            return
        report.append(f"> {title}")
        stats = compare_forecasts(cov, test_df, "Covariates", label)
        for k, v in stats.items():
            report.append(f"{k}: {v:.4f}")
        report.append("")

    add_section("Noise vs Covariates", noise, "Noise")
    add_section("Strong Noise vs Covariates", strong_noise, "StrongNoise")
    add_section("Shuffle vs Covariates", shuffle, "Shuffle")
    add_section("Missing Future vs Covariates", missing, "MissingFuture")
    add_section("Time Shift vs Covariates", time_shift, "TimeShift")
    add_section("Trend Break vs Covariates", trend_break, "TrendBreak")
    add_section("Feature Drop vs Covariates", feature_drop, "FeatureDrop")
    add_section("Partial Mask vs Covariates", partial_mask, "PartialMask")
    add_section("Scaling vs Covariates", scaling, "Scaling")

    # Long horizon descriptive only
    if long_horizon is not None:
        report.append("> Long Horizon Test")
        report.append("This test evaluates model stability on extended forecast horizons (90 steps).")
        report.append("It is not directly comparable with 30-step forecasts and is analyzed separately.")
        report.append(f"Average prediction interval width: {interval_width(long_horizon):.4f}")
        report.append("")

    out_path = os.path.join(OUT, "comparison_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report))

    print(f"[INFO] Saved {out_path}")
