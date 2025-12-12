import os
import pandas as pd
import numpy as np


OUT_DIR = "outputs"


def load_csv(name):
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"[WARNING] Missing file: {name}")
        return None


def mae(a, b):
    return np.mean(np.abs(a - b))


def mse(a, b):
    return np.mean((a - b)**2)


def rmse(a, b):
    return np.sqrt(mse(a, b))


def interval_width(df):
    """Average p90 - p10 interval width."""
    return float(np.mean(df["p90"] - df["p10"]))


def pct_diff(a, b):
    """Percent difference between medians."""
    return float(100 * np.mean((b - a) / (np.abs(a) + 1e-9)))


def compare_two(df_base, df_test, name_base, name_test):
    """Compare medians of two forecast sets."""
    m1 = df_base["median"].values
    m2 = df_test["median"].values

    return {
        "MAE": mae(m1, m2),
        "RMSE": rmse(m1, m2),
        "PercentDifference": pct_diff(m1, m2),
        "IntervalWidth_" + name_base: interval_width(df_base),
        "IntervalWidth_" + name_test: interval_width(df_test)
    }


def write_report(lines):
    """Save results to outputs/comparison_report.txt."""
    path = os.path.join(OUT_DIR, "comparison_report.txt")
    with open(path, "w") as f:
        for l in lines:
            f.write(l + "\n")
    print(f"\n[INFO] Report saved to {path}\n")


if __name__ == "__main__":

    print("=== DNLP Forecast Comparison Tool ===")

    uni = load_csv("univariate.csv")
    cov = load_csv("covariate.csv")
    noise = load_csv("noise_output.csv")
    shuf = load_csv("shuffle_output.csv")
    missing = load_csv("missing_future_output.csv")

    report = []
    report.append("=== DNLP FORECAST COMPARISON REPORT ===\n")

    # -------------------------------------------------------
    # 1. COVARIATES vs UNIVARIATE
    # -------------------------------------------------------
    if uni is not None and cov is not None:
        report.append(">>> Comparison: Covariates vs Univariate")
        stats = compare_two(uni, cov, "Univariate", "Covariate")
        for k, v in stats.items():
            report.append(f"{k}: {v:.4f}")
        report.append("")

    # -------------------------------------------------------
    # 2. NOISE TEST vs COVARIATE
    # -------------------------------------------------------
    if cov is not None and noise is not None:
        report.append(">>> Comparison: Noise Test vs Covariate")
        stats = compare_two(cov, noise, "Covariate", "Noise")
        for k, v in stats.items():
            report.append(f"{k}: {v:.4f}")

        # Check robustness condition
        if stats["MAE"] < 0.05 * np.mean(cov["median"]):
            report.append("Conclusion: Model robust to noise (GOOD)\n")
        else:
            report.append("Conclusion: Noise had significant impact (BAD)\n")

    # -------------------------------------------------------
    # 3. SHUFFLE TEST vs COVARIATE
    # -------------------------------------------------------
    if cov is not None and shuf is not None:
        report.append(">>> Comparison: Shuffle Test vs Covariate")
        stats = compare_two(cov, shuf, "Covariate", "Shuffle")
        for k, v in stats.items():
            report.append(f"{k}: {v:.4f}")

        # If shuffle has big MAE increase → promo was important
        if stats["MAE"] > 0.10 * np.mean(cov["median"]):
            report.append("Conclusion: Model uses Promo signal (GOOD)\n")
        else:
            report.append("Conclusion: Model barely used Promo (WEAK MODEL)\n")

    # -------------------------------------------------------
    # 4. MISSING FUTURE TEST vs COVARIATE
    # -------------------------------------------------------
    if cov is not None and missing is not None:
        report.append(">>> Comparison: Missing Future vs Covariate")
        stats = compare_two(cov, missing, "Covariate", "MissingFuture")
        for k, v in stats.items():
            report.append(f"{k}: {v:.4f}")

        if stats["MAE"] > 0.05 * np.mean(cov["median"]):
            report.append("Conclusion: Model relies on future covariates (EXPECTED)\n")
        else:
            report.append("Conclusion: Model not using known future info (UNEXPECTED)\n")

    # Save report
    write_report(report)

    print("=== DONE ===")
