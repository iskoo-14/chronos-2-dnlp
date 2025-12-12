import os
import numpy as np
import pandas as pd

OUT = "outputs"


def load(name):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df if not df.empty else None
    print(f"[WARN] Missing {name}")
    return None


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def rmse(a, b):
    return float(np.sqrt(mse(a, b)))


def interval_width(df):
    return float(np.mean(df["p90"] - df["p10"]))


def percent_diff(a, b):
    return float(100 * np.mean((b - a) / (np.abs(a) + 1e-9)))


def compare(base, test, n1, n2):
    m1 = base["median"].values
    m2 = test["median"].values

    return {
        "MAE": mae(m1, m2),
        "RMSE": rmse(m1, m2),
        "PercentDifference": percent_diff(m1, m2),
        f"Interval_{n1}": interval_width(base),
        f"Interval_{n2}": interval_width(test),
    }


def write_report(lines):
    path = os.path.join(OUT, "comparison_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Saved {path}")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":

    uni = load("univariate.csv")
    cov = load("covariate.csv")

    noise = load("noise_output.csv")
    shuffle = load("shuffle_output.csv")
    missing = load("missing_future_output.csv")
    strong_noise = load("strong_noise_output.csv")
    time_shift = load("time_shift_output.csv")
    trend_break = load("trend_break_output.csv")
    feature_drop = load("feature_drop_output.csv")
    partial_mask = load("partial_mask_output.csv")
    scaling = load("scaling_output.csv")
    long_horizon = load("long_horizon_output.csv")

    rep = []
    rep.append("=== DNLP FORECAST COMPARISON REPORT ===\n")

    # ----------------------------
    # Baseline
    # ----------------------------
    if uni is not None and cov is not None:
        rep.append("> Covariates vs Univariate")
        stats = compare(uni, cov, "Univariate", "Covariates")
        for k, v in stats.items():
            rep.append(f"{k}: {v:.4f}")
        rep.append("")

    # ----------------------------
    # Robustness tests
    # ----------------------------
    def add_section(label, base, test):
        if base is not None and test is not None:
            rep.append(f"> {label} vs Covariates")
            stats = compare(base, test, "Covariates", label)
            for k, v in stats.items():
                rep.append(f"{k}: {v:.4f}")
            rep.append("")

    add_section("Noise", cov, noise)
    add_section("Shuffle", cov, shuffle)
    add_section("MissingFuture", cov, missing)
    add_section("StrongNoise", cov, strong_noise)
    add_section("TimeShift", cov, time_shift)
    add_section("TrendBreak", cov, trend_break)
    add_section("FeatureDrop", cov, feature_drop)
    add_section("PartialMask", cov, partial_mask)
    add_section("Scaling", cov, scaling)
    add_section("LongHorizon", cov, long_horizon)

    write_report(rep)
