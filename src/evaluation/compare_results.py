import os
import pandas as pd
import numpy as np

OUT = "outputs"


def load(name):
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


def compare(base, test):
    m1 = base["median"].values
    m2 = test["median"].values
    return {
        "MAE": mae(m1, m2),
        "RMSE": rmse(m1, m2),
        "PercentDiff": pct_diff(m1, m2),
        "IntervalBase": interval_width(base),
        "IntervalTest": interval_width(test),
    }


def write_report(lines):
    path = os.path.join(OUT, "comparison_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Saved {path}")


if __name__ == "__main__":

    cov = load("covariate.csv")

    tests = {
        "Noise": "noise_output.csv",
        "StrongNoise": "strong_noise_output.csv",
        "Shuffle": "shuffle_output.csv",
        "MissingFuture": "missing_future_output.csv",
        "FeatureDrop": "feature_drop_output.csv",
        "PartialMask": "partial_mask_output.csv",
        "Scaling": "scaling_output.csv",
        "LongHorizon": "long_horizon_output.csv",
    }

    report = []
    report.append("=== DNLP ROBUSTNESS COMPARISON REPORT ===\n")

    if cov is None:
        report.append("Baseline covariate forecast missing.")
    else:
        for name, file in tests.items():
            df = load(file)
            if df is None:
                continue

            report.append(f"> {name} vs Covariates")
            stats = compare(cov, df)
            for k, v in stats.items():
                report.append(f"{k}: {v:.4f}")
            report.append("")

    write_report(report)
