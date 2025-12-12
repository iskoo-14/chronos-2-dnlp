import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        print(f"[WARN] Missing {name}")
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


# ------------------------------------------------------------
# BASIC PLOTS
# ------------------------------------------------------------

def plot_forecast(df, title, fname, color="#1f77b4"):
    x = np.arange(len(df))
    plt.figure(figsize=(10, 4))
    plt.plot(x, df["median"], label="Median", color=color, linewidth=2)
    plt.fill_between(
        x,
        df["p10"],
        df["p90"],
        color=color,
        alpha=0.25,
        label="Confidence interval"
    )
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


# ------------------------------------------------------------
# COMPARISON PLOTS (ONLY IF SAME HORIZON)
# ------------------------------------------------------------

def _same_length(a, b):
    return len(a) == len(b)


def plot_comparison(base, test, l1, l2, title, fname):
    if not _same_length(base, test):
        print(f"[SKIP] {fname}: different horizons ({len(base)} vs {len(test)})")
        return

    x = np.arange(len(base))
    plt.figure(figsize=(10, 4))
    plt.plot(x, base["median"], label=l1, linewidth=2)
    plt.plot(x, test["median"], label=l2, linewidth=2)
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


def plot_difference(base, test, title, fname):
    if not _same_length(base, test):
        print(f"[SKIP] {fname}: different horizons ({len(base)} vs {len(test)})")
        return

    diff = test["median"].values - base["median"].values
    x = np.arange(len(diff))
    plt.figure(figsize=(10, 3))
    plt.plot(x, diff, color="black", linewidth=1.8)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Difference")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


def plot_pct_difference(base, test, title, fname):
    if not _same_length(base, test):
        print(f"[SKIP] {fname}: different horizons ({len(base)} vs {len(test)})")
        return

    pct = 100 * (test["median"] - base["median"]) / (np.abs(base["median"]) + 1e-6)
    x = np.arange(len(pct))
    plt.figure(figsize=(10, 3))
    plt.plot(x, pct, linewidth=2)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Δ %")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


# ------------------------------------------------------------
# UNCERTAINTY
# ------------------------------------------------------------

def plot_uncertainty(df, title, fname):
    width = df["p90"] - df["p10"]
    x = np.arange(len(width))
    plt.figure(figsize=(10, 3))
    plt.plot(x, width, linewidth=2)
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Interval width")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    uni = load("univariate.csv")
    cov = load("covariate.csv")

    noise = load("noise_output.csv")
    strong_noise = load("strong_noise_output.csv")
    shuffle = load("shuffle_output.csv")
    missing = load("missing_future_output.csv")
    time_shift = load("time_shift_output.csv")
    trend_break = load("trend_break_output.csv")
    feature_drop = load("feature_drop_output.csv")
    partial_mask = load("partial_mask_output.csv")
    scaling = load("scaling_output.csv")
    long_horizon = load("long_horizon_output.csv")

    # --------------------------------------------------------
    # SINGLE FORECASTS
    # --------------------------------------------------------

    if uni is not None:
        plot_forecast(uni, "Univariate Forecast", "univariate.png", "#7f7f7f")

    if cov is not None:
        plot_forecast(cov, "Covariate Forecast", "covariate.png", "#1f77b4")

    if long_horizon is not None:
        plot_forecast(long_horizon, "Long Horizon Forecast (90 steps)", "long_horizon.png")

    # --------------------------------------------------------
    # ROBUSTNESS COMPARISONS
    # --------------------------------------------------------

    tests = [
        (noise, "Noise", "noise"),
        (strong_noise, "Strong Noise", "strong_noise"),
        (shuffle, "Shuffle", "shuffle"),
        (missing, "Missing Future", "missing"),
        (time_shift, "Time Shift", "time_shift"),
        (trend_break, "Trend Break", "trend_break"),
        (feature_drop, "Feature Drop", "feature_drop"),
        (partial_mask, "Partial Mask", "partial_mask"),
        (scaling, "Scaling", "scaling"),
    ]

    for df_test, label, tag in tests:
        if cov is not None and df_test is not None:
            plot_comparison(
                cov, df_test,
                "Covariates", label,
                f"Covariates vs {label}",
                f"cov_vs_{tag}.png"
            )
            plot_difference(
                cov, df_test,
                f"Difference: {label} vs Covariates",
                f"diff_{tag}.png"
            )
            plot_pct_difference(
                cov, df_test,
                f"Percent Difference: {label} vs Covariates",
                f"pct_{tag}.png"
            )

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    if cov is not None:
        plot_uncertainty(cov, "Uncertainty: Covariates", "unc_cov.png")

    if missing is not None:
        plot_uncertainty(missing, "Uncertainty: Missing Future", "unc_missing.png")

    if scaling is not None:
        plot_uncertainty(scaling, "Uncertainty: Scaling", "unc_scaling.png")
