import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        print(f"[WARN] Missing {name}")
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


# -------------------------------------------------
# Plot helpers
# -------------------------------------------------
def plot_forecast(df, title, fname, color="#1f77b4"):
    x = np.arange(len(df))
    plt.figure(figsize=(10, 4))
    plt.plot(x, df["median"], label="Median", color=color, linewidth=2)
    plt.fill_between(x, df["p10"], df["p90"], color=color, alpha=0.25)
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Sales")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


def plot_comparison(base, test, title, fname):
    x = np.arange(len(base))
    plt.figure(figsize=(10, 4))
    plt.plot(x, base["median"], label="Covariates", linewidth=2)
    plt.plot(x, test["median"], label=title.split(" vs ")[1], linewidth=2)
    plt.legend()
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Sales")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


def plot_difference(base, test, title, fname):
    diff = test["median"].values - base["median"].values
    x = np.arange(len(diff))
    plt.figure(figsize=(10, 3))
    plt.plot(x, diff, linewidth=2)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Difference")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


def plot_uncertainty(df, title, fname):
    width = df["p90"] - df["p10"]
    plt.figure(figsize=(10, 3))
    plt.plot(width, linewidth=2)
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Interval width")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


def plot_pct_difference(base, test, title, fname):
    pct = 100 * (test["median"] - base["median"]) / (np.abs(base["median"]) + 1e-6)
    plt.figure(figsize=(10, 3))
    plt.plot(pct, linewidth=2)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("Forecast horizon")
    plt.ylabel("Δ %")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()
    print(f"Saved plot: {fname}")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":

    cov = load("covariate.csv")

    tests = {
        "Noise": load("noise_output.csv"),
        "Shuffle": load("shuffle_output.csv"),
        "MissingFuture": load("missing_future_output.csv"),
        "StrongNoise": load("strong_noise_output.csv"),
        "TimeShift": load("time_shift_output.csv"),
        "TrendBreak": load("trend_break_output.csv"),
        "FeatureDrop": load("feature_drop_output.csv"),
        "PartialMask": load("partial_mask_output.csv"),
        "Scaling": load("scaling_output.csv"),
        "LongHorizon": load("long_horizon_output.csv"),
    }

    if cov is not None:
        plot_forecast(cov, "Covariate Forecast", "covariate.png")
        plot_uncertainty(cov, "Uncertainty: Covariates", "unc_cov.png")

    for name, df in tests.items():
        if df is None:
            continue

        # Long horizon: standalone analysis only
        if name == "LongHorizon":
            plot_forecast(
                df,
                "Long Horizon Forecast",
                "long_horizon.png"
            )
            plot_uncertainty(
                df,
                "Uncertainty: Long Horizon",
                "unc_long_horizon.png"
            )
            continue

        # Standard robustness comparisons
        if cov is not None:
            plot_comparison(
                cov,
                df,
                f"Covariates vs {name}",
                f"cov_vs_{name.lower()}.png"
            )
            plot_difference(
                cov,
                df,
                f"Difference: {name} vs Covariates",
                f"diff_{name.lower()}.png"
            )
            plot_pct_difference(
                cov,
                df,
                f"Percent Difference: {name} vs Covariates",
                f"pct_{name.lower()}.png"
            )
            plot_uncertainty(
                df,
                f"Uncertainty: {name}",
                f"unc_{name.lower()}.png"
            )

