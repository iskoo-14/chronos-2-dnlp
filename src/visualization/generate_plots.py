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


def plot_comparison(base, test, l1, l2, title, fname):
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

def plot_pct_difference(base, test, title, fname):
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


if __name__ == "__main__":

    uni = load("univariate.csv")
    cov = load("covariate.csv")
    noise = load("noise_output.csv")
    shuf = load("shuffle_output.csv")
    miss = load("missing_future_output.csv")

    if uni is not None:
        plot_forecast(uni, "Univariate Forecast", "univariate.png", "#7f7f7f")

    if cov is not None:
        plot_forecast(cov, "Covariate Forecast", "covariate.png", "#1f77b4")

    if cov is not None and noise is not None:
        plot_comparison(cov, noise, "Covariates", "Noise",
                        "Covariates vs Noise", "cov_vs_noise.png")
        plot_difference(cov, noise,
                        "Difference: Noise vs Covariates",
                        "diff_noise.png")

    if cov is not None and shuf is not None:
        plot_comparison(cov, shuf, "Covariates", "Shuffle",
                        "Covariates vs Shuffle", "cov_vs_shuffle.png")
        plot_difference(cov, shuf,
                        "Difference: Shuffle vs Covariates",
                        "diff_shuffle.png")

    if cov is not None and miss is not None:
        plot_comparison(cov, miss, "Covariates", "Missing Future",
                        "Covariates vs Missing Future", "cov_vs_missing.png")
        plot_difference(cov, miss,
                        "Difference: Missing vs Covariates",
                        "diff_missing.png")
    
    if cov is not None and miss is not None:
        plot_uncertainty(cov, "Uncertainty: Covariates", "unc_cov.png")
        plot_uncertainty(miss, "Uncertainty: Missing Future", "unc_missing.png")
        plot_pct_difference(cov, miss,
                            "Percent Difference: Missing vs Covariates",
                            "pct_missing.png")

if cov is not None and shuf is not None:
    plot_pct_difference(cov, shuf,
                        "Percent Difference: Shuffle vs Covariates",
                        "pct_shuffle.png")

