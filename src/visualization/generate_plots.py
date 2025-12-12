import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_csv(name):
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"WARNING: File not found {name}")
        return None


def plot_forecast(df, title, filename):
    plt.figure(figsize=(10, 5))
    x = range(len(df))

    plt.plot(x, df["median"], label="Median", linewidth=2)
    plt.fill_between(x, df["p10"], df["p90"], alpha=0.3, label="Confidence Interval")

    plt.title(title)
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def compare_three(df1, df2, df3, label1, label2, label3, title, filename):
    plt.figure(figsize=(10, 5))
    x = range(len(df1))

    plt.plot(x, df1["median"], label=label1, linewidth=2)
    plt.plot(x, df2["median"], label=label2, linewidth=2)
    plt.plot(x, df3["median"], label=label3, linewidth=2)

    plt.title(title)
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Median Prediction")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    print("=== Generating Forecast Plots ===")

    uni = load_csv("univariate.csv")
    cov = load_csv("covariate.csv")
    noise = load_csv("noise_output.csv")
    shuffle = load_csv("shuffle_output.csv")
    missing = load_csv("missing_future_output.csv")

    # Single plots
    if uni is not None:
        plot_forecast(uni, "Univariate Forecast", "univariate_plot.png")

    if cov is not None:
        plot_forecast(cov, "Covariate Forecast", "covariate_plot.png")

    if noise is not None:
        plot_forecast(noise, "Noise Test Forecast", "noise_plot.png")

    if shuffle is not None:
        plot_forecast(shuffle, "Shuffle Test Forecast", "shuffle_plot.png")

    if missing is not None:
        plot_forecast(missing, "Missing Future Forecast", "missing_plot.png")

    # Comparative plot (most important for the paper)
    if uni is not None and cov is not None and shuffle is not None:
        compare_three(
            uni, cov, shuffle,
            "Univariate", "Covariates", "Shuffle",
            "Comparison: True Covariates vs Univariate vs Shuffle",
            "comparison_plot.png"
        )

    print("=== All plots generated successfully ===")
