import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


# ------------------------------------------------------------
# LOAD (IDENTICO)
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


def plot_forecast_vs_truth(df_pred, y_true, title, fname, color="#1f77b4"):
    x = np.arange(len(y_true))

    plt.figure(figsize=(10, 4))
    plt.plot(x, df_pred["median"], label="Forecast (median)", color=color, linewidth=2)
    plt.fill_between(
        x,
        df_pred["p10"],
        df_pred["p90"],
        color=color,
        alpha=0.25,
        label="Forecast interval"
    )
    plt.plot(x, y_true, label="Ground truth", color="black", linestyle="--", linewidth=2)

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
# COMPARISON PLOTS 
# ------------------------------------------------------------

def plot_case_study(
    y_past,
    y_future,
    df_pred,
    title,
    fname,
    color="#1f77b4"
):
    T_past = len(y_past)
    H = len(y_future)

    x = np.arange(T_past + H)
    y_real = np.concatenate([y_past, y_future])

    plt.figure(figsize=(12, 4))
    plt.plot(x, y_real, color="black", label="Real series")

    x_fut = x[T_past:]
    plt.plot(
        x_fut,
        df_pred["median"],
        color=color,
        linewidth=2,
        label="Forecast (median)"
    )

    plt.fill_between(
        x_fut,
        df_pred["p10"],
        df_pred["p90"],
        color=color,
        alpha=0.25,
        label="Forecast interval"
    )

    plt.axvline(T_past - 1, linestyle="--", color="gray")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname))
    plt.close()


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

    # --------------------------------------------------------
    # DETECT STORES 
    # --------------------------------------------------------

    store_files = [
        f for f in os.listdir(OUT)
        if f.startswith("univariate_store_") and f.endswith(".csv")
    ]

    if len(store_files) == 0:
        store_ids = [None]
    else:
        store_ids = [
            f.replace("univariate_store_", "").replace(".csv", "")
            for f in store_files
        ]

    print(f"[INFO] Generating plots for {len(store_ids)} store(s)")

    # --------------------------------------------------------
    # LOOP STORES
    # --------------------------------------------------------

    for store_id in store_ids:

        suffix = "" if store_id is None else f"_store_{store_id}"
        tag = "" if store_id is None else f"_store_{store_id}"

        print(f"[STORE {store_id if store_id is not None else 'single'}] Plotting")

        uni = load(f"univariate{suffix}.csv")
        cov = load(f"covariate{suffix}.csv")
        gt = load(f"ground_truth{suffix}.csv")

        if uni is None or cov is None or gt is None:
            print("[WARN] Missing baseline files, skipping store")
            continue

        df_full = pd.read_csv("src/data/processed_rossmann_single.csv")
        y_full = df_full["target"].values

        H = len(gt)
        y_past = y_full[-(H + 60):-H]
        y_future = y_full[-H:]
        y_true = gt["y_true"].values

        noise = load(f"noise_output{suffix}.csv")
        strong_noise = load(f"strong_noise_output{suffix}.csv")
        shuffle = load(f"shuffle_output{suffix}.csv")
        missing = load(f"missing_future_output{suffix}.csv")
        time_shift = load(f"time_shift_output{suffix}.csv")
        trend_break = load(f"trend_break_output{suffix}.csv")
        feature_drop = load(f"feature_drop_output{suffix}.csv")
        partial_mask = load(f"partial_mask_output{suffix}.csv")
        scaling = load(f"scaling_output{suffix}.csv")
        long_horizon = load(f"long_horizon_output{suffix}.csv")

        # ----------------------------------------------------
        # SINGLE FORECASTS
        # ----------------------------------------------------

        plot_forecast(uni, "Univariate Forecast", f"univariate{tag}.png", "#7f7f7f")
        plot_forecast(cov, "Covariate Forecast", f"covariate{tag}.png", "#1f77b4")

        if long_horizon is not None:
            plot_forecast(
                long_horizon,
                "Long Horizon Forecast (90 steps)",
                f"long_horizon{tag}.png"
            )

        # ----------------------------------------------------
        # FORECAST VS TRUTH
        # ----------------------------------------------------

        plot_forecast_vs_truth(
            uni,
            y_true,
            "Univariate Forecast vs Ground Truth",
            f"univariate_vs_truth{tag}.png",
            "#7f7f7f"
        )

        plot_forecast_vs_truth(
            cov,
            y_true,
            "Covariate Forecast vs Ground Truth",
            f"covariate_vs_truth{tag}.png",
            "#1f77b4"
        )

        plot_case_study(
            y_past,
            y_future,
            cov,
            "Case Study: Covariates vs Ground Truth",
            f"case_study_covariates{tag}.png"
        )

        plot_case_study(
            y_past,
            y_future,
            uni,
            "Case Study: Univariate vs Ground Truth",
            f"case_study_univariate{tag}.png"
        )

        # ----------------------------------------------------
        # ROBUSTNESS COMPARISONS
        # ----------------------------------------------------

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

        for df_test, label, tag2 in tests:
            if df_test is None:
                continue

            plot_comparison(
                cov, df_test,
                "Covariates", label,
                f"Covariates vs {label}",
                f"cov_vs_{tag2}{tag}.png"
            )

            plot_difference(
                cov, df_test,
                f"Difference: {label} vs Covariates",
                f"diff_{tag2}{tag}.png"
            )

            plot_pct_difference(
                cov, df_test,
                f"Percent Difference: {label} vs Covariates",
                f"pct_{tag2}{tag}.png"
            )

        # ----------------------------------------------------
        # UNCERTAINTY
        # ----------------------------------------------------

        plot_uncertainty(cov, "Uncertainty: Covariates", f"unc_cov{tag}.png")

        if missing is not None:
            plot_uncertainty(
                missing,
                "Uncertainty: Missing Future",
                f"unc_missing{tag}.png"
            )

        if scaling is not None:
            plot_uncertainty(
                scaling,
                "Uncertainty: Scaling",
                f"unc_scaling{tag}.png"
            )

    print("[INFO] Plot generation completed.")
