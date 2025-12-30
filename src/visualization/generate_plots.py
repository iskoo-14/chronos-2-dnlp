import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
FIG_CTX = FIG  # updated per context
os.makedirs(FIG, exist_ok=True)


# ------------------------------------------------------------
# IO HELPERS
# ------------------------------------------------------------

def load(base_dir, name):
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        print(f"[WARN] Missing {name} in {base_dir}")
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def set_fig_dir(path):
    global FIG_CTX
    FIG_CTX = path
    os.makedirs(FIG_CTX, exist_ok=True)


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
    plt.savefig(os.path.join(FIG_CTX, fname))
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
    plt.savefig(os.path.join(FIG_CTX, fname))
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
    plt.savefig(os.path.join(FIG_CTX, fname))
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
    plt.savefig(os.path.join(FIG_CTX, fname))
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
    plt.savefig(os.path.join(FIG_CTX, fname))
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
    plt.ylabel("%")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_CTX, fname))
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
    plt.savefig(os.path.join(FIG_CTX, fname))
    plt.close()
    print(f"Saved plot: {fname}")


# ------------------------------------------------------------
# DISCOVERY HELPERS
# ------------------------------------------------------------

def _list_context_dirs():
    ctx_dirs = []
    for name in os.listdir(OUT):
        path = os.path.join(OUT, name)
        if os.path.isdir(path) and name.startswith("ctx_"):
            try:
                ctx_len = int(name.replace("ctx_", ""))
                ctx_dirs.append((ctx_len, path))
            except ValueError:
                continue
    if len(ctx_dirs) == 0:
        ctx_dirs.append((None, OUT))
    return sorted(ctx_dirs, key=lambda x: (x[0] is None, x[0]))


def _detect_stores(ctx_dir):
    store_files = [
        f for f in os.listdir(ctx_dir)
        if f.startswith("univariate_store_") and f.endswith(".csv")
    ]
    if len(store_files) == 0:
        if os.path.exists(os.path.join(ctx_dir, "univariate.csv")):
            return [None]
        return []
    return [
        f.replace("univariate_store_", "").replace(".csv", "")
        for f in store_files
    ]


def _load_full_series(store_id):
    # prefer per-store processed data; fallback to single-store artifact
    if store_id is not None:
        path = os.path.join("src", "data", f"processed_rossmann_store_{store_id}.csv")
        if os.path.exists(path):
            df_full = pd.read_csv(path)
            if "target" in df_full.columns:
                return df_full["target"].values
    fallback = os.path.join("src", "data", "processed_rossmann_single.csv")
    if os.path.exists(fallback):
        df_full = pd.read_csv(fallback)
        if "target" in df_full.columns:
            return df_full["target"].values
    return None


def _load_robustness_outputs(suffix):
    return {
        "noise": load(OUT, f"noise_output{suffix}.csv"),
        "strong_noise": load(OUT, f"strong_noise_output{suffix}.csv"),
        "shuffle": load(OUT, f"shuffle_output{suffix}.csv"),
        "missing": load(OUT, f"missing_future_output{suffix}.csv"),
        "time_shift": load(OUT, f"time_shift_output{suffix}.csv"),
        "trend_break": load(OUT, f"trend_break_output{suffix}.csv"),
        "feature_drop": load(OUT, f"feature_drop_output{suffix}.csv"),
        "partial_mask": load(OUT, f"partial_mask_output{suffix}.csv"),
        "scaling": load(OUT, f"scaling_output{suffix}.csv"),
        "long_horizon": load(OUT, f"long_horizon_output{suffix}.csv"),
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    context_dirs = _list_context_dirs()
    print(f"[INFO] Found {len(context_dirs)} context folder(s)")

    for ctx_len, ctx_dir in context_dirs:
        ctx_label = "baseline" if ctx_len is None else f"ctx_{ctx_len}"
        fig_dir = os.path.join(FIG, ctx_label)
        set_fig_dir(fig_dir)

        store_ids = _detect_stores(ctx_dir)
        print(f"[INFO] [{ctx_label}] Generating plots for {len(store_ids)} store(s)")

        for store_id in store_ids:

            suffix = "" if store_id is None else f"_store_{store_id}"
            tag = "" if store_id is None else f"_store_{store_id}"

            print(f"[STORE {store_id if store_id is not None else 'single'} | {ctx_label}] Plotting")

            uni = load(ctx_dir, f"univariate{suffix}.csv")
            cov = load(ctx_dir, f"covariate{suffix}.csv")
            gt = load(ctx_dir, f"ground_truth{suffix}.csv")

            if uni is None or cov is None or gt is None:
                print("[WARN] Missing baseline files, skipping store")
                continue

            series_full = _load_full_series(store_id)
            y_true = gt["y_true"].values

            H = len(gt)
            if series_full is not None and len(series_full) >= H + 60:
                y_past = series_full[-(H + 60):-H]
                y_future = series_full[-H:]
            else:
                y_past = None
                y_future = None

            robust = _load_robustness_outputs(suffix)

            # ----------------------------------------------------
            # SINGLE FORECASTS
            # ----------------------------------------------------

            plot_forecast(uni, "Univariate Forecast", f"{ctx_label}_univariate{tag}.png", "#7f7f7f")
            plot_forecast(cov, "Covariate Forecast", f"{ctx_label}_covariate{tag}.png", "#1f77b4")

            if robust["long_horizon"] is not None:
                plot_forecast(
                    robust["long_horizon"],
                    "Long Horizon Forecast (90 steps)",
                    f"{ctx_label}_long_horizon{tag}.png"
                )

            # ----------------------------------------------------
            # FORECAST VS TRUTH
            # ----------------------------------------------------

            plot_forecast_vs_truth(
                uni,
                y_true,
                "Univariate Forecast vs Ground Truth",
                f"{ctx_label}_univariate_vs_truth{tag}.png",
                "#7f7f7f"
            )

            plot_forecast_vs_truth(
                cov,
                y_true,
                "Covariate Forecast vs Ground Truth",
                f"{ctx_label}_covariate_vs_truth{tag}.png",
                "#1f77b4"
            )

            if y_past is not None and y_future is not None:
                plot_case_study(
                    y_past,
                    y_future,
                    cov,
                    "Case Study: Covariates vs Ground Truth",
                    f"{ctx_label}_case_study_covariates{tag}.png"
                )

                plot_case_study(
                    y_past,
                    y_future,
                    uni,
                    "Case Study: Univariate vs Ground Truth",
                    f"{ctx_label}_case_study_univariate{tag}.png"
                )

            # ----------------------------------------------------
            # ROBUSTNESS COMPARISONS (root outputs)
            # ----------------------------------------------------

            tests = [
                (robust["noise"], "Noise", "noise"),
                (robust["strong_noise"], "Strong Noise", "strong_noise"),
                (robust["shuffle"], "Shuffle", "shuffle"),
                (robust["missing"], "Missing Future", "missing"),
                (robust["time_shift"], "Time Shift", "time_shift"),
                (robust["trend_break"], "Trend Break", "trend_break"),
                (robust["feature_drop"], "Feature Drop", "feature_drop"),
                (robust["partial_mask"], "Partial Mask", "partial_mask"),
                (robust["scaling"], "Scaling", "scaling"),
            ]

            for df_test, label, tag2 in tests:
                if df_test is None:
                    continue

                plot_comparison(
                    cov, df_test,
                    "Covariates", label,
                    f"Covariates vs {label}",
                    f"{ctx_label}_cov_vs_{tag2}{tag}.png"
                )

                plot_difference(
                    cov, df_test,
                    f"Difference: {label} vs Covariates",
                    f"{ctx_label}_diff_{tag2}{tag}.png"
                )

                plot_pct_difference(
                    cov, df_test,
                    f"Percent Difference: {label} vs Covariates",
                    f"{ctx_label}_pct_{tag2}{tag}.png"
                )

            # ----------------------------------------------------
            # UNCERTAINTY
            # ----------------------------------------------------

            plot_uncertainty(cov, "Uncertainty: Covariates", f"{ctx_label}_unc_cov{tag}.png")

            if robust["missing"] is not None:
                plot_uncertainty(
                    robust["missing"],
                    "Uncertainty: Missing Future",
                    f"{ctx_label}_unc_missing{tag}.png"
                )

            if robust["scaling"] is not None:
                plot_uncertainty(
                    robust["scaling"],
                    "Uncertainty: Scaling",
                    f"{ctx_label}_unc_scaling{tag}.png"
                )

    print("[INFO] Plot generation completed.")
