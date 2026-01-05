import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.data.make_dataset import temporal_split

OUT = "outputs"
FORECAST_OUT = os.path.join(OUT, "forecasts")
ROBUSTNESS_OUT = os.path.join(OUT, "robustness")
FIG = os.path.join(OUT, "figures")
FIG_CTX = FIG  # updated per context
os.makedirs(FIG, exist_ok=True)

# CONFIG: limit per-store plots to a small sample to avoid thousands of images
GENERATE_PER_STORE = True
# If empty, fallback to the first MAX_PLOTS_PER_CTX detected stores
PLOT_SAMPLE_STORES = []  # empty => use fallback + worst stores from report
MAX_PLOTS_PER_CTX = 20  # broader sample to inspect more edge cases
PLOT_ALL_STORES = False  # set True to plot every store found in a context
# Reproducible randomization for good/sample picks; set to None for non-deterministic
SAMPLE_SEED = 42
# Use a metrics report to force-include worst/best stores
ERROR_REPORT_PATH = os.path.join("reports", "mae_open_closed.csv")
BAD_METRICS = ["mae_closed", "wql"]  # ordered; union of worst across metrics
BAD_TOP_N = 25  # total worst stores to include (union)
BAD_MIN = None  # threshold for worst selection (optional)
GOOD_METRIC = "mae_closed"  # metric to pick good samples (lowest)
GOOD_TOP_N = 5
ZERO_TAIL_THRESHOLD = 0.5  # include stores whose last horizon is mostly zeros
CASE_STUDY_PAST_WINDOW = 180  # days of history to show in case-study plots (None = full)

# ------------------------------------------------------------
# IO HELPERS
# ------------------------------------------------------------

def load(base_dir, name, warn=True):
    path = os.path.join(base_dir, name)
    if not os.path.exists(path):
        if warn:
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

def _time_axis(df):
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"])
    return np.arange(len(df))


def plot_forecast(df, title, fname, color="#1f77b4"):
    x = _time_axis(df)
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


def plot_forecast_vs_truth(df_pred, y_true, title, fname, color="#1f77b4", timestamps=None):
    x = pd.to_datetime(timestamps) if timestamps is not None else _time_axis(df_pred)

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


# ------------------------------------------------------------
# COMPARISON PLOTS
# ------------------------------------------------------------

def plot_case_study(
    y_past,
    y_future,
    df_pred,
    title,
    fname,
    color="#1f77b4",
    t_past=None,
    t_future=None,
):
    # Optionally zoom into the most recent slice of history to avoid overlong plots.
    if CASE_STUDY_PAST_WINDOW is not None and len(y_past) > CASE_STUDY_PAST_WINDOW:
        y_past = y_past[-CASE_STUDY_PAST_WINDOW:]
        if t_past is not None:
            t_past = t_past[-CASE_STUDY_PAST_WINDOW:]

    T_past = len(y_past)
    H = len(y_future)

    if t_past is not None and t_future is not None:
        x = np.concatenate([pd.to_datetime(t_past), pd.to_datetime(t_future)])
    else:
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

    # Align the context/future split with the actual time axis to avoid
    # stretching the plot back to epoch (1970) when timestamps are datetimes.
    split_x = x[T_past - 1] if len(x) > (T_past - 1) else x[-1]
    plt.axvline(split_x, linestyle="--", color="gray")
    plt.xlim(x[0], x[-1])

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


# ------------------------------------------------------------
# DISCOVERY HELPERS
# ------------------------------------------------------------

def _list_context_dirs():
    ctx_dirs = []

    def _scan(base_dir):
        found = []
        if not os.path.exists(base_dir):
            return found
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path) and name.startswith("ctx_"):
                try:
                    ctx_len = int(name.replace("ctx_", ""))
                    found.append((ctx_len, path))
                except ValueError:
                    continue
        return found

    ctx_dirs.extend(_scan(FORECAST_OUT))
    if len(ctx_dirs) == 0:
        ctx_dirs.extend(_scan(OUT))

    if len(ctx_dirs) == 0:
        base = FORECAST_OUT if os.path.exists(FORECAST_OUT) else OUT
        ctx_dirs.append((None, base))

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


def _load_error_stores(report_path=ERROR_REPORT_PATH, metrics=BAD_METRICS, top_n=BAD_TOP_N, min_threshold=BAD_MIN):
    if not os.path.exists(report_path):
        print(f"[WARN] Error report not found: {report_path}")
        return [], None
    try:
        df = pd.read_csv(report_path)
        # Try configured metrics, else fallback to a sensible default if present.
        candidate_metrics = [m for m in metrics if m in df.columns]
        if not candidate_metrics:
            for fallback_metric in ["mae_closed", "mae_all", "wql", "mae_open"]:
                if fallback_metric in df.columns:
                    candidate_metrics = [fallback_metric]
                    break

        if not candidate_metrics:
            print(f"[WARN] No metrics {metrics} (or fallbacks) found in {report_path}")
            return [], None

        store_ids = []
        used_metric = None
        for metric in candidate_metrics:
            used_metric = metric if used_metric is None else used_metric
            df_metric = df.dropna(subset=[metric]).copy()
            if min_threshold is not None:
                df_metric = df_metric[df_metric[metric] >= min_threshold]
            df_metric = df_metric.sort_values(metric, ascending=False)
            ids = df_metric["store_id"].astype(str).head(top_n).tolist()
            store_ids.extend(ids)
        # unique while preserving order
        seen = set()
        uniq = []
        for sid in store_ids:
            if sid not in seen:
                seen.add(sid)
                uniq.append(sid)
        if len(uniq) == 0:
            print(f"[WARN] No stores selected from {report_path} using metrics {candidate_metrics}")
        return uniq[:top_n], used_metric
    except Exception:
        return [], None


def _load_best_stores(report_path=ERROR_REPORT_PATH, metric=GOOD_METRIC, top_n=GOOD_TOP_N):
    if not os.path.exists(report_path):
        return []
    try:
        df = pd.read_csv(report_path)
        if metric not in df.columns:
            return []
        df_metric = df.dropna(subset=[metric]).copy()
        df_metric = df_metric.sort_values(metric, ascending=True)
        ids = df_metric["store_id"].astype(str).head(top_n).tolist()
        return ids
    except Exception:
        return []


def _load_full_series(store_id):
    # prefer per-store processed data; fallback to single-store artifact
    if store_id is not None:
        path = os.path.join("src", "data", f"processed_rossmann_store_{store_id}.csv")
    else:
        path = os.path.join("src", "data", "processed_rossmann_single.csv")
    if not os.path.exists(path):
        return None
    df_full = pd.read_csv(path)
    if "target" in df_full.columns:
        return df_full
    return None


def _load_case_study_series(store_id, ctx_len, horizon):
    df_full = _load_full_series(store_id)
    if df_full is None or "target" not in df_full.columns:
        return None, None, None, None

    try:
        df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
        df_full = df_full.sort_values("timestamp").reset_index(drop=True)
        df_past, df_test = temporal_split(df_full, test_size=horizon)
    except Exception:
        return None, None, None, None

    if ctx_len is not None and len(df_past) > ctx_len:
        df_past = df_past.iloc[-ctx_len:].reset_index(drop=True)

    return (
        df_past["target"].values,
        df_past["timestamp"].values,
        df_test["target"].values,
        df_test["timestamp"].values,
    )


def _find_closed_tail_stores(ctx_dir, store_ids, threshold=ZERO_TAIL_THRESHOLD):
    closed = []
    for store_id in store_ids:
        suffix = "" if store_id is None else f"_store_{store_id}"
        gt = load(ctx_dir, f"ground_truth{suffix}.csv", warn=False)
        if gt is None or "y_true" not in gt.columns:
            continue
        share_zero = (gt["y_true"] == 0).mean()
        if share_zero >= threshold:
            closed.append(store_id)
    return closed


def _robustness_dir_for_ctx(ctx_len):
    if ctx_len is None:
        return ROBUSTNESS_OUT if os.path.exists(ROBUSTNESS_OUT) else OUT
    candidate = os.path.join(ROBUSTNESS_OUT, f"ctx_{ctx_len}")
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(ROBUSTNESS_OUT):
        return ROBUSTNESS_OUT
    return OUT


def _load_robustness_outputs(suffix, ctx_len):
    base_dir = _robustness_dir_for_ctx(ctx_len)
    return {
        "noise": load(base_dir, f"noise_output{suffix}.csv", warn=False),
        "strong_noise": load(base_dir, f"strong_noise_output{suffix}.csv", warn=False),
        "shuffle": load(base_dir, f"shuffle_output{suffix}.csv", warn=False),
        "missing": load(base_dir, f"missing_future_output{suffix}.csv", warn=False),
        "time_shift": load(base_dir, f"time_shift_output{suffix}.csv", warn=False),
        "trend_break": load(base_dir, f"trend_break_output{suffix}.csv", warn=False),
        "feature_drop": load(base_dir, f"feature_drop_output{suffix}.csv", warn=False),
        "partial_mask": load(base_dir, f"partial_mask_output{suffix}.csv", warn=False),
        "scaling": load(base_dir, f"scaling_output{suffix}.csv", warn=False),
        "long_horizon": load(base_dir, f"long_horizon_output{suffix}.csv", warn=False),
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    rng = np.random.default_rng(SAMPLE_SEED) if SAMPLE_SEED is not None else np.random.default_rng()

    selection_summary = []  # track which stores are plotted per context

    context_dirs = _list_context_dirs()
    for ctx_len, ctx_dir in context_dirs:
        ctx_label = "baseline" if ctx_len is None else f"ctx_{ctx_len}"
        fig_dir = os.path.join(FIG, ctx_label)
        set_fig_dir(fig_dir)

        store_ids = _detect_stores(ctx_dir)
        if not store_ids:
            continue

        closed_tail_ids = _find_closed_tail_stores(ctx_dir, store_ids)

        # Build subset: start from worst stores, then best, then optional samples, then fallback.
        error_ids, used_metric = _load_error_stores()
        best_ids = _load_best_stores()
        if SAMPLE_SEED is not None and len(best_ids) > 0:
            rng.shuffle(best_ids)

        def _add_if_present(target_list, candidates):
            for s in candidates:
                s_str = None if s is None else str(s)
                # avoid duplicates even if types differ (int vs str)
                if s_str in [None if x is None else str(x) for x in target_list]:
                    continue
                if s is None and None in store_ids:
                    target_list.append(s)
                elif s_str in [str(x) for x in store_ids]:
                    target_list.append(s_str)

        store_ids_subset = []
        _add_if_present(store_ids_subset, error_ids)  # always include worst cases
        _add_if_present(store_ids_subset, best_ids)   # include a few best as baseline

        if PLOT_SAMPLE_STORES:
            _add_if_present(store_ids_subset, [str(s) for s in PLOT_SAMPLE_STORES])

        if PLOT_ALL_STORES:
            _add_if_present(store_ids_subset, store_ids)
        else:
            target_len = max(BAD_TOP_N + GOOD_TOP_N, MAX_PLOTS_PER_CTX)
            if len(store_ids_subset) < target_len:
                # fill up to target_len with the first stores not already selected
                remaining_needed = target_len - len(store_ids_subset)
                remaining = [
                    s for s in store_ids
                    if (None if s is None else str(s)) not in [None if x is None else str(x) for x in store_ids_subset]
                ]
                if SAMPLE_SEED is not None and len(remaining) > 0:
                    rng.shuffle(remaining)
                _add_if_present(store_ids_subset, remaining[:remaining_needed])

        # ensure closed-tail stores are included for inspection
        _add_if_present(store_ids_subset, closed_tail_ids)

        bad_set = set(error_ids)
        good_set = set(best_ids)

        # Track selections (separate rows for bad/good plus the actual plotted subset)
        selection_summary.append(
            {
                "context": ctx_label,
                "kind": "bad",
                "store_count": len(error_ids),
                "stores": ",".join(error_ids),
            }
        )
        selection_summary.append(
            {
                "context": ctx_label,
                "kind": "good",
                "store_count": len(best_ids),
                "stores": ",".join(best_ids),
            }
        )
        selection_summary.append(
            {
                "context": ctx_label,
                "kind": "plotted",
                "store_count": len(store_ids_subset),
                "stores": ",".join(map(str, store_ids_subset)),
            }
        )

        if not GENERATE_PER_STORE:
            continue

        for store_id in tqdm(store_ids_subset, desc=f"{ctx_label} plots (sampled)", unit="store"):

            store_key = None if store_id is None else str(store_id)
            if store_key in bad_set:
                cat = "bad"
            elif store_key in good_set:
                cat = "good"
            else:
                cat = "other"
            set_fig_dir(os.path.join(fig_dir, cat))

            suffix = "" if store_id is None else f"_store_{store_id}"
            tag = "" if store_id is None else f"_store_{store_id}"

            uni = load(ctx_dir, f"univariate{suffix}.csv")
            cov = load(ctx_dir, f"covariate{suffix}.csv")
            gt = load(ctx_dir, f"ground_truth{suffix}.csv")

            if uni is None or cov is None or gt is None:
                continue

            y_true = gt["y_true"].values
            ts_true = pd.to_datetime(gt["timestamp"]) if "timestamp" in gt.columns else None

            H = len(gt)
            y_past, t_past, y_future, t_future = _load_case_study_series(store_id, ctx_len, H)

            robust = _load_robustness_outputs(suffix, ctx_len)

            # SINGLE FORECASTS
            plot_forecast(uni, "Univariate Forecast", f"{ctx_label}_univariate{tag}.png", "#7f7f7f")
            plot_forecast(cov, "Covariate Forecast", f"{ctx_label}_covariate{tag}.png", "#1f77b4")

            if robust["long_horizon"] is not None:
                plot_forecast(
                    robust["long_horizon"],
                    "Long Horizon Forecast (90 steps)",
                    f"{ctx_label}_long_horizon{tag}.png"
                )

            # FORECAST VS TRUTH
            plot_forecast_vs_truth(
                uni,
                y_true,
                "Univariate Forecast vs Ground Truth",
                f"{ctx_label}_univariate_vs_truth{tag}.png",
                "#7f7f7f",
                timestamps=ts_true
            )

            plot_forecast_vs_truth(
                cov,
                y_true,
                "Covariate Forecast vs Ground Truth",
                f"{ctx_label}_covariate_vs_truth{tag}.png",
                "#1f77b4",
                timestamps=ts_true
            )

            if y_past is not None and y_future is not None:
                plot_case_study(
                    y_past,
                    y_future,
                    cov,
                    "Case Study: Covariates vs Ground Truth",
                    f"{ctx_label}_case_study_covariates{tag}.png",
                    t_past=t_past,
                    t_future=t_future
                )

                plot_case_study(
                    y_past,
                    y_future,
                    uni,
                    "Case Study: Univariate vs Ground Truth",
                    f"{ctx_label}_case_study_univariate{tag}.png",
                    t_past=t_past,
                    t_future=t_future
                )

            # ROBUSTNESS COMPARISONS (root outputs)
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

            # UNCERTAINTY
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

    # Save a summary of which stores were plotted (worst/best/sample) per context
    if selection_summary:
        os.makedirs("reports", exist_ok=True)
        sel_df = pd.DataFrame(selection_summary)
        sel_path = os.path.join("reports", "plot_selection.csv")
        sel_df.to_csv(sel_path, index=False)
        print(f"[INFO] Plot selection summary saved to {sel_path}")
