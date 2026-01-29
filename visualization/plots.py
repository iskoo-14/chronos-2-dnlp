import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from data.make_dataset import temporal_split

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