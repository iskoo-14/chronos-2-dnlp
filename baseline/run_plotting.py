import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from visualization import plots as P

EXPERIMENT = "baseline"
CTX_LEN = 512

# ---- input roots ----
OUT = os.path.join("outputs", EXPERIMENT)
FORECAST_ROOT = os.path.join(OUT, "forecasts", f"ctx_{CTX_LEN}")
UNI_DIR = os.path.join(FORECAST_ROOT, "univariate/predictions")
COV_DIR = os.path.join(FORECAST_ROOT, "covariate/predictions")
GT_DIR = os.path.join(OUT, "ground_truth")
ROBUSTNESS_OUT = os.path.join(OUT, "robustness")

# ---- output roots ----
VIS_ROOT = os.path.join("visualization", EXPERIMENT, f"ctx_{CTX_LEN}")
PLOT_SELECTION_OUT = os.path.join(VIS_ROOT, "plot_selection.csv")

# ---- reports input ----
ERROR_REPORT_PATH = os.path.join("reports", EXPERIMENT, "mae_open_closed.csv")

# ---- selection config ----
GENERATE_PER_STORE = True
PLOT_SAMPLE_STORES = []
MAX_PLOTS_PER_CTX = 20
PLOT_ALL_STORES = False
SAMPLE_SEED = 42

BAD_METRICS = ["mae_closed", "wql"]
BAD_TOP_N = 25
BAD_MIN = None

GOOD_METRIC = "mae_closed"
GOOD_TOP_N = 5

ZERO_TAIL_THRESHOLD = 0.5
CASE_STUDY_PAST_WINDOW = 180


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_csv(path: str, warn: bool = True):
    if not os.path.exists(path):
        if warn:
            print(f"[WARN] Missing file: {path}")
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def set_plot_dir(plot_type: str, cat: str, *subtypes: str) -> str:
    out_dir = os.path.join(VIS_ROOT, plot_type, *subtypes, cat)
    ensure_dir(out_dir)
    P.set_fig_dir(out_dir)
    return out_dir


def detect_store_ids_from_univariate(univariate_dir: str):
    if not os.path.exists(univariate_dir):
        return []
    files = [
        f for f in os.listdir(univariate_dir)
        if f.startswith("forecast_store_") and f.endswith(".csv")
    ]
    ids = [f.replace("forecast_store_", "").replace(".csv", "") for f in files]
    ids = sorted(set(ids), key=lambda x: int(x))
    return ids


def pred_path_for(kind: str, store_id: str) -> str:
    base_dir = UNI_DIR if kind == "univariate" else COV_DIR
    return os.path.join(base_dir, f"forecast_store_{store_id}.csv")


def gt_path_for(store_id: str) -> str:
    candidates = [
        os.path.join(GT_DIR, f"ground_truth_store_{store_id}.csv"),
        os.path.join(GT_DIR, f"forecast_store_{store_id}.csv"),
        os.path.join(GT_DIR, f"store_{store_id}.csv"),
        os.path.join(GT_DIR, f"{store_id}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def robustness_dir_for_ctx(ctx_len: int) -> str:
    cand = os.path.join(ROBUSTNESS_OUT, f"ctx_{ctx_len}")
    if os.path.exists(cand):
        return cand
    if os.path.exists(ROBUSTNESS_OUT):
        return ROBUSTNESS_OUT
    return OUT


def load_robustness_outputs(store_id: str, ctx_len: int):
    base_dir = robustness_dir_for_ctx(ctx_len)
    suffix = f"_store_{store_id}"

    def _l(filename: str):
        return load_csv(os.path.join(base_dir, filename), warn=False)

    return {
        "noise": _l(f"noise_output{suffix}.csv"),
        "strong_noise": _l(f"strong_noise_output{suffix}.csv"),
        "shuffle": _l(f"shuffle_output{suffix}.csv"),
        "missing": _l(f"missing_future_output{suffix}.csv"),
        "time_shift": _l(f"time_shift_output{suffix}.csv"),
        "trend_break": _l(f"trend_break_output{suffix}.csv"),
        "feature_drop": _l(f"feature_drop_output{suffix}.csv"),
        "partial_mask": _l(f"partial_mask_output{suffix}.csv"),
        "scaling": _l(f"scaling_output{suffix}.csv"),
        "long_horizon": _l(f"long_horizon_output{suffix}.csv"),
    }


def load_error_stores(
    report_path=ERROR_REPORT_PATH,
    metrics=BAD_METRICS,
    top_n=BAD_TOP_N,
    min_threshold=BAD_MIN,
):
    if not os.path.exists(report_path):
        print(f"[WARN] Error report not found: {report_path}")
        return [], None

    df = pd.read_csv(report_path)

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

    # unique preserve order
    seen, uniq = set(), []
    for sid in store_ids:
        if sid not in seen:
            seen.add(sid)
            uniq.append(sid)

    return uniq[:top_n], used_metric


def load_best_stores(report_path=ERROR_REPORT_PATH, metric=GOOD_METRIC, top_n=GOOD_TOP_N):
    if not os.path.exists(report_path):
        return []
    df = pd.read_csv(report_path)
    if metric not in df.columns:
        return []
    df_metric = df.dropna(subset=[metric]).copy().sort_values(metric, ascending=True)
    return df_metric["store_id"].astype(str).head(top_n).tolist()


def find_closed_tail_stores(store_ids, threshold=ZERO_TAIL_THRESHOLD):
    closed = []
    for store_id in store_ids:
        gt = load_csv(gt_path_for(store_id), warn=False)
        if gt is None or "y_true" not in gt.columns:
            continue
        if (gt["y_true"] == 0).mean() >= threshold:
            closed.append(store_id)
    return closed


if __name__ == "__main__":

    # ensure base output dirs
    ensure_dir(VIS_ROOT)
    ensure_dir(os.path.dirname(PLOT_SELECTION_OUT))

    # keep consistent case-study behavior
    P.CASE_STUDY_PAST_WINDOW = CASE_STUDY_PAST_WINDOW

    print("[INFO] Inputs:")
    print("  UNI_DIR:", UNI_DIR)
    print("  COV_DIR:", COV_DIR)
    print("  GT_DIR:", GT_DIR)
    print("  ROBUSTNESS_OUT:", ROBUSTNESS_OUT)
    print("  ERROR_REPORT_PATH:", ERROR_REPORT_PATH)
    print("[INFO] Outputs:")
    print("  VIS_ROOT:", VIS_ROOT)
    print("  PLOT_SELECTION_OUT:", PLOT_SELECTION_OUT)

    rng = np.random.default_rng(SAMPLE_SEED) if SAMPLE_SEED is not None else np.random.default_rng()
    selection_summary = []

    store_ids = detect_store_ids_from_univariate(UNI_DIR)
    if not store_ids:
        raise RuntimeError(f"No stores found in {UNI_DIR}. Expected forecast_store_<id>.csv")

    closed_tail_ids = find_closed_tail_stores(store_ids)

    error_ids, _ = load_error_stores()
    best_ids = load_best_stores()
    if SAMPLE_SEED is not None and len(best_ids) > 0:
        rng.shuffle(best_ids)

    def _add_if_present(target_list, candidates):
        existing = set(map(str, target_list))
        all_ids = set(map(str, store_ids))
        for s in candidates:
            s = str(s)
            if s in existing:
                continue
            if s in all_ids:
                target_list.append(s)
                existing.add(s)

    store_ids_subset = []
    _add_if_present(store_ids_subset, error_ids)  # worst
    _add_if_present(store_ids_subset, best_ids)   # best

    if PLOT_SAMPLE_STORES:
        _add_if_present(store_ids_subset, [str(s) for s in PLOT_SAMPLE_STORES])

    if PLOT_ALL_STORES:
        _add_if_present(store_ids_subset, store_ids)
    else:
        target_len = max(BAD_TOP_N + GOOD_TOP_N, MAX_PLOTS_PER_CTX)
        if len(store_ids_subset) < target_len:
            remaining = [s for s in store_ids if str(s) not in set(store_ids_subset)]
            if SAMPLE_SEED is not None and len(remaining) > 0:
                rng.shuffle(remaining)
            _add_if_present(store_ids_subset, remaining[: (target_len - len(store_ids_subset))])

    _add_if_present(store_ids_subset, closed_tail_ids)

    bad_set = set(map(str, error_ids))
    good_set = set(map(str, best_ids))

    # summary rows
    selection_summary.append(
        {"context": f"ctx_{CTX_LEN}", "kind": "bad", "store_count": len(error_ids), "stores": ",".join(map(str, error_ids))}
    )
    selection_summary.append(
        {"context": f"ctx_{CTX_LEN}", "kind": "good", "store_count": len(best_ids), "stores": ",".join(map(str, best_ids))}
    )
    selection_summary.append(
        {"context": f"ctx_{CTX_LEN}", "kind": "plotted", "store_count": len(store_ids_subset), "stores": ",".join(map(str, store_ids_subset))}
    )

    if not GENERATE_PER_STORE:
        pd.DataFrame(selection_summary).to_csv(PLOT_SELECTION_OUT, index=False)
        print(f"[INFO] Plot selection summary saved to {PLOT_SELECTION_OUT}")
        raise SystemExit(0)

    for store_id in tqdm(store_ids_subset, desc=f"ctx_{CTX_LEN} plots", unit="store"):
        store_id = str(store_id)

        # category by store quality
        if store_id in bad_set:
            cat = "bad"
        elif store_id in good_set:
            cat = "good"
        else:
            cat = "other"

        tag = f"_store_{store_id}"

        uni = load_csv(pred_path_for("univariate", store_id), warn=False)
        cov = load_csv(pred_path_for("covariate", store_id), warn=False)
        gt = load_csv(gt_path_for(store_id), warn=False)

        if uni is None or cov is None or gt is None:
            continue
        if "median" not in uni.columns or "median" not in cov.columns or "y_true" not in gt.columns:
            continue

        y_true = gt["y_true"].values
        ts_true = pd.to_datetime(gt["timestamp"]) if "timestamp" in gt.columns else None

        H = len(gt)
        # uses your existing plots.py helper (it will simply return Nones if it can't find processed data)
        y_past, t_past, y_future, t_future = P._load_case_study_series(store_id, CTX_LEN, H)

        robust = load_robustness_outputs(store_id, CTX_LEN)

        # 1) forecasts
        set_plot_dir("forecasts", cat)
        P.plot_forecast(uni, "Univariate Forecast", f"ctx_{CTX_LEN}_univariate{tag}.png", "#7f7f7f")
        P.plot_forecast(cov, "Covariate Forecast", f"ctx_{CTX_LEN}_covariate{tag}.png", "#1f77b4")

        # 2) long horizon
        if robust.get("long_horizon") is not None:
            set_plot_dir("long_horizon", cat)
            P.plot_forecast(
                robust["long_horizon"],
                "Long Horizon Forecast (90 steps)",
                f"ctx_{CTX_LEN}_long_horizon{tag}.png",
            )

        # 3) vs truth
        set_plot_dir("vs_truth", cat)
        P.plot_forecast_vs_truth(
            uni,
            y_true,
            "Univariate Forecast vs Ground Truth",
            f"ctx_{CTX_LEN}_univariate_vs_truth{tag}.png",
            "#7f7f7f",
            timestamps=ts_true,
        )
        P.plot_forecast_vs_truth(
            cov,
            y_true,
            "Covariate Forecast vs Ground Truth",
            f"ctx_{CTX_LEN}_covariate_vs_truth{tag}.png",
            "#1f77b4",
            timestamps=ts_true,
        )

        # 4) case study
        if y_past is not None and y_future is not None:
            set_plot_dir("case_study", cat)
            P.plot_case_study(
                y_past,
                y_future,
                cov,
                "Case Study: Covariates vs Ground Truth",
                f"ctx_{CTX_LEN}_case_study_covariates{tag}.png",
                t_past=t_past,
                t_future=t_future,
            )
            P.plot_case_study(
                y_past,
                y_future,
                uni,
                "Case Study: Univariate vs Ground Truth",
                f"ctx_{CTX_LEN}_case_study_univariate{tag}.png",
                t_past=t_past,
                t_future=t_future,
            )

        # 5) robustness comparisons (three buckets)
        tests = [
            (robust.get("noise"), "Noise", "noise"),
            (robust.get("strong_noise"), "Strong Noise", "strong_noise"),
            (robust.get("shuffle"), "Shuffle", "shuffle"),
            (robust.get("missing"), "Missing Future", "missing"),
            (robust.get("time_shift"), "Time Shift", "time_shift"),
            (robust.get("trend_break"), "Trend Break", "trend_break"),
            (robust.get("feature_drop"), "Feature Drop", "feature_drop"),
            (robust.get("partial_mask"), "Partial Mask", "partial_mask"),
            (robust.get("scaling"), "Scaling", "scaling"),
        ]

        for df_test, label, tag2 in tests:
            if df_test is None:
                continue

            set_plot_dir("robustness", cat, "comparison")
            P.plot_comparison(
                cov,
                df_test,
                "Covariates",
                label,
                f"Covariates vs {label}",
                f"ctx_{CTX_LEN}_cov_vs_{tag2}{tag}.png",
            )

            set_plot_dir("robustness", cat, "diff")
            P.plot_difference(
                cov,
                df_test,
                f"Difference: {label} vs Covariates",
                f"ctx_{CTX_LEN}_diff_{tag2}{tag}.png",
            )

            set_plot_dir("robustness", cat, "pct")
            P.plot_pct_difference(
                cov,
                df_test,
                f"Percent Difference: {label} vs Covariates",
                f"ctx_{CTX_LEN}_pct_{tag2}{tag}.png",
            )

        # 6) uncertainty
        set_plot_dir("uncertainty", cat)
        P.plot_uncertainty(cov, "Uncertainty: Covariates", f"ctx_{CTX_LEN}_unc_cov{tag}.png")

        if robust.get("missing") is not None:
            P.plot_uncertainty(
                robust["missing"],
                "Uncertainty: Missing Future",
                f"ctx_{CTX_LEN}_unc_missing{tag}.png",
            )
        if robust.get("scaling") is not None:
            P.plot_uncertainty(
                robust["scaling"],
                "Uncertainty: Scaling",
                f"ctx_{CTX_LEN}_unc_scaling{tag}.png",
            )

    # Save summary
    pd.DataFrame(selection_summary).to_csv(PLOT_SELECTION_OUT, index=False)
    print(f"[INFO] Plot selection summary saved to {PLOT_SELECTION_OUT}")
    print(f"[INFO] Figures saved under {VIS_ROOT}")
