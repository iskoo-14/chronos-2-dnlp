import os
import numpy as np
import pandas as pd

from models.covariate import predict_df_covariates
from models.univariate import save_quantiles_csv
from .io import ensure_dir


def _log(msg, verbose):
    if verbose:
        print(msg)


def _ensure_outputs_dir(output_root=None):
    # output_root will be explicitly passed by run_robustness
    if output_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_root = os.path.join(project_root, "outputs", "baseline", "robustness")
    os.makedirs(output_root, exist_ok=True)
    return output_root


def _make_context_future(df, horizon=30, context_len=None):
    if "timestamp" not in df.columns:
        raise ValueError("df must contain 'timestamp' column")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) <= horizon:
        raise ValueError("Not enough rows for the requested horizon")

    if context_len is None:
        context_len = max(0, len(df) - horizon)

    start = max(0, len(df) - (context_len + horizon))
    mid = len(df) - horizon

    context_df = df.iloc[start:mid].reset_index(drop=True)
    future_df = df.iloc[mid:].reset_index(drop=True)
    return context_df, future_df


def _base_covariates():
    past_only = ["Customers"]
    future_covs = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]
    return past_only, future_covs


def _prepare_cov_frames(
    context_df,
    future_df,
    past_only,
    future_covs,
    extra_covs=None,
    drop_covs=None,
):
    if extra_covs is None:
        extra_covs = []
    if drop_covs is None:
        drop_covs = []

    past_only_use = [c for c in past_only if c not in drop_covs]
    future_covs_use = [c for c in future_covs if c not in drop_covs]
    extra_covs_use = [c for c in extra_covs if c not in drop_covs]

    ctx_cols = ["id", "timestamp", "target"] + past_only_use + future_covs_use + extra_covs_use
    fut_cols = ["id", "timestamp"] + future_covs_use + extra_covs_use

    for c in ctx_cols:
        if c not in context_df.columns:
            raise ValueError(f"Missing '{c}' in context_df")
    for c in fut_cols:
        if c not in future_df.columns:
            raise ValueError(f"Missing '{c}' in future_df")

    context_cov = context_df[ctx_cols].copy()
    future_cov = future_df[fut_cols].copy()

    # align dtypes between context and future for shared columns
    for col in fut_cols:
        if col in context_cov.columns and col in future_cov.columns:
            if pd.api.types.is_datetime64_any_dtype(context_cov[col]):
                continue
            try:
                future_cov[col] = future_cov[col].astype(context_cov[col].dtype, copy=False)
            except Exception:
                pass

    return context_cov, future_cov


def _run_predict_df(model, context_df, future_df, horizon, out_name, output_root=None):
    pred = predict_df_covariates(model, context_df, future_df, horizon=horizon)
    out_path = os.path.join(_ensure_outputs_dir(output_root), out_name)
    save_quantiles_csv(pred, out_path, verbose=False)
    return pred


# Robustness tests
def noise_test(model, df, horizon=30, seed=0, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Noise test: add random covariate", verbose)
    np.random.seed(seed)

    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    ctx = ctx.copy()
    fut = fut.copy()
    ctx["RandomNoise"] = np.random.randn(len(ctx)).astype(np.float32)
    fut["RandomNoise"] = np.random.randn(len(fut)).astype(np.float32)

    context_cov, future_cov = _prepare_cov_frames(ctx, fut, past_only, future_covs, extra_covs=["RandomNoise"])
    return _run_predict_df(model, context_cov, future_cov, horizon, f"noise_output{suffix}.csv", output_root)


def strong_noise_test(model, df, horizon=30, sigma=5.0, seed=0, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Strong noise: add Gaussian noise to covariates", verbose)
    np.random.seed(seed)

    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    ctx2 = ctx.copy()
    fut2 = fut.copy()

    noisy_cols = [c for c in past_only + future_covs if c not in {"Open", "SchoolHoliday", "StateHoliday", "DayOfWeek"}]
    for c in noisy_cols:
        ctx2[c] = (ctx2[c].astype(float) + sigma * np.random.randn(len(ctx2))).astype(np.float32)
        if c in fut2.columns:
            fut2[c] = (fut2[c].astype(float) + sigma * np.random.randn(len(fut2))).astype(np.float32)

    context_cov, future_cov = _prepare_cov_frames(ctx2, fut2, past_only, future_covs)
    return _run_predict_df(model, context_cov, future_cov, horizon, f"strong_noise_output{suffix}.csv", output_root)


def shuffle_test(model, df, horizon=30, seed=0, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Shuffle test: shuffle Promo to break temporal correlation", verbose)
    np.random.seed(seed)

    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    ctx2 = ctx.copy()
    fut2 = fut.copy()

    promo_all = np.concatenate([ctx2["Promo"].values, fut2["Promo"].values])
    promo_all = np.random.permutation(promo_all)

    ctx2["Promo"] = promo_all[:len(ctx2)]
    fut2["Promo"] = promo_all[len(ctx2):]

    context_cov, future_cov = _prepare_cov_frames(ctx2, fut2, past_only, future_covs)
    return _run_predict_df(model, context_cov, future_cov, horizon, f"shuffle_output{suffix}.csv", output_root)


def missing_future_test(model, df, horizon=30, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Missing future: mask future SchoolHoliday", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    fut2 = fut.copy()
    fut2["SchoolHoliday"] = np.nan

    context_cov, future_cov = _prepare_cov_frames(ctx, fut2, past_only, future_covs)
    return _run_predict_df(model, context_cov, future_cov, horizon, f"missing_future_output{suffix}.csv", output_root)


def time_shift_test(model, df, horizon=30, shift=7, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Time shift: shift Promo forward/backward", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    promo_all = np.concatenate([ctx["Promo"].values, fut["Promo"].values])
    promo_all = np.roll(promo_all, shift)

    ctx2 = ctx.copy()
    fut2 = fut.copy()
    ctx2["Promo"] = promo_all[:len(ctx2)]
    fut2["Promo"] = promo_all[len(ctx2):]

    context_cov, future_cov = _prepare_cov_frames(ctx2, fut2, past_only, future_covs)
    return _run_predict_df(model, context_cov, future_cov, horizon, f"time_shift_output{suffix}.csv", output_root)


def trend_break_test(model, df, horizon=30, jump=1.0, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Trend break: structural change in Promo", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    fut2 = fut.copy()
    half = len(fut2) // 2
    fut2.loc[half:, "Promo"] = (fut2.loc[half:, "Promo"].astype(float) + jump).astype(np.float32)

    context_cov, future_cov = _prepare_cov_frames(ctx, fut2, past_only, future_covs)
    return _run_predict_df(model, context_cov, future_cov, horizon, f"trend_break_output{suffix}.csv", output_root)


def feature_drop_test(model, df, horizon=30, drop_feature="Promo", suffix="", context_len=None, output_root=None, verbose=True):
    _log(f"[ROBUSTNESS] Feature drop: remove '{drop_feature}' from covariates", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    context_cov, future_cov = _prepare_cov_frames(ctx, fut, past_only, future_covs, drop_covs=[drop_feature])
    return _run_predict_df(model, context_cov, future_cov, horizon, f"feature_drop_output{suffix}.csv", output_root)


def partial_mask_test(model, df, horizon=30, frac=0.3, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Partial mask: mask last portion of Promo history", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    ctx2 = ctx.copy()
    n = len(ctx2)
    k = int(max(1, frac * n))
    ctx2.loc[n - k:, "Promo"] = 0

    context_cov, future_cov = _prepare_cov_frames(ctx2, fut, past_only, future_covs)
    return _run_predict_df(model, context_cov, future_cov, horizon, f"partial_mask_output{suffix}.csv", output_root)


def scaling_test(model, df, horizon=30, scale=10.0, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Scaling: rescale covariates", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    ctx2 = ctx.copy()
    fut2 = fut.copy()

    for c in [c for c in past_only + future_covs if c not in {"Open", "SchoolHoliday", "StateHoliday", "DayOfWeek"}]:
        ctx2[c] = (ctx2[c].astype(float) * scale).astype(np.float32)
        if c in fut2.columns:
            fut2[c] = (fut2[c].astype(float) * scale).astype(np.float32)

    context_cov, future_cov = _prepare_cov_frames(ctx2, fut2, past_only, future_covs)
    return _run_predict_df(model, context_cov, future_cov, horizon, f"scaling_output{suffix}.csv", output_root)


def long_horizon_test(model, df, horizon=90, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Long horizon: descriptive stability test (90 steps)", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    context_cov, future_cov = _prepare_cov_frames(ctx, fut, past_only, future_covs)
    return _run_predict_df(model, context_cov, future_cov, horizon, f"long_horizon_output{suffix}.csv", output_root)


def run_all_robustness_tests(model, df, store_id=None, context_len=None, output_root=None, verbose=False):
    suffix = "" if store_id is None else f"_store_{store_id}"

    noise_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    strong_noise_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    shuffle_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    missing_future_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    time_shift_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    trend_break_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    feature_drop_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    partial_mask_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    scaling_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)
    long_horizon_test(model, df, suffix=suffix, context_len=context_len, output_root=output_root, verbose=verbose)

# EVALUATION HELPERS

def _load_csv_if_exists(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _interval_width(df: pd.DataFrame) -> float:
    return float(np.mean(df["p90"] - df["p10"]))


def _pct_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(100 * np.mean((b - a) / (np.abs(a) + 1e-9)))


def _compare_forecasts(base: pd.DataFrame, test: pd.DataFrame, label_base: str, label_test: str) -> dict:
    m1 = base["median"].to_numpy(dtype=float)
    m2 = test["median"].to_numpy(dtype=float)
    if len(m1) != len(m2):
        raise ValueError(f"Cannot compare different horizons: {len(m1)} vs {len(m2)}")

    return {
        "MAE": float(np.mean(np.abs(m1 - m2))),
        "RMSE": float(np.sqrt(np.mean((m1 - m2) ** 2))),
        "PercentDifference": _pct_diff(m1, m2),
        f"Interval_{label_base}": _interval_width(base),
        f"Interval_{label_test}": _interval_width(test),
    }


# Create rbustness reports
def collect_robustness_summary(
    experiment: str,
    forecasts_root: str,
    reports_dir: str,
    best_context_length: int,
) -> tuple[str | None, str | None]:

    ensure_dir(reports_dir)

    ctx_cov_pred_dir = os.path.join(
        forecasts_root, f"ctx_{best_context_length}", "covariate", "predictions"
    )
    robust_dir = os.path.join("outputs", experiment, "robustness", f"ctx_{best_context_length}")

    if not os.path.exists(ctx_cov_pred_dir):
        print(f"[WARN] Missing baseline covariate predictions dir: {ctx_cov_pred_dir}")
        return None, None
    if not os.path.exists(robust_dir):
        print(f"[WARN] Missing robustness dir: {robust_dir}")
        return None, None

    robustness_map = {
        "Noise": "noise_output",
        "StrongNoise": "strong_noise_output",
        "Shuffle": "shuffle_output",
        "MissingFuture": "missing_future_output",
        "TimeShift": "time_shift_output",
        "TrendBreak": "trend_break_output",
        "FeatureDrop": "feature_drop_output",
        "PartialMask": "partial_mask_output",
        "Scaling": "scaling_output",
    }

    # detect store ids from baseline covariate predictions
    store_ids = []
    for f in os.listdir(ctx_cov_pred_dir):
        if f.startswith("forecast_store_") and f.endswith(".csv"):
            sid = f.replace("forecast_store_", "").replace(".csv", "")
            store_ids.append(sid)

    if not store_ids:
        print(f"[WARN] No stores detected in {ctx_cov_pred_dir}")
        return None, None

    records: list[dict] = []
    for sid in store_ids:
        base_path = os.path.join(ctx_cov_pred_dir, f"forecast_store_{sid}.csv")
        cov = _load_csv_if_exists(base_path)
        if cov is None or not all(c in cov.columns for c in ["p10", "median", "p90"]):
            continue

        for test_name, prefix in robustness_map.items():
            test_path = os.path.join(robust_dir, f"{prefix}_store_{sid}.csv")
            test_df = _load_csv_if_exists(test_path)
            if test_df is None or not all(c in test_df.columns for c in ["p10", "median", "p90"]):
                continue

            stats = _compare_forecasts(cov, test_df, "Covariates", test_name)
            records.append(
                {
                    "context_length": int(best_context_length),
                    "store_id": int(sid) if str(sid).isdigit() else sid,
                    "test_name": test_name,
                    **stats,
                }
            )

    if not records:
        print("[WARN] Robustness report not produced (no matching outputs found).")
        return None, None

    df = pd.DataFrame(records)
    per_store_path = os.path.join(reports_dir, "robustness_per_store_merged.csv")
    df.to_csv(per_store_path, index=False)

    numeric_cols = [c for c in df.columns if c not in ["store_id", "test_name", "context_length"]]
    summary = df.groupby("test_name")[numeric_cols].agg(["mean", "std", "median"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    summary_path = os.path.join(reports_dir, "robustness_summary.csv")
    summary.to_csv(summary_path, index=False)

    return per_store_path, summary_path