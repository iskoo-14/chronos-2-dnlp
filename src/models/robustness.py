import os
import numpy as np
import pandas as pd

from src.models.predict_model import predict_df_covariates, save_quantiles_csv


# ------------------------------------------------------------
# LOGGING UTILS
# ------------------------------------------------------------

def _log(msg, verbose):
    if verbose:
        print(msg)



# ------------------------------------------------------------
# PATH UTILS 
# ------------------------------------------------------------

def _ensure_outputs_dir(output_root=None):
    if output_root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_root = os.path.join(root, "outputs")
    os.makedirs(output_root, exist_ok=True)
    return output_root


# ------------------------------------------------------------
# CONTEXT / FUTURE SPLIT 
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# COVARIATE SETS 
# ------------------------------------------------------------

def _base_covariates():
    past_only = ["Customers"]
    future_covs = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]
    return past_only, future_covs


# ------------------------------------------------------------
# PREPARE CONTEXT / FUTURE FRAMES 
# ------------------------------------------------------------

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

    # align dtypes between context and future for shared columns (Chronos requires matching types)
    for col in fut_cols:
        if col in context_cov.columns and col in future_cov.columns:
            if pd.api.types.is_datetime64_any_dtype(context_cov[col]):
                # keep datetime as-is
                continue
            try:
                future_cov[col] = future_cov[col].astype(context_cov[col].dtype, copy=False)
            except Exception:
                # fallback: leave as-is if casting fails
                pass

    return context_cov, future_cov


# ------------------------------------------------------------
# RUN PREDICT_DF 
# ------------------------------------------------------------

def _run_predict_df(model, context_df, future_df, horizon, out_name, output_root=None):
    pred = predict_df_covariates(model, context_df, future_df, horizon=horizon)
    out_path = os.path.join(_ensure_outputs_dir(output_root), out_name)
    save_quantiles_csv(pred, out_path, verbose=False)
    return pred


# ------------------------------------------------------------
# ROBUSTNESS TESTS 
# ------------------------------------------------------------

def noise_test(model, df, horizon=30, seed=0, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Noise test: add random covariate", verbose)
    np.random.seed(seed)

    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    ctx = ctx.copy()
    fut = fut.copy()
    ctx["RandomNoise"] = np.random.randn(len(ctx)).astype(np.float32)
    fut["RandomNoise"] = np.random.randn(len(fut)).astype(np.float32)

    context_cov, future_cov = _prepare_cov_frames(
        ctx, fut, past_only, future_covs, extra_covs=["RandomNoise"]
    )
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"noise_output{suffix}.csv",
        output_root=output_root,
    )


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
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"strong_noise_output{suffix}.csv",
        output_root=output_root,
    )


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
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"shuffle_output{suffix}.csv",
        output_root=output_root,
    )


def missing_future_test(model, df, horizon=30, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Missing future: mask future SchoolHoliday", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    fut2 = fut.copy()
    # use NaN (float) to simulate missing information (Chronos will mask)
    fut2["SchoolHoliday"] = np.nan

    context_cov, future_cov = _prepare_cov_frames(ctx, fut2, past_only, future_covs)
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"missing_future_output{suffix}.csv",
        output_root=output_root,
    )


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
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"time_shift_output{suffix}.csv",
        output_root=output_root,
    )


def trend_break_test(model, df, horizon=30, jump=1.0, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Trend break: structural change in Promo", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    fut2 = fut.copy()
    half = len(fut2) // 2
    fut2.loc[half:, "Promo"] = (fut2.loc[half:, "Promo"].astype(float) + jump).astype(np.float32)

    context_cov, future_cov = _prepare_cov_frames(ctx, fut2, past_only, future_covs)
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"trend_break_output{suffix}.csv",
        output_root=output_root,
    )


def feature_drop_test(model, df, horizon=30, drop_feature="Promo", suffix="", context_len=None, output_root=None, verbose=True):
    _log(f"[ROBUSTNESS] Feature drop: remove '{drop_feature}' from covariates", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    context_cov, future_cov = _prepare_cov_frames(
        ctx,
        fut,
        past_only,
        future_covs,
        drop_covs=[drop_feature],
    )
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"feature_drop_output{suffix}.csv",
        output_root=output_root,
    )


def partial_mask_test(model, df, horizon=30, frac=0.3, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Partial mask: mask last portion of Promo history", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    ctx2 = ctx.copy()
    n = len(ctx2)
    k = int(max(1, frac * n))
    ctx2.loc[n - k:, "Promo"] = 0

    context_cov, future_cov = _prepare_cov_frames(ctx2, fut, past_only, future_covs)
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"partial_mask_output{suffix}.csv",
        output_root=output_root,
    )


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
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"scaling_output{suffix}.csv",
        output_root=output_root,
    )


def long_horizon_test(model, df, horizon=90, suffix="", context_len=None, output_root=None, verbose=True):
    _log("[ROBUSTNESS] Long horizon: descriptive stability test (90 steps)", verbose)
    ctx, fut = _make_context_future(df, horizon=horizon, context_len=context_len)
    past_only, future_covs = _base_covariates()

    context_cov, future_cov = _prepare_cov_frames(ctx, fut, past_only, future_covs)
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"long_horizon_output{suffix}.csv",
        output_root=output_root,
    )


# ------------------------------------------------------------
# MULTISTORE RUNNER 
# ------------------------------------------------------------

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
