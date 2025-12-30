import os
import numpy as np
import pandas as pd

from src.models.predict_model import predict_df_covariates, save_quantiles_csv



# ------------------------------------------------------------
# PATH UTILS 
# ------------------------------------------------------------

def _ensure_outputs_dir():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ------------------------------------------------------------
# CONTEXT / FUTURE SPLIT 
# ------------------------------------------------------------

def _make_context_future(df, horizon=30, context_len=256):
    if "timestamp" not in df.columns:
        raise ValueError("df must contain 'timestamp' column")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) <= horizon:
        raise ValueError("Not enough rows for the requested horizon")

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

    return context_cov, future_cov


# ------------------------------------------------------------
# RUN PREDICT_DF 
# ------------------------------------------------------------

def _run_predict_df(model, context_df, future_df, horizon, out_name):
    pred = predict_df_covariates(model, context_df, future_df, horizon=horizon)
    out_path = os.path.join(_ensure_outputs_dir(), out_name)
    save_quantiles_csv(pred, out_path)
    return pred


# ------------------------------------------------------------
# ROBUSTNESS TESTS 
# ------------------------------------------------------------

def noise_test(model, df, horizon=30, seed=0, suffix=""):
    print("[ROBUSTNESS] Noise test: add random covariate")
    np.random.seed(seed)

    ctx, fut = _make_context_future(df, horizon=horizon)
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
    )


def strong_noise_test(model, df, horizon=30, sigma=5.0, seed=0, suffix=""):
    print("[ROBUSTNESS] Strong noise: add Gaussian noise to covariates")
    np.random.seed(seed)

    ctx, fut = _make_context_future(df, horizon=horizon)
    past_only, future_covs = _base_covariates()

    ctx2 = ctx.copy()
    fut2 = fut.copy()

    noisy_cols = past_only + future_covs
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
    )


def shuffle_test(model, df, horizon=30, seed=0, suffix=""):
    print("[ROBUSTNESS] Shuffle test: shuffle Promo to break temporal correlation")
    np.random.seed(seed)

    ctx, fut = _make_context_future(df, horizon=horizon)
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
    )


def missing_future_test(model, df, horizon=30, suffix=""):
    print("[ROBUSTNESS] Missing future: mask future SchoolHoliday")
    ctx, fut = _make_context_future(df, horizon=horizon)
    past_only, future_covs = _base_covariates()

    fut2 = fut.copy()
    fut2["SchoolHoliday"] = 0

    context_cov, future_cov = _prepare_cov_frames(ctx, fut2, past_only, future_covs)
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"missing_future_output{suffix}.csv",
    )


def time_shift_test(model, df, horizon=30, shift=7, suffix=""):
    print("[ROBUSTNESS] Time shift: shift Promo forward/backward")
    ctx, fut = _make_context_future(df, horizon=horizon)
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
    )


def trend_break_test(model, df, horizon=30, jump=1.0, suffix=""):
    print("[ROBUSTNESS] Trend break: structural change in Promo")
    ctx, fut = _make_context_future(df, horizon=horizon)
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
    )


def feature_drop_test(model, df, horizon=30, drop_feature="Promo", suffix=""):
    print(f"[ROBUSTNESS] Feature drop: remove '{drop_feature}' from covariates")
    ctx, fut = _make_context_future(df, horizon=horizon)
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
    )


def partial_mask_test(model, df, horizon=30, frac=0.3, suffix=""):
    print("[ROBUSTNESS] Partial mask: mask last portion of Promo history")
    ctx, fut = _make_context_future(df, horizon=horizon)
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
    )


def scaling_test(model, df, horizon=30, scale=10.0, suffix=""):
    print("[ROBUSTNESS] Scaling: rescale covariates")
    ctx, fut = _make_context_future(df, horizon=horizon)
    past_only, future_covs = _base_covariates()

    ctx2 = ctx.copy()
    fut2 = fut.copy()

    for c in past_only + future_covs:
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
    )


def long_horizon_test(model, df, horizon=90, suffix=""):
    print("[ROBUSTNESS] Long horizon: descriptive stability test (90 steps)")
    ctx, fut = _make_context_future(df, horizon=horizon)
    past_only, future_covs = _base_covariates()

    context_cov, future_cov = _prepare_cov_frames(ctx, fut, past_only, future_covs)
    return _run_predict_df(
        model,
        context_cov,
        future_cov,
        horizon,
        f"long_horizon_output{suffix}.csv",
    )


# ------------------------------------------------------------
# MULTISTORE RUNNER 
# ------------------------------------------------------------

def run_all_robustness_tests(model, df, store_id=None):
    suffix = "" if store_id is None else f"_store_{store_id}"

    noise_test(model, df, suffix=suffix)
    strong_noise_test(model, df, suffix=suffix)
    shuffle_test(model, df, suffix=suffix)
    missing_future_test(model, df, suffix=suffix)
    time_shift_test(model, df, suffix=suffix)
    trend_break_test(model, df, suffix=suffix)
    feature_drop_test(model, df, suffix=suffix)
    partial_mask_test(model, df, suffix=suffix)
    scaling_test(model, df, suffix=suffix)
    long_horizon_test(model, df, suffix=suffix)
