import os
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

def _ensure_outputs_dir():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _save_pred_csv(pred: dict, filename: str):
    out_dir = _ensure_outputs_dir()
    path = os.path.join(out_dir, filename)
    pd.DataFrame(
        {"p10": pred["p10"], "median": pred["median"], "p90": pred["p90"]}
    ).to_csv(path, index=False)
    return path


# ------------------------------------------------------------
# CHRONOS INPUT BUILDERS (CORRECT FORMAT)
# ------------------------------------------------------------

def _to_1d_float(x):
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32)


def _build_task_from_df(df, target, horizon=30, known_future_cols=None, extra_past_cols=None):
    """
    Build a Chronos-2 dict task:
      - target: 1D length T
      - past_covariates: dict of 1D arrays length T
      - future_covariates: dict of 1D arrays length horizon, subset of past_covariates keys
    """
    if known_future_cols is None:
        known_future_cols = []
    if extra_past_cols is None:
        extra_past_cols = []

    T = len(target)
    target_1d = _to_1d_float(target)

    # Basic covariate set: take numeric columns except obvious non-features
    drop_cols = {"Sales", "Date"}
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Ensure required columns exist if asked
    for c in known_future_cols + extra_past_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in df.")

    cov_cols = sorted(set(numeric_cols + extra_past_cols + known_future_cols))

    # past covariates: full length T
    past_cov = {}
    for c in cov_cols:
        v = df[c].to_numpy()
        if len(v) != T:
            raise ValueError(f"Covariate '{c}' length {len(v)} != target length {T}")
        v = np.nan_to_num(v, nan=0.0).astype(np.float32)
        past_cov[c] = v

    # future covariates: only last horizon values, only for known_future_cols
    fut_cov = {}
    for c in known_future_cols:
        fut = past_cov[c][-horizon:]
        if len(fut) != horizon:
            raise ValueError(f"Future cov '{c}' must have length horizon={horizon}, got {len(fut)}")
        fut_cov[c] = fut

    task = {
        "target": target_1d,
        "past_covariates": past_cov
    }
    if len(fut_cov) > 0:
        task["future_covariates"] = fut_cov

    return task


def _run_chronos(model, df, target, horizon=30, known_future_cols=None, extra_past_cols=None):
    task = _build_task_from_df(
        df=df,
        target=target,
        horizon=horizon,
        known_future_cols=known_future_cols,
        extra_past_cols=extra_past_cols,
    )

    # Chronos2 returns list[Tensor] with shape (n_variates, n_quantiles, horizon)
    # Here we forecast only 1 target => take first series/variate
    out = model.predict([task], prediction_length=horizon)[0]  # torch.Tensor

    # quantiles are typically 0.1..0.9; we take p10, median (0.5), p90
    # out shape: (n_variates=1, n_quantiles, horizon)
    out_np = out.detach().cpu().numpy()
    qdim = out_np.shape[1]

    # defensively map indices: assume quantiles ordered
    # p10 -> first, median -> middle, p90 -> last
    p10 = out_np[0, 0, :].tolist()
    med = out_np[0, qdim // 2, :].tolist()
    p90 = out_np[0, -1, :].tolist()

    return {"p10": p10, "median": med, "p90": p90}


# ------------------------------------------------------------
# ROBUSTNESS TESTS (WORKING WITH df, NOT WITH cov matrix)
# ------------------------------------------------------------

def noise_test(model, df, target, covariates=None, horizon=30):
    """
    Adds an irrelevant random noise covariate (past-only).
    A robust model should ignore it (little change vs baseline).
    """
    df2 = df.copy()
    df2["NoiseRandom"] = np.random.randn(len(df2)).astype(np.float32)

    pred = _run_chronos(
        model=model,
        df=df2,
        target=target,
        horizon=horizon,
        known_future_cols=[],                 # noise is past-only
        extra_past_cols=["NoiseRandom"],
    )
    _save_pred_csv(pred, "noise_output.csv")
    return pred


def strong_noise_test(model, df, target, covariates=None, horizon=30, scale=10.0):
    """
    Adds strong Gaussian noise to numeric covariates (past-only).
    """
    df2 = df.copy()

    drop_cols = {"Sales", "Date"}
    numeric_cols = [c for c in df2.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df2[c])]

    for c in numeric_cols:
        v = df2[c].to_numpy(dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        v = v + scale * np.random.randn(len(v)).astype(np.float32)
        df2[c] = v

    pred = _run_chronos(
        model=model,
        df=df2,
        target=target,
        horizon=horizon,
        known_future_cols=[],
        extra_past_cols=[],
    )
    _save_pred_csv(pred, "strong_noise_output.csv")
    return pred


def shuffle_test(model, df, target, covariates=None, horizon=30):
    """
    Shuffles Promo to break its temporal structure.
    Promo is treated as known-future covariate (we also pass future slice).
    """
    if "Promo" not in df.columns:
        raise ValueError("Promo not found for shuffle test")

    df2 = df.copy()
    df2["Promo"] = df2["Promo"].sample(frac=1, random_state=42).to_numpy()

    pred = _run_chronos(
        model=model,
        df=df2,
        target=target,
        horizon=horizon,
        known_future_cols=["Promo"],
        extra_past_cols=[],
    )
    _save_pred_csv(pred, "shuffle_output.csv")
    return pred


def missing_future_test(model, df, target, covariates=None, horizon=30):
    """
    Masks last horizon values of SchoolHoliday (known-future), to simulate missing future info.
    """
    if "SchoolHoliday" not in df.columns:
        raise ValueError("SchoolHoliday not found for missing-future test")

    df2 = df.copy()
    sh = df2["SchoolHoliday"].to_numpy(dtype=np.float32)
    sh = np.nan_to_num(sh, nan=0.0)
    sh[-horizon:] = 0.0
    df2["SchoolHoliday"] = sh

    pred = _run_chronos(
        model=model,
        df=df2,
        target=target,
        horizon=horizon,
        known_future_cols=["SchoolHoliday"],
        extra_past_cols=[],
    )
    _save_pred_csv(pred, "missing_future_output.csv")
    return pred


def time_shift_test(model, df, target, covariates=None, horizon=30, shift=7):
    """
    Shifts Promo (known-future) by 'shift' days (circular).
    """
    if "Promo" not in df.columns:
        raise ValueError("Promo not found for time-shift test")

    df2 = df.copy()
    promo = df2["Promo"].to_numpy(dtype=np.float32)
    promo = np.nan_to_num(promo, nan=0.0)
    df2["Promo"] = np.roll(promo, shift)

    pred = _run_chronos(
        model=model,
        df=df2,
        target=target,
        horizon=horizon,
        known_future_cols=["Promo"],
        extra_past_cols=[],
    )
    _save_pred_csv(pred, "time_shift_output.csv")
    return pred


def trend_break_test(model, df, target, covariates=None, horizon=30, magnitude=0.5):
    """
    Applies a multiplicative trend break to a known-future covariate.
    Here we use Promo if available, otherwise SchoolHoliday.
    """
    df2 = df.copy()

    col = None
    if "Promo" in df2.columns:
        col = "Promo"
    elif "SchoolHoliday" in df2.columns:
        col = "SchoolHoliday"
    else:
        raise ValueError("Need Promo or SchoolHoliday for trend-break test")

    v = df2[col].to_numpy(dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0)
    split = int(0.8 * len(v))
    v[split:] = v[split:] * (1.0 + float(magnitude))
    df2[col] = v

    pred = _run_chronos(
        model=model,
        df=df2,
        target=target,
        horizon=horizon,
        known_future_cols=[col],
        extra_past_cols=[],
    )
    _save_pred_csv(pred, "trend_break_output.csv")
    return pred
