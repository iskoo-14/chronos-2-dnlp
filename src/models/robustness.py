import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# OUTPUT UTILS
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
        {
            "p10": pred["p10"],
            "median": pred["median"],
            "p90": pred["p90"],
        }
    ).to_csv(path, index=False)
    return path


# ------------------------------------------------------------
# CHRONOS TASK BUILDERS (CORRECT API)
# ------------------------------------------------------------

def _to_1d_float(x):
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32)


def _build_task_from_df(
    df,
    target,
    horizon=30,
    known_future_cols=None,
    extra_past_cols=None,
):
    if known_future_cols is None:
        known_future_cols = []
    if extra_past_cols is None:
        extra_past_cols = []

    T = len(target)
    target_1d = _to_1d_float(target)

    drop_cols = {"Sales", "Date"}
    numeric_cols = [
        c
        for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    for c in known_future_cols + extra_past_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in df")

    cov_cols = sorted(set(numeric_cols + known_future_cols + extra_past_cols))

    past_cov = {}
    for c in cov_cols:
        v = df[c].to_numpy(dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        if len(v) != T:
            raise ValueError(f"Covariate '{c}' length mismatch")
        past_cov[c] = v

    fut_cov = {}
    for c in known_future_cols:
        fut = past_cov[c][-horizon:]
        if len(fut) != horizon:
            raise ValueError(f"Future cov '{c}' must have length {horizon}")
        fut_cov[c] = fut

    task = {
        "target": target_1d,
        "past_covariates": past_cov,
    }
    if len(fut_cov) > 0:
        task["future_covariates"] = fut_cov

    return task


def _run_chronos(
    model,
    df,
    target,
    horizon=30,
    known_future_cols=None,
    extra_past_cols=None,
):
    task = _build_task_from_df(
        df=df,
        target=target,
        horizon=horizon,
        known_future_cols=known_future_cols,
        extra_past_cols=extra_past_cols,
    )

    out = model.predict([task], prediction_length=horizon)[0]
    out_np = out.detach().cpu().numpy()

    qdim = out_np.shape[1]

    return {
        "p10": out_np[0, 0, :].tolist(),
        "median": out_np[0, qdim // 2, :].tolist(),
        "p90": out_np[0, -1, :].tolist(),
    }


# ------------------------------------------------------------
# ROBUSTNESS TESTS
# ------------------------------------------------------------

def noise_test(model, df, target, covariates=None, horizon=30):
    df2 = df.copy()
    df2["NoiseRandom"] = np.random.randn(len(df2)).astype(np.float32)

    pred = _run_chronos(
        model,
        df2,
        target,
        horizon=horizon,
        known_future_cols=[],
        extra_past_cols=["NoiseRandom"],
    )
    _save_pred_csv(pred, "noise_output.csv")
    return pred


def strong_noise_test(model, df, target, covariates=None, horizon=30, scale=10.0):
    df2 = df.copy()

    drop_cols = {"Sales", "Date"}
    numeric_cols = [
        c
        for c in df2.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df2[c])
    ]

    for c in numeric_cols:
        v = df2[c].to_numpy(dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        df2[c] = v + scale * np.random.randn(len(v)).astype(np.float32)

    pred = _run_chronos(model, df2, target, horizon=horizon)
    _save_pred_csv(pred, "strong_noise_output.csv")
    return pred


def shuffle_test(model, df, target, covariates=None, horizon=30):
    if "Promo" not in df.columns:
        raise ValueError("Promo not found for shuffle test")

    df2 = df.copy()
    df2["Promo"] = df2["Promo"].sample(frac=1, random_state=42).to_numpy()

    pred = _run_chronos(
        model,
        df2,
        target,
        horizon=horizon,
        known_future_cols=["Promo"],
    )
    _save_pred_csv(pred, "shuffle_output.csv")
    return pred


def missing_future_test(model, df, target, covariates=None, horizon=30):
    if "SchoolHoliday" not in df.columns:
        raise ValueError("SchoolHoliday not found")

    df2 = df.copy()
    sh = df2["SchoolHoliday"].to_numpy(dtype=np.float32)
    sh = np.nan_to_num(sh, nan=0.0)
    sh[-horizon:] = 0.0
    df2["SchoolHoliday"] = sh

    pred = _run_chronos(
        model,
        df2,
        target,
        horizon=horizon,
        known_future_cols=["SchoolHoliday"],
    )
    _save_pred_csv(pred, "missing_future_output.csv")
    return pred


def time_shift_test(model, df, target, covariates=None, horizon=30, shift=7):
    if "Promo" not in df.columns:
        raise ValueError("Promo not found")

    df2 = df.copy()
    promo = df2["Promo"].to_numpy(dtype=np.float32)
    promo = np.nan_to_num(promo, nan=0.0)
    df2["Promo"] = np.roll(promo, shift)

    pred = _run_chronos(
        model,
        df2,
        target,
        horizon=horizon,
        known_future_cols=["Promo"],
    )
    _save_pred_csv(pred, "time_shift_output.csv")
    return pred


def trend_break_test(model, df, target, covariates=None, horizon=30, magnitude=0.5):
    df2 = df.copy()

    if "Promo" in df2.columns:
        col = "Promo"
    elif "SchoolHoliday" in df2.columns:
        col = "SchoolHoliday"
    else:
        raise ValueError("Need Promo or SchoolHoliday")

    v = df2[col].to_numpy(dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0)
    split = int(0.8 * len(v))
    v[split:] *= (1.0 + magnitude)
    df2[col] = v

    pred = _run_chronos(
        model,
        df2,
        target,
        horizon=horizon,
        known_future_cols=[col],
    )
    _save_pred_csv(pred, "trend_break_output.csv")
    return pred


def feature_drop_test(model, df, target, covariates=None, horizon=30):
    df2 = df.copy()
    drop_cols = ["Promo", "SchoolHoliday"]
    drop_cols = [c for c in drop_cols if c in df2.columns]
    df2 = df2.drop(columns=drop_cols)

    pred = _run_chronos(model, df2, target, horizon=horizon)
    _save_pred_csv(pred, "feature_drop_output.csv")
    return pred


def partial_mask_test(model, df, target, covariates=None, horizon=30, mask_ratio=0.3):
    df2 = df.copy()
    split = int(len(df2) * (1 - mask_ratio))

    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            v = df2[c].to_numpy(dtype=np.float32)
            v[split:] = 0.0
            df2[c] = v

    pred = _run_chronos(model, df2, target, horizon=horizon)
    _save_pred_csv(pred, "partial_mask_output.csv")
    return pred


def scaling_test(model, df, target, covariates=None, horizon=30, scale=2.0):
    df2 = df.copy()

    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].to_numpy(dtype=np.float32) * scale

    pred = _run_chronos(model, df2, target, horizon=horizon)
    _save_pred_csv(pred, "scaling_output.csv")
    return pred


def long_horizon_test(model, df, target, covariates=None, horizon=90):
    pred = _run_chronos(model, df, target, horizon=horizon)
    _save_pred_csv(pred, "long_horizon_output.csv")
    return pred
