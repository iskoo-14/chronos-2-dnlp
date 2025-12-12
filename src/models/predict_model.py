import numpy as np
import torch
from chronos import Chronos2Pipeline


def load_model(model_name: str = "amazon/chronos-2"):
    print(f"Loading Chronos-2 model: {model_name}")
    return Chronos2Pipeline.from_pretrained(model_name)


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x)


def _as_1d_float(series) -> torch.Tensor:
    s = _to_tensor(series).flatten().to(torch.float32)
    return s


def _as_FT_float(cov) -> torch.Tensor:
    """
    Normalize covariates to shape (F, T).
    Accepts:
      - (T, F) or (F, T) as numpy or torch
    """
    c = _to_tensor(cov).to(torch.float32)

    if c.ndim != 2:
        raise ValueError(f"Covariates must be 2D, got shape={tuple(c.shape)}")

    # If it's (T,F), transpose to (F,T)
    # Heuristic: T is much larger than F in our dataset.
    if c.shape[0] > c.shape[1]:
        c = c.T

    return c


def _pick_quantiles(pred_tensor: torch.Tensor):
    """
    pred_tensor shape: (n_variates, n_quantiles, horizon)
    We try to return p10, median, p90.
    """
    qdim = pred_tensor.shape[1]

    if qdim >= 9:
        i10, i50, i90 = 0, 4, 8
    else:
        i10 = 0
        i50 = qdim // 2
        i90 = qdim - 1

    p10 = pred_tensor[:, i10, :].detach().cpu().numpy().flatten()
    med = pred_tensor[:, i50, :].detach().cpu().numpy().flatten()
    p90 = pred_tensor[:, i90, :].detach().cpu().numpy().flatten()
    return p10, med, p90


def predict_univariate(model, series, horizon: int = 30):
    """
    Univariate forecast using only the target series.
    Chronos2 accepts:
      - a list of 1D tensors
      - or a 3D tensor (batch, n_variates, T)
    We'll use list-of-1D to avoid shape mistakes.
    """
    series = _as_1d_float(series)

    pred_list = model.predict(
        inputs=[series],
        prediction_length=horizon,
        limit_prediction_length=True
    )

    pred = pred_list[0]  # (1, n_quantiles, horizon)
    p10, med, p90 = _pick_quantiles(pred)

    return {"p10": p10, "median": med, "p90": p90}


def predict_covariates(model, series, past_cov, future_cov_full, horizon: int = 30, debug: bool = False):
    """
    Multivariate with covariates using the list-of-dict API.
    IMPORTANT:
      - past_covariates values must be 1D length T
      - future_covariates values must be 1D length horizon
    """
    target = _as_1d_float(series)

    past_cov = _as_FT_float(past_cov)                 # (F,T)
    future_cov_full = _as_FT_float(future_cov_full)   # (F,T) after normalization

    F, T = past_cov.shape

    if debug:
        print("\n=== DEBUG predict_covariates ===")
        print(f"Past cov shape (F,T) = {past_cov.shape}")
        print(f"Future cov full shape (F,T) = {future_cov_full.shape}")
        print(f"Horizon = {horizon}")

    # past cov dict: each feature is 1D length T
    past_cov_dict = {f"f{i}": past_cov[i].flatten() for i in range(F)}

    # future cov dict: each feature is 1D length horizon
    future_cov_dict = {}
    for i in range(F):
        full_series = future_cov_full[i].flatten()  # length T
        future_slice = full_series[-horizon:]       # length horizon

        if future_slice.numel() != horizon:
            raise ValueError(
                f"Future cov f{i} has length {future_slice.numel()} but horizon={horizon}. "
                f"Check covariate shapes."
            )

        future_cov_dict[f"f{i}"] = future_slice

    task = [{
        "target": target,
        "past_covariates": past_cov_dict,
        "future_covariates": future_cov_dict
    }]

    pred_list = model.predict(
        inputs=task,
        prediction_length=horizon,
        limit_prediction_length=True
    )

    pred = pred_list[0]  # (n_targets(=1), n_quantiles, horizon)
    p10, med, p90 = _pick_quantiles(pred)

    return {"p10": p10, "median": med, "p90": p90}
