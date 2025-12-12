import numpy as np
import torch
from chronos import Chronos2Pipeline


# ===================================================================
# LOAD MODEL
# ===================================================================

def load_model(model_name="amazon/chronos-2"):
    print(f"Loading Chronos-2 model: {model_name}")
    return Chronos2Pipeline.from_pretrained(model_name)


# ===================================================================
# INTERNAL NORMALIZATION + QUANTILE EXTRACTION
# ===================================================================

def _normalize_output(pred):
    """
    Normalizes the raw Chronos-2 prediction output into:
        - median forecast
        - p10 forecast
        - p90 forecast

    Chronos-2 returns: [ tensor of shape (1, 30, 30) ]
    meaning:
        (1 sample, 30 parallel outputs, horizon=30)

    We:
        1) remove sample dim -> (30, 30)
        2) compute quantiles across axis=0
        3) return dict {median, p10, p90}
    """

    # pred is always a list -> extract tensor
    if isinstance(pred, list):
        pred = pred[0]

    # tensor -> numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    pred = np.asarray(pred)

    # remove sample dimension (1, 30, 30) -> (30, 30)
    if pred.ndim == 3:
        pred = pred.squeeze(0)

    # pred shape now = (num_internal_paths, horizon)
    # i.e. (30, 30)

    # Compute quantiles across the internal stochastic outputs
    median_forecast = np.median(pred, axis=0)
    p10_forecast = np.percentile(pred, 10, axis=0)
    p90_forecast = np.percentile(pred, 90, axis=0)

    return {
        "median": median_forecast.astype(float),
        "p10": p10_forecast.astype(float),
        "p90": p90_forecast.astype(float)
    }


# ===================================================================
# UNIVARIATE PREDICTION
# ===================================================================

def predict_univariate(model, series, horizon=30):
    """
    Computes quantile forecasts for univariate series.
    Returns a dict: {median, p10, p90}
    """

    series = torch.tensor(series.astype(np.float32))

    # Chronos-2 expects list of input tensors
    inputs = [series]

    pred = model.predict(
        inputs,
        prediction_length=horizon
    )

    return _normalize_output(pred)


# ===================================================================
# MULTIVARIATE / COVARIATE PREDICTION
# ===================================================================

def predict_covariates(model, series, past_cov, fut_cov, horizon=30):
    series = torch.as_tensor(series.astype(np.float32))

    past_cov = torch.as_tensor(past_cov.astype(np.float32).T)
    fut_cov = torch.as_tensor(past_cov[:, -horizon:])

    inputs = [series, past_cov, fut_cov]

    pred = model.predict(inputs, prediction_length=horizon)

    return _normalize_output(pred)


