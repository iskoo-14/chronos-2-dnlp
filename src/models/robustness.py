import os
import pandas as pd
import torch

from src.models.predict_model import predict_covariates


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def ensure_outputs():
    """Ensure the outputs directory exists."""
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_result(name, pred_dict):
    """Save robustness prediction results as CSV."""
    out_dir = ensure_outputs()
    out_path = os.path.join(out_dir, name)

    df = pd.DataFrame({
        "p10": pred_dict["p10"],
        "median": pred_dict["median"],
        "p90": pred_dict["p90"]
    })

    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------
# 1. Noise Test
# ---------------------------------------------------------------------

def noise_test(model, df, target, covariates):
    """
    Adds a random noise covariate and checks whether the model ignores it.
    """
    df_copy = df.copy()
    df_copy["RandomNoiseRobust"] = torch.randn(len(df_copy)).tolist()

    # Rebuild covariates including noise column
    noise_cov = torch.cat([
        covariates,
        torch.tensor(df_copy["RandomNoiseRobust"].values, dtype=torch.float32).unsqueeze(0)
    ], dim=0)

    pred = predict_covariates(model, target, noise_cov, noise_cov)

    save_result("noise_output.csv", pred)
    return pred


# ---------------------------------------------------------------------
# 2. Shuffle Test
# ---------------------------------------------------------------------

def shuffle_test(model, df, target, covariates):
    """
    Shuffles Promo values to destroy correlation.
    """
    df_copy = df.copy()
    
    if "Promo" not in df_copy.columns:
        raise ValueError("Promo column not found for shuffle test.")

    shuffled = df_copy["Promo"].sample(frac=1, random_state=42).values
    df_copy["PromoShuffled"] = shuffled

    # Replace Promo in covariates
    promo_idx = None
    # Identify Promo index inside covariate rows by scanning the original covariates
    # The first row that exactly matches Promo values is our target.
    for i in range(covariates.shape[0]):
        if torch.allclose(covariates[i], torch.tensor(df_copy["Promo"].astype(float).values)):
            promo_idx = i
            break

    if promo_idx is not None:
        cov_shuf = covariates.clone()
        cov_shuf[promo_idx] = torch.tensor(shuffled, dtype=torch.float32)
    else:
        # Fallback: append shuffled promo covariate
        cov_shuf = torch.cat([
            covariates,
            torch.tensor(shuffled, dtype=torch.float32).unsqueeze(0)
        ], dim=0)

    pred = predict_covariates(model, target, cov_shuf, cov_shuf)

    save_result("shuffle_output.csv", pred)
    return pred


# ---------------------------------------------------------------------
# 3. Missing Future Test
# ---------------------------------------------------------------------

def missing_future_test(model, df, target, covariates):
    """
    Removes future covariate values such as SchoolHoliday to test
    how much the model depends on known-future information.
    """

    df_copy = df.copy()

    if "SchoolHoliday" not in df_copy.columns:
        raise ValueError("SchoolHoliday column missing for missing-future test.")

    # Remove last horizon values
    horizon = 30
    missing = df_copy["SchoolHoliday"].astype(float).values.copy()
    missing[-horizon:] = 0  # Mask future values

    # Replace covariate row for SchoolHoliday
    sh_idx = None
    for i in range(covariates.shape[0]):
        if torch.allclose(covariates[i], torch.tensor(df_copy["SchoolHoliday"].astype(float).values)):
            sh_idx = i
            break

    if sh_idx is not None:
        cov_missing = covariates.clone()
        cov_missing[sh_idx] = torch.tensor(missing, dtype=torch.float32)
    else:
        cov_missing = torch.cat([
            covariates,
            torch.tensor(missing, dtype=torch.float32).unsqueeze(0)
        ], dim=0)

    pred = predict_covariates(model, target, cov_missing, cov_missing)

    save_result("missing_future_output.csv", pred)
    return pred
