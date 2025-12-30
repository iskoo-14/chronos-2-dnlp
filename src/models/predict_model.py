# =========================
# FILE: src/models/predict_model.py
# =========================
import os
import pandas as pd
from chronos import Chronos2Pipeline


def load_model(model_name="amazon/chronos-2"):
    print(f"[INFO] Loading Chronos-2 model: {model_name}")
    return Chronos2Pipeline.from_pretrained(model_name)


def predict_df_univariate(pipeline, context_df, horizon=30):
    return pipeline.predict_df(
        context_df,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )


def predict_df_covariates(pipeline, context_df, future_df, horizon=30):
    return pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )


def pred_df_to_quantiles(pred_df):
    # Support both naming conventions
    # quickstart returns columns like "0.1", "0.5", "0.9" + "predictions"
    if "0.1" in pred_df.columns and "0.9" in pred_df.columns:
        p10 = pred_df["0.1"].values
        p90 = pred_df["0.9"].values
        if "predictions" in pred_df.columns:
            med = pred_df["predictions"].values
        elif "0.5" in pred_df.columns:
            med = pred_df["0.5"].values
        else:
            raise ValueError("Cannot find median column in prediction df")
        return p10, med, p90

    # fallback (should not happen with Chronos2Pipeline.predict_df)
    raise ValueError(f"Unexpected prediction df columns: {list(pred_df.columns)}")


def save_quantiles_csv(pred_df, out_path, verbose=True):
    p10, med, p90 = pred_df_to_quantiles(pred_df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame({"p10": p10, "median": med, "p90": p90}).to_csv(out_path, index=False)
    if verbose:
        print(f"[INFO] Saved forecast: {out_path}")
