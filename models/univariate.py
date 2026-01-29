import os
import pandas as pd


def predict_df_univariate(pipeline, context_df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    return pipeline.predict_df(
        context_df,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )


def pred_df_to_quantiles(pred_df: pd.DataFrame):
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

    # alternative style
    if "p10" in pred_df.columns and "p90" in pred_df.columns:
        p10 = pred_df["p10"].values
        p90 = pred_df["p90"].values
        if "median" in pred_df.columns:
            med = pred_df["median"].values
        elif "p50" in pred_df.columns:
            med = pred_df["p50"].values
        else:
            raise ValueError("Cannot find median column in prediction df")
        return p10, med, p90

    raise ValueError(f"Unexpected prediction df columns: {list(pred_df.columns)}")


def save_quantiles_csv(pred_df: pd.DataFrame, out_path: str, verbose: bool = True) -> None:
    p10, med, p90 = pred_df_to_quantiles(pred_df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cols = {"p10": p10, "median": med, "p90": p90}
    if "timestamp" in pred_df.columns:
        cols = {"timestamp": pd.to_datetime(pred_df["timestamp"]).values, **cols}

    pd.DataFrame(cols).to_csv(out_path, index=False)
    if verbose:
        print(f"[INFO] Saved forecast: {out_path}")
