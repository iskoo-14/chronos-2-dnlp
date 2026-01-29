import pandas as pd


def predict_df_covariates(pipeline, context_df: pd.DataFrame, future_df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    return pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
