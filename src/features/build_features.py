import numpy as np
import pandas as pd


def extract_target(df, target_col="Sales"):
    """Extract target series as float32 array."""
    return df[target_col].astype(float).values.astype(np.float32)


def extract_covariates(df, target_col="Sales"):
    """Extract all non-target numerical features as covariates."""
    
    # Remove target and date (non-feature)
    df = df.drop(columns=[target_col, "Date"], errors="ignore")

    # Ensure every column is numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Enforce consistent column ordering
    df = df.sort_index(axis=1)

    return df.values.astype(np.float32)
