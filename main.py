import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd

from src.data.make_dataset import (
    load_raw_data, clean_data, add_time_features, fix_mixed_types, save_processed, temporal_split
)
from src.features.build_features import extract_target, extract_covariates
from src.models.predict_model import (
    load_model, predict_univariate, predict_covariates
)


def ensure_output_folder(path="outputs"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == "__main__":
    print("=== DNLP PIPELINE STARTED ===")

    base = os.path.dirname(os.path.abspath(__file__))
    output_dir = ensure_output_folder("outputs")

    # ------------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------------
    print("Loading raw data...")
    train_path = os.path.join(base, "src/data/train.csv")
    store_path = os.path.join(base, "src/data/store.csv")

    df = load_raw_data(train_path, store_path)

    print("Cleaning data...")
    df = clean_data(df)
    df = add_time_features(df)
    df = fix_mixed_types(df)

    processed_path = os.path.join(base, "src/data/processed_rossmann.csv")
    save_processed(df, processed_path)
    # ------------------------------------------------------------
    # TEMPORAL SPLIT (ZERO-SHOT EVALUATION)
    # ------------------------------------------------------------
    print("Applying temporal split...")
    df_past, df_test = temporal_split(df, test_size=30)
    # ------------------------------------------------------------
    # SAVE GROUND TRUTH (FOR WQL EVALUATION)
    # ------------------------------------------------------------
    y_true = df_test["Sales"].values

    pd.DataFrame(
        {"y_true": y_true}
    ).to_csv(os.path.join(output_dir, "ground_truth.csv"), index=False)

    print("Saved ground_truth.csv")

    # ------------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------------
    print("Extracting features (past only)...")
    target_past = extract_target(df_past)
    covariates_past = extract_covariates(df_past)


    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------
    print("Loading Chronos-2 model...")
    model = load_model("amazon/chronos-2")

    # ------------------------------------------------------------
    # UNIVARIATE FORECAST
    # ------------------------------------------------------------
    print("Running univariate forecast...")
    uni_out = predict_univariate(model, target_past)

    pd.DataFrame({
        "p10": uni_out["p10"],
        "median": uni_out["median"],
        "p90": uni_out["p90"]
    }).to_csv(os.path.join(output_dir, "univariate.csv"), index=False)
    print("Saved univariate.csv")

    # ------------------------------------------------------------
    # COVARIATE FORECAST
    # ------------------------------------------------------------
    print("Running covariate forecast...")
    cov_out = predict_covariates(model, target_past, covariates_past, covariates_past)

    pd.DataFrame({
        "p10": cov_out["p10"],
        "median": cov_out["median"],
        "p90": cov_out["p90"]
    }).to_csv(os.path.join(output_dir, "covariate.csv"), index=False)
    print("Saved covariate.csv")

    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
