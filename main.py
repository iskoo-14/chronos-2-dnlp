# =========================
# FILE: main.py
# =========================
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd

from src.data.make_dataset import (
    load_raw_data,
    clean_data,
    add_time_features,
    select_important_features,
    fix_mixed_types,
    to_chronos_df,
    temporal_split,
    save_processed,
)
from src.models.predict_model import (
    load_model,
    predict_df_univariate,
    predict_df_covariates,
    save_quantiles_csv,
)


def ensure_output_folder(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    print("===================================================")
    print("=== DNLP PIPELINE STARTED (Chronos quickstart aligned) ===")
    print("===================================================")

    base = os.path.dirname(os.path.abspath(__file__))
    output_dir = ensure_output_folder(os.path.join(base, "outputs"))

    train_path = os.path.join(base, "src/data/train.csv")
    store_path = os.path.join(base, "src/data/store.csv")

    STORE_ID = 1
    HORIZON = 30
    CONTEXT_LEN = 256

    print("[STEP] Loading raw data (single store)...")
    df = load_raw_data(train_path, store_path, store_id=STORE_ID)

    print("[STEP] Cleaning (keep closed days)...")
    df = clean_data(df, keep_closed_days=True)

    print("[STEP] Adding time features...")
    df = add_time_features(df)

    print("[STEP] Selecting paper-style covariates...")
    df = select_important_features(df)

    print("[STEP] Fixing mixed types...")
    df = fix_mixed_types(df)

    print("[STEP] Converting to Chronos dataframe format...")
    df = to_chronos_df(df)

    processed_path = os.path.join(base, "src/data/processed_rossmann_single.csv")
    save_processed(df, processed_path)

    print("[STEP] Temporal split: last 30 days as ground truth...")
    df_past, df_test = temporal_split(df, test_size=HORIZON)

    print(f"[INFO] Past length: {len(df_past)} | Test length: {len(df_test)}")

    # keep last CONTEXT_LEN as context
    if len(df_past) > CONTEXT_LEN:
        df_past = df_past.iloc[-CONTEXT_LEN:].reset_index(drop=True)
        print(f"[INFO] Context truncated to last {CONTEXT_LEN} rows")

    gt_path = os.path.join(output_dir, "ground_truth.csv")
    pd.DataFrame({"y_true": df_test["target"].values}).to_csv(gt_path, index=False)
    print(f"[INFO] Saved ground truth: {gt_path}")

    print("[STEP] Loading Chronos-2 model...")
    pipeline = load_model("amazon/chronos-2")

    # covariates sets (paper-style)
    PAST_ONLY_COVS = ["Customers"]
    KNOWN_FUTURE_COVS = ["Open", "Promo", "SchoolHoliday", "StateHoliday", "DayOfWeek"]

    # ----------------------------
    # UNIVARIATE
    # ----------------------------
    print("[STEP] Running univariate forecast...")
    context_uni = df_past[["id", "timestamp", "target"]].copy()
    pred_uni = predict_df_univariate(pipeline, context_uni, horizon=HORIZON)
    save_quantiles_csv(pred_uni, os.path.join(output_dir, "univariate.csv"))

    # ----------------------------
    # COVARIATE (ICL) FORECAST
    # ----------------------------
    print("[STEP] Running covariate forecast (predict_df)...")
    context_cov = df_past[["id", "timestamp", "target"] + PAST_ONLY_COVS + KNOWN_FUTURE_COVS].copy()
    future_cov = df_test[["id", "timestamp"] + KNOWN_FUTURE_COVS].copy()

    pred_cov = predict_df_covariates(pipeline, context_cov, future_cov, horizon=HORIZON)
    save_quantiles_csv(pred_cov, os.path.join(output_dir, "covariate.csv"))

    print("===================================================")
    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("===================================================")
