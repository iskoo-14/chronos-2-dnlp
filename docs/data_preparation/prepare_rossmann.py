import pandas as pd
import numpy as np
import os

def load_rossmann(train_path: str, store_path: str) -> pd.DataFrame:
    # Load and merge Rossmann train + store CSV files
    train = pd.read_csv(train_path)
    store = pd.read_csv(store_path)
    df = train.merge(store, on="Store", how="left")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove closed days and fill missing values
    df = df[df["Open"] == 1]
    df = df.dropna(subset=["Sales"])
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add temporal covariates
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df

def add_experiment_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add noise feature for robustness experiments
    df["RandomNoise"] = np.random.normal(0, 1, size=len(df))
    return df

def save_processed(df: pd.DataFrame, out_path: str):
    # Save dataset after preprocessing
    df.to_csv(out_path, index=False)
    print(f"Processed dataset saved to: {out_path}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    train_csv = os.path.join(base_path, "../../data/train.csv")
    store_csv = os.path.join(base_path, "../../data/store.csv")
    
    output_csv = os.path.join(base_path, "processed_rossmann.csv")
    
    print("Loading dataset...")
    df = load_rossmann(train_csv, store_csv)

    print("Cleaning dataset...")
    df = clean_data(df)

    print("Adding temporal covariates...")
    df = add_time_features(df)

    print("Adding robustness analysis covariates...")
    df = add_experiment_features(df)

    print("Saving processed dataset...")
    save_processed(df, output_csv)
