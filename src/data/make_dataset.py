import pandas as pd
import numpy as np
import os

def temporal_split(df, test_size=30):
    if len(df) <= test_size:
        raise ValueError("Dataset too small for temporal split")

    df_past = df.iloc[:-test_size].reset_index(drop=True)
    df_test = df.iloc[-test_size:].reset_index(drop=True)
    return df_past, df_test

def load_raw_data(train_path, store_path):
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path, low_memory=False)
    return train.merge(store, on="Store", how="left")


def clean_data(df):
    df = df[df["Open"] == 1]
    df = df.dropna(subset=["Sales"])
    
    # Modern syntax (no FutureWarning)
    df = df.ffill().bfill()
    return df


def add_time_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df


def fix_mixed_types(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    return df.fillna(0)


def save_processed(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(base, "train.csv")
    store_path = os.path.join(base, "store.csv")
    out_path = os.path.join(base, "processed_rossmann.csv")

    print("Loading raw data...")
    df = load_raw_data(train_path, store_path)

    print("Cleaning...")
    df = clean_data(df)

    print("Adding time features...")
    df = add_time_features(df)

    print("Fixing mixed types...")
    df = fix_mixed_types(df)

    print("Saving...")
    save_processed(df, out_path)
