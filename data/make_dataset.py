import pandas as pd
import os

def load_raw_data(train_path: str, store_path: str, store_id: int | None = None) -> pd.DataFrame:
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path, low_memory=False)

    df = train.merge(store, on="Store", how="left")

    if store_id is not None:
        df = df[df["Store"] == store_id].copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame, keep_closed_days: bool = True) -> pd.DataFrame:
    df = df.copy()

    if not keep_closed_days and "Open" in df.columns:
        df = df[df["Open"] == 1].copy()

    target_col = "Sales" if "Sales" in df.columns else None
    target_series = df[target_col] if target_col else None

    static_cols = {
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
    }
    fill_cols = [c for c in df.columns if c in static_cols]

    if fill_cols:
        df[fill_cols] = (
            df.groupby("Store")[fill_cols].ffill().bfill().infer_objects(copy=False)
        )

    if target_col:
        df[target_col] = target_series

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

    return df


def enforce_daily_frequency_store(df_store: pd.DataFrame, date_col: str = "Date", store_col: str = "Store") -> pd.DataFrame:
    g = df_store.copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    full_idx = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
    g = g.set_index(date_col).reindex(full_idx)
    g.index.name = date_col
    g = g.reset_index()

    g[store_col] = df_store[store_col].iloc[0]
    return g


def enforce_daily_frequency_all_stores(df: pd.DataFrame, store_col: str = "Store", date_col: str = "Date") -> pd.DataFrame:
    return pd.concat(
        [enforce_daily_frequency_store(g, date_col=date_col, store_col=store_col) for _, g in df.groupby(store_col)],
        ignore_index=True,
    )


def to_chronos_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"Store": "id", "Date": "timestamp", "Sales": "target"})
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return out

def select_important_features(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "id",
        "timestamp",
        "target",
        "Customers", # past-only
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "DayOfWeek",
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[keep_cols].copy()

def fix_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target_cols = {"Sales", "target"}

    # deterministic mapping for StateHoliday
    if "StateHoliday" in df.columns:
        mapping = {"0": 0, "a": 1, "b": 2, "c": 3}
        df["StateHoliday"] = (
            df["StateHoliday"].astype(str).str.lower().map(mapping).fillna(0).astype(int)
        )

    for col in df.columns:
        if df[col].dtype == "object":
            if col == "StateHoliday":
                continue
            df[col] = df[col].astype("category").cat.codes

    # Fill NaNs for numeric covariates except Open
    for col in df.columns:
        if col in target_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if col == "Open":
                continue
            df[col] = df[col].fillna(0)

    return df


def save_processed(df, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def save_processed_per_store(df_chronos, out_dir: str, id_col="id"):
    os.makedirs(out_dir, exist_ok=True)
    for sid, g in df_chronos.groupby(id_col):
        out_path = os.path.join(out_dir, f"processed_store_{int(sid)}.csv")
        g.sort_values("timestamp").to_csv(out_path, index=False)
        
def temporal_split(df: pd.DataFrame, test_size: int = 30):
    g = df.copy()

    if "timestamp" in g.columns:
        g["timestamp"] = pd.to_datetime(g["timestamp"])
        g = g.sort_values("timestamp").reset_index(drop=True)
    elif "Date" in g.columns:
        g["Date"] = pd.to_datetime(g["Date"])
        g = g.sort_values("Date").reset_index(drop=True)

    if len(g) <= test_size:
        raise ValueError("Dataset too small for temporal split")

    df_past = g.iloc[:-test_size].reset_index(drop=True)
    df_test = g.iloc[-test_size:].reset_index(drop=True)
    return df_past, df_test

