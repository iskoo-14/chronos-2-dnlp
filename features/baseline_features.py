import pandas as pd

# baseline set (like in the paper)
PAST_ONLY_COVS = ["Customers"]
KNOWN_FUTURE_COVS = ["Open", "Promo", "StateHoliday", "SchoolHoliday", "DayOfWeek"]

BASELINE_KEEP_COLS = ["id", "timestamp", "target"] + PAST_ONLY_COVS + KNOWN_FUTURE_COVS


def select_baseline_features(df_chronos: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BASELINE_KEEP_COLS if c not in df_chronos.columns]
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")
    return df_chronos[BASELINE_KEEP_COLS].copy()
