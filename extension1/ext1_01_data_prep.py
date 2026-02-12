# ext1_data_prep.py
import os
import argparse
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import HORIZON, SAVE_FUTURE_DEBUG, SKIP_EXISTING_GT_DEBUG
from data.make_dataset import temporal_split
from extension1.feature_utilities import build_extension1_files, extension1_covariate_sets


EXPERIMENT = "extension1"
CTX_LEN = 512

PROCESSED_DIR_BASELINE = os.path.join("data", "processed")
PROCESSED_DIR_EXT1     = os.path.join("data", "extension1")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_global_dirs(project_root: str) -> dict:
    base = ensure_dir(os.path.join(project_root, "outputs", EXPERIMENT))
    return {
        "gt_dir": ensure_dir(os.path.join(base, "ground_truth")),
        "dbg_dir": ensure_dir(os.path.join(base, "debug")),
    }


def read_valid_store_ids(path: str = os.path.join("reports", "valid_store_ids.txt")) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run preprocessing first.")
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(int(s))
    return ids


def read_processed_store(store_id: int, processed_dir: str) -> pd.DataFrame:
    p = os.path.join(processed_dir, f"processed_store_{store_id}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing processed file: {p}")
    df = pd.read_csv(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns and ("DayOfWeek" not in df.columns or df["DayOfWeek"].isna().any()):
        df = df.copy()
        df["DayOfWeek"] = pd.to_datetime(df["timestamp"]).dt.dayofweek + 1
    return df


def build_future_debug(df_test: pd.DataFrame) -> pd.DataFrame:
    dbg_cols = [c for c in [
        "timestamp","target","Open","Promo","SchoolHoliday","StateHoliday","DayOfWeek","Customers",
        "week_sin","week_cos","month_sin","month_cos","quarter"
    ] if c in df_test.columns]
    dbg = df_test[dbg_cols].copy()
    if "target" in dbg.columns:
        dbg = dbg.rename(columns={"target": "y_true"})
    return dbg


def main(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    store_ids = read_valid_store_ids()

    # 1) build extension1 processed files
    build_extension1_files(
        processed_dir=PROCESSED_DIR_BASELINE,
        output_dir=PROCESSED_DIR_EXT1,
        store_ids=store_ids,
        include_ema=args.include_ema,
        include_chg=args.include_chg,
        include_rolling=args.include_rolling,
    )

    # 2) cov sets (da znaš šta treba da postoji; ovde možeš samo da proveriš)
    PAST_ONLY_COVS, FUTURE_KNOWN_COVS = extension1_covariate_sets(
        include_ema=args.include_ema,
        include_chg=args.include_chg,
        include_rolling=args.include_rolling,
    )

    # 3) write GT/debug
    global_dirs = make_global_dirs(project_root)
    gt_dir, dbg_dir = global_dirs["gt_dir"], global_dirs["dbg_dir"]

    for sid in store_ids:
        df = read_processed_store(sid, processed_dir=PROCESSED_DIR_EXT1)
        df = ensure_dayofweek(df)

        df_past, df_test = temporal_split(df, test_size=HORIZON)
        if len(df_past) > CTX_LEN:
            df_past = df_past.iloc[-CTX_LEN:].reset_index(drop=True)

        df_test = ensure_dayofweek(df_test)

        gt_path = os.path.join(gt_dir, f"ground_truth_store_{sid}.csv")
        if (not SKIP_EXISTING_GT_DEBUG) or (not os.path.exists(gt_path)):
            out = {"timestamp": df_test["timestamp"], "y_true": df_test["target"]}
            if "Open" in df_test.columns:
                out["Open"] = df_test["Open"]
            pd.DataFrame(out).to_csv(gt_path, index=False)

        if SAVE_FUTURE_DEBUG:
            dbg_path = os.path.join(dbg_dir, f"future_debug_store_{sid}.csv")
            if (not SKIP_EXISTING_GT_DEBUG) or (not os.path.exists(dbg_path)):
                build_future_debug(df_test).to_csv(dbg_path, index=False)

        # opciono: validacija da covs postoje
        if args.check_covs:
            need_ctx = ["id","timestamp","target"] + PAST_ONLY_COVS + FUTURE_KNOWN_COVS
            need_fut = ["id","timestamp"] + FUTURE_KNOWN_COVS
            miss_ctx = [c for c in need_ctx if c not in df_past.columns]
            miss_fut = [c for c in need_fut if c not in df_test.columns]
            if miss_ctx or miss_fut:
                print(f"[WARN] Store {sid}: missing ctx={miss_ctx} fut={miss_fut}")

    print("[OK] Data prep finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_ema", action="store_true")
    ap.add_argument("--include_chg", action="store_true")
    ap.add_argument("--include_rolling", action="store_true")
    ap.add_argument("--check_covs", action="store_true")
    args = ap.parse_args()

    main(args)
