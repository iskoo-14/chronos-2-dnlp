import os
import argparse
import pandas as pd
import time, traceback

from config import PROCESSED_DIR, HORIZON
from data.make_dataset import temporal_split
from models.chronos import load_model
from evaluation.robustness import run_all_robustness_tests
from evaluation.io import ensure_dir
from evaluation.select_best_context import select_best_context
from evaluation.robustness import collect_robustness_summary

# helper functions
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_valid_store_ids(path: str = os.path.join("reports", "valid_store_ids.txt")) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run run_preprocessing.py first.")
    ids: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(line))
    return ids


def read_processed_store(store_id: int) -> pd.DataFrame:
    p = os.path.join(PROCESSED_DIR, f"processed_store_{store_id}.csv")
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


if __name__ == "__main__":
    # trying something for the extension, not done yet
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--model_name", type=str, default="amazon/chronos-2")
    parser.add_argument("--skip_existing_robustness", action="store_true")
    args = parser.parse_args()

    forecasts_root = os.path.join("outputs", args.experiment, "forecasts")
    reports_dir = ensure_dir(os.path.join("reports", args.experiment))
    robust_root = ensure_dir(os.path.join("outputs", args.experiment, "robustness"))

    # take BEST context
    best_summary = select_best_context(reports_dir=reports_dir)
    cov_best = [b for b in best_summary if b["mode"] == "covariate"]
    if not cov_best:
        raise SystemExit("[ERROR] No best covariate context found. Run run_evaluations.py first.")
    best_ctx = int(cov_best[0]["best_context"])

    ctx_out = ensure_dir(os.path.join(robust_root, f"ctx_{best_ctx}"))

    print("===================================================")
    print(f"=== ROBUSTNESS — experiment: {args.experiment} ===")
    print("===================================================")
    pipeline = load_model(args.model_name)

    store_ids = read_valid_store_ids()

    for i, sid in enumerate(store_ids, start=1):
        t0 = time.time()
        print(f"[INFO] ({i}/{len(store_ids)}) store {sid} — start", flush=True)
        try:
            df = read_processed_store(sid)
            df = ensure_dayofweek(df)

            df_past, df_test = temporal_split(df, test_size=HORIZON)

            if len(df_past) > best_ctx:
                df_past = df_past.iloc[-best_ctx:].reset_index(drop=True)

            df_for_rob = pd.concat([df_past, df_test], ignore_index=True)

            run_all_robustness_tests(
                pipeline,
                df_for_rob,
                store_id=sid,
                context_len=best_ctx,
                output_root=ctx_out
            )


        except Exception as e:
            print(f"[ERROR] store {sid} failed: {e}", flush=True)
            traceback.print_exc()
            raise


    # create robustness report
    rb_per_store, rb_summary = collect_robustness_summary(
        experiment=args.experiment,
        forecasts_root=forecasts_root,
        reports_dir=reports_dir,
        best_context_length=best_ctx,
    )

    if rb_per_store:
        print(f"[INFO] Robustness merged saved to {rb_per_store}")
    if rb_summary:
        print(f"[INFO] Robustness summary saved to {rb_summary}")

    print("===================================================")
    print("=== ROBUSTNESS DONE ===")
    print("===================================================")