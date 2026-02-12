# ext2_eval_delta_wql.py
from pathlib import Path
import pandas as pd

from visualization.plots import plot_delta_wql_by_cluster


WQL_BASE_FILE = Path("reports") / "baseline" / "wql_per_store.csv"
WQL_EXT_FILE  = Path("reports") / "extension1" / "wql_per_store.csv"
MAP_FILE      = Path("data") / "extension2" / "shop_cluster_mapping.csv"

OUT_TABLE     = Path("reports") / "extension2" / "wql_delta_by_cluster.csv"
FIG_DIR       = Path("outputs") / "figures"
CONTEXT_LENGTH = 512


def main():
    base_df = pd.read_csv(WQL_BASE_FILE)
    new_df  = pd.read_csv(WQL_EXT_FILE)
    map_df  = pd.read_csv(MAP_FILE)

    # Ensure same dtype for join key
    for df in (base_df, new_df, map_df):
        df["store_id"] = df["store_id"].astype(int)

    # focus on one context length
    base_df = base_df[base_df["context_length"] == CONTEXT_LENGTH].copy()
    new_df  = new_df[new_df["context_length"] == CONTEXT_LENGTH].copy()

    # merge baseline vs new + cluster
    df = (
        base_df[["store_id", "mode", "wql"]]
        .rename(columns={"wql": "wql_base"})
        .merge(
            new_df[["store_id", "mode", "wql"]].rename(columns={"wql": "wql_new"}),
            on=["store_id", "mode"],
            how="inner",
        )
        .merge(
            map_df[["store_id", "cluster"]],
            on="store_id",
            how="inner",
        )
    )

    # delta (negative = improvement)
    df["delta_wql"] = df["wql_new"] - df["wql_base"]

    summary = (
        df.groupby(["cluster", "mode"])["delta_wql"]
          .agg(
              count="count",
              mean="mean",
              median="median",
              std="std",
              p25=lambda s: s.quantile(0.25),
              p75=lambda s: s.quantile(0.75),
              p90=lambda s: s.quantile(0.90),
          )
          .reset_index()
          .round(6)
          .sort_values(["mode", "cluster"])
    )

    print("\n=== ΔWQL by cluster (new − baseline) ===")
    print(summary)

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLE, index=False)
    print(f"[OK] Saved table: {OUT_TABLE}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_delta_wql_by_cluster(
        df,
        png_path=str(FIG_DIR / "delta_wql_by_cluster.png"),
    )
    print(f"[OK] Saved plot: {FIG_DIR / 'delta_wql_by_cluster.png'}")


if __name__ == "__main__":
    main()
