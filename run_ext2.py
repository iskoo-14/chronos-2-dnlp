import pandas as pd
from pathlib import Path
import numpy as np
from data.make_dataset import aggregate_shop
from clustering.cluster_shops import run_kmeans_clustering, run_gmm_clustering
from visualization.plots import plot_cluster_profiles_zscore, plot_delta_wql_by_cluster
import matplotlib.pyplot as plt
import seaborn as sns


# ROOT PATHS
INPUT_DIR = Path("data") / "extension1"
OUTPUT_DIR = Path("data") / "extension2"

OUTPUT_FILE = OUTPUT_DIR / "shop_features_for_clustering.csv" #DATASET WHERE DATA ARE AGGREGATED AND ON WHICH WE PERFORM CLUSTERING
CLUSTERED_CSV = OUTPUT_DIR / "shop_features_with_clusters.csv" #CLUSTERED DATASET

#WQL RESULTS FOR BASELINE AND EXTENDED VERSIONS
WQL_BASE_FILE = Path("reports") / "baseline" / "wql_per_store.csv"      # baseline
WQL_EXT_FILE  = Path("reports") / "extension1" / "wql_per_store.csv"      # with new features
MAP_FILE      = Path("data") / "extension2" / "shop_cluster_mapping.csv"

#EVALUATIONS
OUT_TABLE     = Path("reports") / "extension2" / "wql_delta_by_cluster.csv"
OUT_PLOT      = Path("outputs") / "figures" / "delta_wql_by_cluster.pdf"

CONTEXT_LENGTH = 512

#1 - Creation of dataset for clusterization 
rows = []
for csv_file in INPUT_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)

    shop_id = df["id"].iloc[0]
    shop_features = aggregate_shop(df, shop_id)
    rows.append(shop_features)

cluster_df = (
    pd.DataFrame(rows)
    .sort_values("shop_id")
    .reset_index(drop=True)
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cluster_df.to_csv(OUTPUT_FILE, index=False)

#2 - Running clusterization algorithm
df_with_clusters, k_summary, cluster_sil = run_kmeans_clustering(
    features_csv=Path("data/extension2/shop_features_for_clustering.csv"),
    output_csv=Path("data/extension2/km_clusters.csv"),
    k_min=2,
    k_max=8
)

## Exporting k_summary, and cluster_silhouettes scores
k_summary.to_csv(
    "reports/extension2/k_summary.csv",
    index=False
)
cluster_sil.to_csv(
    "reports/extension2/cluster_sil.csv",
    index=False
)

#3 - Cluster interpretation
key_features = [
    "mean_target",
    "cv_target",
    "mean_abs_chg",
    "spike_rate",
    "corr_lag52",
    "weekly_strength",
    "yearly_strength",
    "promo_lift",
    "closed_rate",
]

profile, profile_z = plot_cluster_profiles_zscore(
    df_with_clusters=df_with_clusters,
    key_features=key_features,
    fig_path="outputs/figures/cluster_profiles_kmeans.png"
)

#4 - Exporting shop-cluster mapping
shop_cluster_map = (
    df_with_clusters[["shop_id", "cluster"]]
    .rename(columns={"shop_id": "store_id"})
    .sort_values("store_id")
)

shop_cluster_map.to_csv(
    "data/extension2/shop_cluster_mapping.csv",
    index=False
)

#5 - Evaluating results 

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
        new_df[["store_id", "mode", "wql"]]
        .rename(columns={"wql": "wql_new"}),
        on=["store_id", "mode"],
        how="inner"
    )
    .merge(
        map_df[["store_id", "cluster"]],
        on="store_id",
        how="inner"
    )
)

# 3) DELTA WQL  (negative = improvement)
df["delta_wql"] = df["wql_new"] - df["wql_base"]

# 4) AGGREGATE BY CLUSTER & MODE
summary = (
    df
    .groupby(["cluster", "mode"])["delta_wql"]
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

# 
OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_TABLE, index=False)
print(f"[INFO] Saved table: {OUT_TABLE}")

# 6 Visualization
plot_delta_wql_by_cluster(
    df,
    png_path="outputs/figures/delta_wql_by_cluster.png"
)

