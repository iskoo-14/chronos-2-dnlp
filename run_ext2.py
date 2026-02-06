import pandas as pd
from pathlib import Path
import numpy as np
from data.make_dataset import aggregate_shop
from clustering.cluster_shops import run_kmeans_clustering, run_gmm_clustering
from visualization.plots import plot_cluster_profiles_zscore
import matplotlib.pyplot as plt

# Paths
INPUT_DIR = Path("data") / "extension1"
OUTPUT_DIR = Path("data") / "extension2"
OUTPUT_FILE = OUTPUT_DIR / "shop_features_for_clustering.csv"
CLUSTERED_CSV = OUTPUT_DIR / "shop_features_with_clusters.csv"


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

#2 - Running clusterization algorithms
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


