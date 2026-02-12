# ext2_data_prep.py
import argparse
from pathlib import Path
import pandas as pd
from data.make_dataset import aggregate_shop
from clustering_utilities import run_kmeans_clustering
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization.plots import plot_cluster_profiles_zscore

# ROOT PATHS
INPUT_DIR  = Path("data") / "extension1"
OUTPUT_DIR = Path("data") / "extension1"
REPORT_DIR = Path("reports") / "extension1"
FIG_DIR    = Path("outputs") / "figures"

OUTPUT_FILE = OUTPUT_DIR / "shop_features_for_clustering.csv"
KM_CSV      = OUTPUT_DIR / "km_clusters.csv"
MAP_FILE    = OUTPUT_DIR / "shop_cluster_mapping.csv"


def build_clustering_dataset(input_dir: Path, output_csv: Path) -> pd.DataFrame:
    rows = []
    for csv_file in input_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        shop_id = int(df["id"].iloc[0])
        shop_features = aggregate_shop(df, shop_id)
        rows.append(shop_features)

    cluster_df = (
        pd.DataFrame(rows)
        .sort_values("shop_id")
        .reset_index(drop=True)
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(output_csv, index=False)
    print(f"[OK] Saved clustering dataset: {output_csv}")
    return cluster_df


def run_clustering(features_csv: Path, algo: str, k_min: int, k_max: int) -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if algo == "kmeans":
        df_with_clusters, k_summary, cluster_sil = run_kmeans_clustering(
            features_csv=features_csv,
            output_csv=KM_CSV,
            k_min=k_min,
            k_max=k_max
        )
        k_summary.to_csv(REPORT_DIR / "k_summary.csv", index=False)
        cluster_sil.to_csv(REPORT_DIR / "cluster_sil.csv", index=False)
        print(f"[OK] Saved: {REPORT_DIR / 'k_summary.csv'}")
        print(f"[OK] Saved: {REPORT_DIR / 'cluster_sil.csv'}")
        print(f"[OK] Saved clustered dataset: {KM_CSV}")
        return df_with_clusters
    else:
        raise ValueError("algo must be 'kmeans' ")


def export_shop_cluster_mapping(df_with_clusters: pd.DataFrame, out_csv: Path):
    shop_cluster_map = (
        df_with_clusters[["shop_id", "cluster"]]
        .rename(columns={"shop_id": "store_id"})
        .sort_values("store_id")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shop_cluster_map.to_csv(out_csv, index=False)
    print(f"[OK] Saved mapping: {out_csv}")


def plot_cluster_profiles(df_with_clusters: pd.DataFrame, fig_path: Path):
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
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_cluster_profiles_zscore(
        df_with_clusters=df_with_clusters,
        key_features=key_features,
        fig_path=str(fig_path),
    )
    print(f"[OK] Saved profile plot: {fig_path}")


def main(args):
    # 1) build dataset
    build_clustering_dataset(INPUT_DIR, OUTPUT_FILE)

    # 2) clustering
    df_with_clusters = run_clustering(
        features_csv=OUTPUT_FILE,
        algo=args.algo,
        k_min=args.k_min,
        k_max=args.k_max,
    )

    # 3) profile plot
    plot_cluster_profiles(df_with_clusters, FIG_DIR / f"cluster_profiles_{args.algo}.png")

    # 4) mapping export
    export_shop_cluster_mapping(df_with_clusters, MAP_FILE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, default="kmeans", choices=["kmeans", "gmm"])
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=8)
    args = ap.parse_args()
    main(args)
