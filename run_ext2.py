import pandas as pd
from pathlib import Path

from data.make_dataset import aggregate_shop


# Paths
INPUT_DIR = Path("data") / "extension1"
OUTPUT_DIR = Path("data") / "extension2"
OUTPUT_FILE = OUTPUT_DIR / "shop_features_for_clustering.csv"

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

#2
