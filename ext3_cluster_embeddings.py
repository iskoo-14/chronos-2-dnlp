import os
import pandas as pd
from sklearn.preprocessing import normalize

TRAIN_WITH_CLUSTERS = "data/extension3/shop_embeddings/train_with_clusters.csv"
OUT_DIR = "data/extension3/cluster_embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Load train_with_clusters.csv
# -----------------------------
df = pd.read_csv(TRAIN_WITH_CLUSTERS)
# expected: shop_id, <embedding cols...>, cluster

# -----------------------------
# 2) Identify embedding columns
# -----------------------------
embedding_cols = [c for c in df.columns if c not in ["shop_id", "cluster"]]

# -----------------------------
# 3) Group by cluster + mean
# -----------------------------
df_cluster_emb = (
    df
    .groupby("cluster")[embedding_cols]
    .mean()
    .reset_index()
)

# -----------------------------
# 4) L2 normalize cluster embeddings
# -----------------------------
X = normalize(df_cluster_emb[embedding_cols].values, norm="l2")
df_cluster_emb[embedding_cols] = X

# -----------------------------
# 5) Save
# -----------------------------
out_cluster_path = os.path.join(OUT_DIR, "cluster_embeddings_mean.csv")
df_cluster_emb.to_csv(out_cluster_path, index=False)

print("Saved:", out_cluster_path)
print(df_cluster_emb.head())
