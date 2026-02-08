import pandas as pd

# 1) Učitaj embeddinge shopova
df_emb = pd.read_csv("data/extension3/shop_embeddings_chronos2.csv")
# očekuje: shop_id, e0, e1, ..., eN

# 2) Učitaj klastere (soft-assigned)
df_clusters = pd.read_csv("data/extension3/shop_clusters_umap_hdbscan_softassign.csv")
# očekuje: shop_id, final_cluster, ...

# 3) Uzmi samo potrebne kolone i preimenuj
df_clusters = df_clusters[["shop_id", "final_cluster"]].rename(
    columns={"final_cluster": "cluster"}
)

# 4) Join po shop_id
df_out = df_emb.merge(
    df_clusters,
    on="shop_id",
    how="inner"   # inner = samo shopovi koji imaju klaster
)

# 5) Snimi novi CSV
out_path = "data/extension3/shop_embeddings_with_cluster.csv"
df_out.to_csv(out_path, index=False)

print("Saved:", out_path)
print(df_out.head())



import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

# 1) Učitaj shop embeddinge sa klasterom
df = pd.read_csv("data/extension3/shop_embeddings_with_cluster.csv")
# očekuje: shop_id | e0 | e1 | ... | eN | cluster

# 2) Izdvoji embedding kolone
embedding_cols = [c for c in df.columns if c not in ["shop_id", "cluster"]]

# 3) Group by cluster + mean
df_cluster_emb = (
    df
    .groupby("cluster")[embedding_cols]
    .mean()
    .reset_index()
)

# 4) (PREPORUČENO) L2 normalizacija klaster embeddinga
X = df_cluster_emb[embedding_cols].values
X = normalize(X, norm="l2")
df_cluster_emb[embedding_cols] = X

# 5) Snimi
out_path = "data/extension3/cluster_embeddings_mean.csv"
df_cluster_emb.to_csv(out_path, index=False)

print("Saved:", out_path)
print(df_cluster_emb.head())
