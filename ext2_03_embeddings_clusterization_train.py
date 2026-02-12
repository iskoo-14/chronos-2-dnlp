import os
import joblib
import umap
import hdbscan
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from ext2_02_embeddings_clusterization_anlysis import soft_assign_outliers

MODEL_DIR = "models/umap_hdbscan"
TRAIN_DATA = "data/extension3/shop_embeddings/train.csv"
OUT_TRAIN_FINAL = "data/extension3/shop_embeddings/train_with_clusters.csv"

# Soft-assign settings (mora biti isto na testu)
Q = 0.95
SLACK = 1.2

# -----------------------------
# 1) Load TRAIN
# -----------------------------
df_train = pd.read_csv(TRAIN_DATA)
shop_ids = df_train["shop_id"].values
X_train = df_train.drop(columns=["shop_id"]).values
print("Train shape:", df_train.shape)

# -----------------------------
# 2) Normalize
# -----------------------------
X_train_norm = normalize(X_train, norm="l2")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 3) Fit UMAP
# -----------------------------
umap_model = umap.UMAP(
    n_neighbors=20,
    min_dist=0.0,
    n_components=20,
    metric="cosine",
    random_state=0
)
Z_train = umap_model.fit_transform(X_train_norm)
joblib.dump(umap_model, os.path.join(MODEL_DIR, "umap_model.joblib"))
print("Saved UMAP model.")

# -----------------------------
# 4) Fit HDBSCAN (core)
# -----------------------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)
core_labels = clusterer.fit_predict(Z_train)
joblib.dump(clusterer, os.path.join(MODEL_DIR, "hdbscan_model.joblib"))
print("Saved HDBSCAN model.")

print("Train core outliers (-1):", int(np.sum(core_labels == -1)))
print("Train core clusters:", len([c for c in np.unique(core_labels) if c != -1]))

# -----------------------------
# 5) SOFT/FORCED ASSIGN ON TRAIN  (OVO MORA PRVO)
#    final_labels = finalni klasteri bez -1
# -----------------------------
final_labels, _, _, _, _ = soft_assign_outliers(
    Z=Z_train,
    labels=core_labels,
    probs=None,
    q=Q,
    slack=SLACK
)

print("Train final outliers (-1):", int(np.sum(final_labels == -1)))
print("Train final clusters:", len(np.unique(final_labels)))

# -----------------------------
# 6) Build assignment_state from TRAIN FINAL clusters (centroids + thresholds)
#    (dakle, koristi final_labels, NE core_labels)
# -----------------------------
clusters = sorted(np.unique(final_labels))  # nema -1 ako je forced
centroids = {}
thresholds = {}

for c in clusters:
    idx = np.where(final_labels == c)[0]
    Xc = Z_train[idx]
    mu = Xc.mean(axis=0)
    centroids[int(c)] = mu

    d = np.linalg.norm(Xc - mu, axis=1)
    base = np.percentile(d, Q * 100) if len(d) >= 5 else float(np.max(d))
    thresholds[int(c)] = float(base * SLACK)

assignment_state = {
    "q": Q,
    "slack": SLACK,
    "clusters": [int(c) for c in clusters],
    "centroids": centroids,
    "thresholds": thresholds
}

joblib.dump(assignment_state, os.path.join(MODEL_DIR, "assignment_state.joblib"))
print("Saved assignment_state.joblib (based on TRAIN FINAL clusters).")

# -----------------------------
# 7) Save TRAIN output: ONLY shop_id + final_cluster
# -----------------------------

# -----------------------------
# 7) Save TRAIN output: FULL train + final_cluster column

# -----------------------------
out_df = df_train.copy()
out_df["cluster"] = final_labels

os.makedirs(os.path.dirname(OUT_TRAIN_FINAL), exist_ok=True)
out_df.to_csv(OUT_TRAIN_FINAL, index=False)
print("Saved:", OUT_TRAIN_FINAL)

print("\nAll saved in:", MODEL_DIR)

