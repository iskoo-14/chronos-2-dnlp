import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from hdbscan.prediction import approximate_predict

MODEL_DIR = "models/umap_hdbscan"
TEST_DATA = "data/extension2/shop_embeddings/test.csv"
OUT_TEST_FINAL = "data/extension2/shop_embeddings/test_with_clusters.csv"

def assign_outliers_using_train_state(Z, labels, state):
    
    """
    Forced assignment: only labels equal to -1 are assigned to the nearest TRAIN centroid.
    Centroids/thresholds come from the final TRAIN clusters.
    """
    final_labels = labels.copy()

    clusters = state["clusters"]
    centroids = state["centroids"]

    if len(clusters) == 0:
        return final_labels

    C = np.vstack([centroids[c] for c in clusters])  # (K, dim)

    out_idx = np.where(labels == -1)[0]
    if len(out_idx) == 0:
        return final_labels

    Z_out = Z[out_idx]
    dmat = np.linalg.norm(Z_out[:, None, :] - C[None, :, :], axis=2)  # (M, K)
    best_k = np.argmin(dmat, axis=1)

    for j, i in enumerate(out_idx):
        final_labels[i] = clusters[int(best_k[j])]

    return final_labels


print("===================================================")
print(f"=== CLUSTERING INFERENCE PIPELINE STARTS  ===")
print("===================================================")

# -----------------------------
# 1) Load models + TRAIN assignment_state
# -----------------------------
umap_model = joblib.load(os.path.join(MODEL_DIR, "umap_model.joblib"))
clusterer  = joblib.load(os.path.join(MODEL_DIR, "hdbscan_model.joblib"))
state      = joblib.load(os.path.join(MODEL_DIR, "assignment_state.joblib"))

# -----------------------------
# 2) Load TEST
# -----------------------------
df_test = pd.read_csv(TEST_DATA)
shop_ids_test = df_test["shop_id"].values
X_test = df_test.drop(columns=["shop_id"]).values
print("Test shape:", df_test.shape)

# -----------------------------
# 3) Normalize + UMAP transform
# -----------------------------
X_test_norm = normalize(X_test, norm="l2")
Z_test = umap_model.transform(X_test_norm)

# -----------------------------
# 4) HDBSCAN inference (core labels on test)
# -----------------------------
test_core_labels, _ = approximate_predict(clusterer, Z_test)
print("Test core outliers (-1):", int(np.sum(test_core_labels == -1)))

# -----------------------------
# 5) Assign -1 using TRAIN FINAL centroids
# -----------------------------
test_final_labels = assign_outliers_using_train_state(Z_test, test_core_labels, state)
print("Test final outliers (-1):", int(np.sum(test_final_labels == -1)))

# -----------------------------
# 6) Save ONLY shop_id + final_cluster
# -----------------------------

out_df = df_test.copy()
out_df["cluster"] = test_final_labels

os.makedirs(os.path.dirname(OUT_TEST_FINAL), exist_ok=True)
out_df.to_csv(OUT_TEST_FINAL, index=False)
print("Saved:", OUT_TEST_FINAL)


print("===================================================")
print(f"=== CLUSTERING INFERENCE PIPELINE FINISHED  ===")
print("===================================================")