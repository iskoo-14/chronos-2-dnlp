# 
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from collections import Counter
import hdbscan
import matplotlib.pyplot as plt
import umap.umap_ as umap  # pip install umap-learn

# -----------------------------
# Helpers
# -----------------------------
def summarize_clusters(shop_ids, X, labels, probs, persistence=None, top_k=5):
    uniq = [c for c in np.unique(labels) if c != -1]
    print("\n====================")
    print("HDBSCAN ANALYSIS REPORT")
    print("====================")

    n_clusters = len(uniq)
    noise_frac = np.mean(labels == -1)
    print(f"Broj klastera: {n_clusters}")
    print(f"Noise (%): {100*noise_frac:.2f}")

    counts = Counter(labels)
    sizes = {k: v for k, v in counts.items() if k != -1}
    print("\nVeličine klastera (top 10):")
    for k, v in sorted(sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"  Cluster {k:>3}: {v}")

    rows = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        p = probs[idx] if probs is not None else None

        size = len(idx)
        mean_p = float(np.mean(p)) if p is not None else np.nan
        med_p  = float(np.median(p)) if p is not None else np.nan
        min_p  = float(np.min(p)) if p is not None else np.nan

        pers = float(persistence[c]) if persistence is not None and c < len(persistence) else np.nan

        # medoid (najbliža tačka centroidu u prostoru gde klasteruješ)
        Xc = X[idx]
        centroid = Xc.mean(axis=0, keepdims=True)
        d = np.linalg.norm(Xc - centroid, axis=1)
        medoid_idx = idx[int(np.argmin(d))]
        medoid_shop = shop_ids[medoid_idx]

        if p is not None:
            top_local = np.argsort(-p)[:top_k]
            top_members = shop_ids[idx[top_local]]
        else:
            top_members = shop_ids[idx[:top_k]]

        rows.append({
            "cluster": c,
            "size": size,
            "mean_prob": mean_p,
            "median_prob": med_p,
            "min_prob": min_p,
            "persistence": pers,
            "medoid_shop_id": medoid_shop,
            "top_members_shop_id": ",".join(map(str, top_members))
        })

    report = pd.DataFrame(rows).sort_values(
        by=["size", "mean_prob"], ascending=[False, False]
    )

    print("\n--- Per-cluster kvalitet (sort po size, pa mean_prob) ---")
    with pd.option_context("display.max_colwidth", 80):
        print(report.head(50).to_string(index=False))

    return report


def analyze_outliers(shop_ids, labels, outlier_scores=None, top_k=20):
    out_idx = np.where(labels == -1)[0]
    print("\n--- Outlieri (-1) ---")
    print("Broj outliera:", len(out_idx))

    if len(out_idx) == 0:
        return None

    if outlier_scores is None:
        print("Nema outlier_scores_.")
        print("Prvih nekoliko outlier shop_id:", shop_ids[out_idx[:min(10, len(out_idx))]])
        return None

    scores = outlier_scores[out_idx]
    top = np.argsort(-scores)[:min(top_k, len(out_idx))]
    top_idx = out_idx[top]

    out_df = pd.DataFrame({
        "shop_id": shop_ids[top_idx],
        "outlier_score": scores[top]
    })
    print(f"Top {len(out_df)} outliera po outlier_score:")
    print(out_df.to_string(index=False))
    return out_df


def plot_2d(Z2, labels, title="Clusters (2D)"):
    plt.figure(figsize=(9, 7))
    mask_noise = labels == -1

    plt.scatter(Z2[mask_noise, 0], Z2[mask_noise, 1], s=10, alpha=0.35)

    for c in sorted([k for k in np.unique(labels) if k != -1]):
        idx = labels == c
        plt.scatter(Z2[idx, 0], Z2[idx, 1], s=14, alpha=0.85, label=f"C{c}")

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    if len([k for k in np.unique(labels) if k != -1]) <= 15:
        plt.legend(markerscale=1.2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def soft_assign_outliers(Z, labels, probs=None, q=0.95, slack=1.2):
    """
    Soft-assign HDBSCAN outliere (-1) najbližem klasteru u prostoru Z.
    Prag po klasteru = q-percentil distanci članova do centroida * slack.
    Vraća: final_labels, assign_confidence, nearest_cluster, nearest_dist, per_cluster_threshold
    """
    labels = labels.copy()
    n = len(labels)

    clusters = [c for c in np.unique(labels) if c != -1]
    if len(clusters) == 0:
        return labels, np.zeros(n), np.full(n, -1), np.full(n, np.nan), {}

    centroids = {}
    thresholds = {}

    for c in clusters:
        idx = np.where(labels == c)[0]
        Xc = Z[idx]
        mu = Xc.mean(axis=0)
        centroids[c] = mu

        d = np.linalg.norm(Xc - mu, axis=1)
        base = np.percentile(d, q * 100) if len(d) >= 5 else float(np.max(d))
        thresholds[c] = float(base * slack)

    C = np.vstack([centroids[c] for c in clusters])  # (K, dim)
    out_idx = np.where(labels == -1)[0]

    final_labels = labels.copy()
    nearest_cluster = np.full(n, -1, dtype=int)
    nearest_dist = np.full(n, np.nan, dtype=float)
    assign_conf = np.zeros(n, dtype=float)

    if len(out_idx) == 0:
        if probs is not None:
            assign_conf[np.where(labels != -1)[0]] = probs[np.where(labels != -1)[0]]
        return final_labels, assign_conf, nearest_cluster, nearest_dist, thresholds

    Z_out = Z[out_idx]
    dmat = np.linalg.norm(Z_out[:, None, :] - C[None, :, :], axis=2)  # (M, K)

    best_k = np.argmin(dmat, axis=1)
    best_d = dmat[np.arange(len(out_idx)), best_k]

    for j, i in enumerate(out_idx):
        c = clusters[int(best_k[j])]
        d = float(best_d[j])

        nearest_cluster[i] = c
        nearest_dist[i] = d

        thr = thresholds[c]
        # if d <= thr:
        final_labels[i] = c
        assign_conf[i] = max(0.0, 1.0 - (d / thr))
        # else:
        #     final_labels[i] = -1
        #     assign_conf[i] = 0.0

    if probs is not None:
        non_out = np.where(labels != -1)[0]
        assign_conf[non_out] = probs[non_out]

    return final_labels, assign_conf, nearest_cluster, nearest_dist, thresholds


# -----------------------------
# 1) Load embeddings
# -----------------------------
df = pd.read_csv("data/extension3/shop_embeddings_chronos2.csv")
shop_ids = df["shop_id"].values
X = df.drop(columns=["shop_id"]).values
print("Shape:", df.shape)

# -----------------------------
# 2) L2 normalize
# -----------------------------
X_norm = normalize(X, norm="l2")

# -----------------------------
# 3) UMAP -> (space for clustering)
# -----------------------------
umap_cluster = umap.UMAP(
    n_neighbors=20,
    min_dist=0.0,
    n_components=20,
    metric="cosine",
    random_state=0
)
Z = umap_cluster.fit_transform(X_norm)  # (N, 20)

umap_viz = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=0
)
Z2 = umap_viz.fit_transform(X_norm)

# -----------------------------
# 4) HDBSCAN on UMAP space
# -----------------------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom"
)

labels = clusterer.fit_predict(Z)
probs = getattr(clusterer, "probabilities_", None)
persistence = getattr(clusterer, "cluster_persistence_", None)
outlier_scores = getattr(clusterer, "outlier_scores_", None)

# -----------------------------
# 5) Soft-assign outliers
# -----------------------------
final_labels, assign_conf, nn_cluster, nn_dist, thresholds = soft_assign_outliers(
    Z=Z,
    labels=labels,
    probs=probs,
    q=0.95,
    slack=1.2
)

print("\n--- Soft-assign summary ---")
n_out_before = int(np.sum(labels == -1))
n_out_after  = int(np.sum(final_labels == -1))
print("Outlieri prije:", n_out_before)
print("Outlieri poslije:", n_out_after)
print("Dodijeljeno:", n_out_before - n_out_after)

# -----------------------------
# 6) Analysis report (core clusters)
# -----------------------------
report = summarize_clusters(
    shop_ids=shop_ids,
    X=Z,
    labels=labels,
    probs=probs,
    persistence=persistence,
    top_k=5
)

out_df = analyze_outliers(
    shop_ids=shop_ids,
    labels=labels,
    outlier_scores=outlier_scores,
    top_k=20
)

# -----------------------------
# 7) Visualization (UMAP 2D) using final labels
# -----------------------------
plot_2d(Z2, final_labels, title="Shop clusters (UMAP(cosine) → HDBSCAN + soft-assign)")

# -----------------------------
# 8) Save results
# -----------------------------
out = pd.DataFrame({
    "shop_id": shop_ids,
    "core_cluster": labels,
    "final_cluster": final_labels,
    "is_outlier_core": (labels == -1),
    "is_outlier_final": (final_labels == -1),
    "assign_confidence": assign_conf,
    "nearest_cluster": nn_cluster,
    "nearest_dist": nn_dist
})

if probs is not None:
    out["cluster_prob_core"] = probs
if outlier_scores is not None:
    out["outlier_score"] = outlier_scores

out_path = "data/extension3/shop_clusters_umap_hdbscan_softassign.csv"
out.to_csv(out_path, index=False)
print("\nSaved:", out_path)

report_path = "data/extension3/shop_cluster_report_umap.csv"
report.to_csv(report_path, index=False)
print("Saved:", report_path)
