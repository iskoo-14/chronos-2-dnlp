import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from collections import Counter
import hdbscan
import matplotlib
matplotlib.use("Agg")  # MUST be before importing pyplot
import matplotlib.pyplot as plt
import umap.umap_ as umap  # pip install umap-learn
import os



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
# Helpers for scenario summary
# -----------------------------
def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan
    return float(np.sum(values * weights) / np.sum(weights))

def scenario_metrics(labels, probs, persistence):
    # core labels (bez soft-assign)
    n = len(labels)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / n if n > 0 else np.nan

    # cluster sizes (core)
    unique = sorted([c for c in np.unique(labels) if c != -1])
    sizes = np.array([np.sum(labels == c) for c in unique], dtype=int)

    n_clusters = len(unique)
    max_cluster = int(sizes.max()) if len(sizes) else 0

    # weighted mean prob (core, ignore noise)
    if probs is not None and len(unique) > 0:
        mask = labels != -1
        w_mean_prob = float(np.mean(probs[mask])) if np.any(mask) else np.nan
    else:
        w_mean_prob = np.nan

    # weighted mean persistence (po klasteru, weighted by size)
    if persistence is not None and len(unique) > 0:
        # cluster_persistence_ je po klaster labelima (0..k-1) u internom smislu.
        # Najsigurnije je uzeti persistence po redosledu unique cluster labela
        # samo ako su labeli 0..k-1. Ako nisu, mapiraćemo:
        pers_map = {}
        for i, c in enumerate(sorted(unique)):
            # Ako su klasteri numerisani 0..k-1, i==c. Ali ne mora uvijek.
            # cluster_persistence_ je indexiran internim cluster id-jem,
            # koji najčešće odgovara labelu. Ako ti to nije slučaj, reci pa ćemo preciznije.
            if c < len(persistence):
                pers_map[c] = persistence[c]
        pers_vals = np.array([pers_map.get(c, np.nan) for c in unique], dtype=float)
        # weighted mean, ignoriši NaN:
        ok = ~np.isnan(pers_vals)
        w_mean_pers = weighted_mean(pers_vals[ok], sizes[ok]) if np.any(ok) else np.nan
    else:
        w_mean_pers = np.nan

    return {
        "n_points": n,
        "n_clusters_core": n_clusters,
        "noise_core": n_noise,
        "noise_core_ratio": noise_ratio,
        "max_cluster_core": max_cluster,
        "mean_prob_core": w_mean_prob,
        "mean_persistence_core_w": w_mean_pers,
    }
    

def plot_2d_and_save(Z2, labels, title, save_path, dpi=200):
    """
    Pozove postojeći plot_2d(...) i zatim snimi trenutnu figuru na save_path.
    """
    plot_2d(Z2, labels, title=title)  # tvoja funkcija
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()  # bitno da se ne gomilaju figure u petlji




#PATHS
TRAIN_DATA = "data/extension3/shop_embeddings/train.csv"
TEST_DATA = "data/extension3/shop_embeddings/test.csv"


# -----------------------------
# 1) Load embeddings
# -----------------------------
df_train = pd.read_csv(TRAIN_DATA)
shop_ids = df_train["shop_id"].values
X = df_train.drop(columns=["shop_id"]).values
print("Shape:", df_train.shape)

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

Z_train = umap_cluster.fit_transform(X_norm)  # (N, 20)

#Used for vizualization and interpretation
umap_viz = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=0
)

Z2_train = umap_viz.fit_transform(X_norm)

scenarios = [
    {
        "name": "S1_base",
        "hdb": dict(
            min_cluster_size=10,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        ),
        "description": "Baseline – balanced density sensitivity"
    },
    {
        "name": "S2_more_strict_density",
        "hdb": dict(
            min_cluster_size=10,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
        ),
        "description": "Stricter density → more noise, purer cores"
    },
    {
        "name": "S3_larger_min_cluster",
        "hdb": dict(
            min_cluster_size=20,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        ),
        "description": "Coarser segmentation → macro clusters"
    },
    {
        "name": "S4_leaf_granular",
        "hdb": dict(
            min_cluster_size=10,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="leaf",
        ),
        "description": "Fine-grained segmentation (over-segmentation check)"
    },
    {
        "name": "S5_high_density_strict",
        "hdb": dict(
            min_cluster_size=20,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
        ),
        "description": "Very conservative clustering (stress test)"
    },
]

base_dir = "data/extension3/scenarios"
os.makedirs(base_dir, exist_ok=True)

all_summaries = []
all_outputs = []

for sc in scenarios:
    name = sc["name"]
    hdb_params = sc["hdb"]

    # --- Per-scenario folders ---
    sc_dir = os.path.join(base_dir, name)
    plots_dir = os.path.join(sc_dir, "plots")
    outputs_dir = os.path.join(sc_dir, "outputs")
    reports_dir = os.path.join(sc_dir, "reports")
    outliers_dir = os.path.join(sc_dir, "outliers")

    for d in [plots_dir, outputs_dir, reports_dir, outliers_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n================= {name} =================")
    print("HDBSCAN:", hdb_params)

    # -----------------------------
    # 4) HDBSCAN on UMAP space
    # -----------------------------
    clusterer = hdbscan.HDBSCAN(**hdb_params)
    labels = clusterer.fit_predict(Z_train)

    probs = getattr(clusterer, "probabilities_", None)
    persistence = getattr(clusterer, "cluster_persistence_", None)
    outlier_scores = getattr(clusterer, "outlier_scores_", None)

    # -----------------------------
    # 5) Soft-assign outliers
    # -----------------------------
    final_labels, assign_conf, nn_cluster, nn_dist, thresholds = soft_assign_outliers(
        Z=Z_train,
        labels=labels,
        probs=probs,
    )

    print("\n--- Soft-assign summary ---")
    n_out_before = int(np.sum(labels == -1))
    n_out_after  = int(np.sum(final_labels == -1))
    print("Outliers before:", n_out_before)
    print("Outliers after:", n_out_after)
    print("Assigned:", n_out_before - n_out_after)

    # -----------------------------
    # 6) Analysis report (core clusters)
    # -----------------------------
    report = summarize_clusters(
        shop_ids=shop_ids,
        X=Z_train,
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
    # 7) Visualization + SAVE plots
    # -----------------------------
    # Core labels plot
    core_plot_path = os.path.join(plots_dir, f"{name}_core_labels.png")
    plot_2d_and_save(Z2_train, labels, title=f"{name}: CORE (HDBSCAN)", save_path=core_plot_path)

    # Final labels plot (after soft-assign)
    final_plot_path = os.path.join(plots_dir, f"{name}_final_labels.png")
    plot_2d_and_save(Z2_train, final_labels, title=f"{name}: FINAL (HDBSCAN + soft-assign)", save_path=final_plot_path)

    print("Saved plots:", core_plot_path, "and", final_plot_path)

    # -----------------------------
    # 8) Save results (CSV)
    # -----------------------------
    out = pd.DataFrame({
        "scenario": name,
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

    out_path = os.path.join(outputs_dir, f"shop_clusters_{name}.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

    report_path = os.path.join(reports_dir, f"shop_cluster_report_{name}.csv")
    report.to_csv(report_path, index=False)
    print("Saved:", report_path)

    outliers_path = os.path.join(outliers_dir, f"shop_outliers_{name}.csv")
    out_df.to_csv(outliers_path, index=False)
    print("Saved:", outliers_path)

    # (Optional) save thresholds/info from soft-assign for reproducibility
    thr_path = os.path.join(outputs_dir, f"softassign_thresholds_{name}.json")
    try:
        import json
        with open(thr_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=2)
        print("Saved:", thr_path)
    except Exception as e:
        print("Could not save thresholds JSON:", e)

    # Collect for master outputs
    all_outputs.append(out)

    # Scenario summary (ako već imaš scenario_metrics)
    sm = scenario_metrics(labels, probs, persistence)
    sm.update({
        "scenario": name,
        "min_cluster_size": hdb_params["min_cluster_size"],
        "min_samples": hdb_params["min_samples"],
        "cluster_selection_method": hdb_params["cluster_selection_method"],
        "outliers_final": n_out_after,
        "outliers_final_ratio": n_out_after / len(labels),
        "assigned_soft": (n_out_before - n_out_after),
        "core_plot_path": core_plot_path,
        "final_plot_path": final_plot_path,
    })
    all_summaries.append(sm)

# -----------------------------
# Save master comparison files
# -----------------------------
summary_df = pd.DataFrame(all_summaries).sort_values(
    by=["mean_persistence_core_w", "mean_prob_core", "outliers_final_ratio"],
    ascending=[False, False, True]
)

summary_path = os.path.join(base_dir, "scenario_comparison_summary.csv")
summary_df.to_csv(summary_path, index=False)
print("\nSaved scenario comparison:", summary_path)

all_out = pd.concat(all_outputs, ignore_index=True)
all_out_path = os.path.join(base_dir, "shop_clusters_all_scenarios.csv")
all_out.to_csv(all_out_path, index=False)
print("Saved all scenarios assignments:", all_out_path)





































































# # -----------------------------
# # 4) HDBSCAN on UMAP space
# # -----------------------------
# clusterer = hdbscan.HDBSCAN(
#     min_cluster_size=10,
#     min_samples=2,
#     metric="euclidean",
#     cluster_selection_method="eom"
# )

# labels = clusterer.fit_predict(Z_train)
# probs = getattr(clusterer, "probabilities_", None)
# persistence = getattr(clusterer, "cluster_persistence_", None)
# outlier_scores = getattr(clusterer, "outlier_scores_", None)

# # -----------------------------
# # 5) Soft-assign outliers
# # -----------------------------
# final_labels, assign_conf, nn_cluster, nn_dist, thresholds = soft_assign_outliers(
#     Z=Z_train,
#     labels=labels,
#     probs=probs,
#     q=0.95,
#     slack=1.2
# )

# print("\n--- Soft-assign summary ---")
# n_out_before = int(np.sum(labels == -1))
# n_out_after  = int(np.sum(final_labels == -1))
# print("Outliers before:", n_out_before)
# print("Outliers after:", n_out_after)
# print("Assigned:", n_out_before - n_out_after)

# # -----------------------------
# # 6) Analysis report (core clusters)
# # -----------------------------
# report = summarize_clusters(
#     shop_ids=shop_ids,
#     X=Z_train,
#     labels=labels,
#     probs=probs,
#     persistence=persistence,
#     top_k=5
# )

# out_df = analyze_outliers(
#     shop_ids=shop_ids,
#     labels=labels,
#     outlier_scores=outlier_scores,
#     top_k=20
# )

# # -----------------------------
# # 7) Visualization (UMAP 2D) using final labels
# # -----------------------------
# plot_2d(Z2_train, final_labels, title="Shop clusters (UMAP(cosine) → HDBSCAN + soft-assign)")

# # -----------------------------
# # 8) Save results
# # -----------------------------
# out = pd.DataFrame({
#     "shop_id": shop_ids,
#     "core_cluster": labels,
#     "final_cluster": final_labels,
#     "is_outlier_core": (labels == -1),
#     "is_outlier_final": (final_labels == -1),
#     "assign_confidence": assign_conf,
#     "nearest_cluster": nn_cluster,
#     "nearest_dist": nn_dist
# })

# if probs is not None:
#     out["cluster_prob_core"] = probs
# if outlier_scores is not None:
#     out["outlier_score"] = outlier_scores

# out_path = "data/extension3/shop_clusters_umap_hdbscan_softassign.csv"
# out.to_csv(out_path, index=False)
# print("\nSaved:", out_path)

# report_path = "data/extension3/shop_cluster_report_umap.csv"
# report.to_csv(report_path, index=False)
# print("Saved:", report_path)
