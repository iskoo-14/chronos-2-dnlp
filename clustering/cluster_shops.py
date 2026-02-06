from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

def run_kmeans_clustering(
    features_csv: Path,
    output_csv: Path,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    n_init: int = 10,
):
    """
    Reads shop-level feature CSV, standardizes features, tries KMeans for k in [k_min, k_max],
    selects the k with best Silhouette score, writes clusters to output_csv,
    and prints a sorted summary of Silhouette scores. Also prints per-cluster silhouette means.
    """

    df = pd.read_csv(features_csv)
    if "shop_id" not in df.columns:
        raise ValueError("Expected column 'shop_id' in features_csv")

    X = df.drop(columns=["shop_id"])
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 shops to compute meaningful clustering metrics.")

    X_scaled = StandardScaler().fit_transform(X)

    results = []
    best = {"k": None, "score": -np.inf, "labels": None}

    for k in range(k_min, k_max + 1):
        # Silhouette is not defined if k >= n_samples or if a cluster has 1 unique label only
        if k >= X_scaled.shape[0]:
            continue

        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_scaled)

        # In rare cases KMeans can create fewer than k unique clusters (degenerate case)
        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(X_scaled, labels)
        results.append({"k": k, "silhouette": score})

        if score > best["score"]:
            best.update({"k": k, "score": score, "labels": labels})

    if best["k"] is None:
        raise RuntimeError("Failed to find a valid k for silhouette scoring. Check your data.")

    # Print sorted summary
    summary_df = pd.DataFrame(results).sort_values("silhouette", ascending=False).reset_index(drop=True)
    print("\n=== Silhouette summary (sorted) ===")
    print(summary_df.to_string(index=False, formatters={"silhouette": "{:.4f}".format}))
    print(f"\nBest k = {best['k']} with silhouette = {best['score']:.4f}")

    # Assign best labels
    df["cluster"] = best["labels"]

    # Per-cluster silhouette means (useful for interpretation)
    sil_samples = silhouette_samples(X_scaled, best["labels"])
    df["_silhouette"] = sil_samples

    per_cluster = (
        df.groupby("cluster")["_silhouette"]
          .agg(["count", "mean", "std"])
          .sort_values("mean", ascending=False)
    )

    print("\n=== Per-cluster silhouette (best k) ===")
    print(per_cluster.to_string(formatters={"mean": "{:.4f}".format, "std": "{:.4f}".format}))

    # Save output without the helper column if you want clean CSV:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_silhouette"]).to_csv(output_csv, index=False)

    return df, summary_df, per_cluster


def run_gmm_clustering(
    features_csv: Path,
    output_csv: Path,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    covariance_type: str = "full",  # "full" je default i najfleksibilniji
    selection_criterion: str = "bic",  # "bic" ili "aic"
):
    """
    Reads shop-level feature CSV, standardizes features, tries GaussianMixture for k in [k_min, k_max],
    selects the k with best BIC/AIC (lower is better), writes clusters to output_csv,
    and prints a sorted summary (criterion + silhouette). Also prints per-cluster silhouette means.

    Returns:
        df_with_clusters, summary_df, per_cluster_df
    """

    df = pd.read_csv(features_csv)
    if "shop_id" not in df.columns:
        raise ValueError("Expected column 'shop_id' in features_csv")

    X = df.drop(columns=["shop_id"])
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 shops to compute meaningful clustering metrics.")

    X_scaled = StandardScaler().fit_transform(X)

    selection_criterion = selection_criterion.lower()
    if selection_criterion not in {"bic", "aic"}:
        raise ValueError("selection_criterion must be 'bic' or 'aic'")

    results = []
    best = {"k": None, "criterion": np.inf, "labels": None, "model": None, "silhouette": None}

    for k in range(k_min, k_max + 1):
        if k >= X_scaled.shape[0]:
            continue

        gmm = GaussianMixture(
            n_components=k,
            random_state=random_state,
            covariance_type=covariance_type,
        )
        labels = gmm.fit_predict(X_scaled)

        # GMM can also collapse components; ensure at least 2 clusters
        if len(np.unique(labels)) < 2:
            continue

        bic = gmm.bic(X_scaled)
        aic = gmm.aic(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        results.append({
            "k": k,
            "bic": bic,
            "aic": aic,
            "silhouette": sil
        })

        criterion_value = bic if selection_criterion == "bic" else aic
        if criterion_value < best["criterion"]:
            best.update({
                "k": k,
                "criterion": criterion_value,
                "labels": labels,
                "model": gmm,
                "silhouette": sil,
            })

    if best["k"] is None:
        raise RuntimeError("Failed to find a valid GMM configuration. Check your data.")

    summary_df = pd.DataFrame(results).sort_values(selection_criterion, ascending=True).reset_index(drop=True)

    print(f"\n=== GMM summary (sorted by {selection_criterion.upper()}) ===")
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "bic": "{:.1f}".format,
                "aic": "{:.1f}".format,
                "silhouette": "{:.4f}".format,
            },
        )
    )
    print(
        f"\nBest k = {best['k']} by {selection_criterion.upper()} "
        f"({best['criterion']:.1f}), silhouette = {best['silhouette']:.4f}, cov_type = {covariance_type}"
    )

    # Assign best labels
    df["cluster"] = best["labels"]

    # Per-cluster silhouette means (for interpretability)
    sil_samples = silhouette_samples(X_scaled, best["labels"])
    df["_silhouette"] = sil_samples

    per_cluster = (
        df.groupby("cluster")["_silhouette"]
          .agg(["count", "mean", "std"])
          .sort_values("mean", ascending=False)
    )

    print("\n=== Per-cluster silhouette (best GMM) ===")
    print(per_cluster.to_string(formatters={"mean": "{:.4f}".format, "std": "{:.4f}".format}))

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["_silhouette"]).to_csv(output_csv, index=False)

    return df, summary_df, per_cluster