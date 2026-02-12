import os
import glob
import re
import random

import numpy as np
import pandas as pd
import torch

from config import HORIZON
from models.residual_head import QuantileResidualHead, pinball_loss, enforce_monotonic

# for evaluation
from evaluation.io import ensure_dir
from evaluation.compare_results import compute_wql_per_store, summarize_wql, write_comparison_report
from evaluation.metrics import compute_mae_open_closed


# PATHS
FORECASTS_DIR = r"outputs/extension1/forecasts/ctx_512/covariate"
GT_DIR        = r"outputs/extension1/ground_truth"

SHOP_EMB_DIR  = r"data/extension3/shop_embeddings"
TRAIN_SHOP_EMB_CSV = os.path.join(SHOP_EMB_DIR, "train_with_clusters.csv")
TEST_SHOP_EMB_CSV  = os.path.join(SHOP_EMB_DIR, "test_with_clusters.csv")

CLUST_EMB_DIR = r"data/extension3/cluster_embeddings"
CLUST_EMB_CSV = os.path.join(CLUST_EMB_DIR, "cluster_embeddings_mean.csv")

OUT_DIR = r"outputs/extension3"
os.makedirs(OUT_DIR, exist_ok=True)

CORR_DIR = os.path.join(OUT_DIR, "forecasts", "ctx_512", "covariate")
os.makedirs(CORR_DIR, exist_ok=True)


# PARAMS
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTILES = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32, device=DEVICE)
FORECAST_COLS = ["p10", "median", "p90"]


# UTILS
def build_examples_for_stores(
    store_ids: set[int],
    shop_emb: dict[int, np.ndarray],
    shop_cluster: dict[int, int],
    cluster_emb: dict[int, np.ndarray],
    forecasts_dir: str,
    gt_dir: str,
    horizon: int,
    forecast_cols: list[str],
):
    examples = []

    for fp in glob.glob(os.path.join(forecasts_dir, "forecast_store_*.csv")):
        m = re.search(r"store_(\d+)\.csv", fp)
        if not m:
            continue

        sid = int(m.group(1))
        if sid not in store_ids:
            continue

        gp = os.path.join(gt_dir, f"ground_truth_store_{sid}.csv")
        if not os.path.exists(gp):
            continue

        if sid not in shop_emb or sid not in shop_cluster:
            continue

        cl = int(shop_cluster[sid])
        if cl not in cluster_emb:
            continue

        df_f = pd.read_csv(fp)
        df_g = pd.read_csv(gp)

        df_f["_dt_"] = pd.to_datetime(df_f["timestamp"])
        df_g["_dt_"] = pd.to_datetime(df_g["timestamp"])

        df_m = df_f.merge(df_g[["_dt_", "y_true"]], on="_dt_").sort_values("_dt_")
        df_m = df_m.iloc[:horizon]

        if len(df_m) == 0:
            continue

        y_ch = df_m[forecast_cols].values.T.astype(np.float32)  # (3, H)
        y_true = df_m["y_true"].values.astype(np.float32)       # (H,)

        z_shop = shop_emb[sid]
        z_cluster = cluster_emb[cl]
        x_head = np.concatenate([z_shop, z_cluster]).astype(np.float32)

        examples.append((sid, x_head, y_ch, y_true))

    return examples


# 1) LOAD SPLIT FROM EMBEDDINGS FILES
df_train_shop = pd.read_csv(TRAIN_SHOP_EMB_CSV)
df_test_shop  = pd.read_csv(TEST_SHOP_EMB_CSV)

emb_cols = [c for c in df_train_shop.columns if c not in ["shop_id", "cluster"]]
emb_cols_test = [c for c in df_test_shop.columns if c not in ["shop_id", "cluster"]]

train_ids = sorted(df_train_shop["shop_id"].astype(int).unique().tolist())
test_ids  = sorted(df_test_shop["shop_id"].astype(int).unique().tolist())

train_set = set(train_ids)
test_set  = set(test_ids)

train_shop_emb = {int(r.shop_id): r[emb_cols].values.astype(np.float32) for _, r in df_train_shop.iterrows()}
train_shop_cluster = {int(r.shop_id): int(r.cluster) for _, r in df_train_shop.iterrows()}

test_shop_emb = {int(r.shop_id): r[emb_cols].values.astype(np.float32) for _, r in df_test_shop.iterrows()}
test_shop_cluster = {int(r.shop_id): int(r.cluster) for _, r in df_test_shop.iterrows()}

print(f"Split from embeddings: n_train={len(train_ids)} n_test={len(test_ids)}")


# 2) LOAD CLUSTER EMBEDDINGS
df_cluster = pd.read_csv(CLUST_EMB_CSV)
if "cluster" not in df_cluster.columns:
    raise ValueError(f"{CLUST_EMB_CSV} must have 'cluster' column.")
missing = [c for c in emb_cols if c not in df_cluster.columns]
if missing:
    raise ValueError(f"Cluster embedding file missing columns (first 10): {missing[:10]}")

cluster_emb = {int(r.cluster): r[emb_cols].values.astype(np.float32) for _, r in df_cluster.iterrows()}


# 3) BUILD TRAIN/TEST EXAMPLES (forecast + gt)
train_ex = build_examples_for_stores(
    store_ids=train_set,
    shop_emb=train_shop_emb,
    shop_cluster=train_shop_cluster,
    cluster_emb=cluster_emb,
    forecasts_dir=FORECASTS_DIR,
    gt_dir=GT_DIR,
    horizon=HORIZON,
    forecast_cols=FORECAST_COLS,
)

test_ex = build_examples_for_stores(
    store_ids=test_set,
    shop_emb=test_shop_emb,
    shop_cluster=test_shop_cluster,
    cluster_emb=cluster_emb,
    forecasts_dir=FORECASTS_DIR,
    gt_dir=GT_DIR,
    horizon=HORIZON,
    forecast_cols=FORECAST_COLS,
)

d_in = train_ex[0][1].shape[0]
H = HORIZON


# 4) MODEL + TRAIN
head = QuantileResidualHead(d_in=d_in, horizon=H).to(DEVICE)
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

for epoch in range(1, EPOCHS + 1):
    random.shuffle(train_ex)
    head.train()
    losses = []

    for _, x_head, y_ch, y_true in train_ex:
        x = torch.tensor(x_head, device=DEVICE).unsqueeze(0)       # (1, d_in)
        y_ch_t = torch.tensor(y_ch, device=DEVICE).unsqueeze(0)     # (1, 3, H)
        y_true_t = torch.tensor(y_true, device=DEVICE).unsqueeze(0) # (1, H)

        delta = head(x)  # (1, 3, H)
        y_final = enforce_monotonic(y_ch_t + delta)

        loss = pinball_loss(y_true_t, y_final, QUANTILES)
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"[Epoch {epoch:03d}] train_pinball={float(np.mean(losses)):.5f}")

torch.save(head.state_dict(), os.path.join(OUT_DIR, "quantile_residual_head.pt"))
print("Saved model:", os.path.join(OUT_DIR, "quantile_residual_head.pt"))


# 5) APPLY ON TEST + SAVE FINAL FORECASTS
head.eval()

for sid, x_head, y_ch, _ in test_ex:
    x = torch.tensor(x_head, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        delta = head(x).squeeze(0).cpu().numpy()  # (3, H)

    # enforce monotonic: p10 <= p50 <= p90
    y_final = np.sort(y_ch + delta, axis=0)  # (3, H)

    in_path = os.path.join(FORECASTS_DIR, f"forecast_store_{sid}.csv")
    if not os.path.exists(in_path):
        continue

    df_f = pd.read_csv(in_path)
    df_f = df_f.copy()

    df_f.loc[:HORIZON - 1, FORECAST_COLS] = y_final.T
    df_f.to_csv(os.path.join(CORR_DIR, f"forecast_store_{sid}.csv"), index=False)


# 6) EVALUATION (TEST ONLY)
REPORTS_DIR = ensure_dir(os.path.join("reports", "extension3_test_only"))

TEST_FORECASTS_ROOT = ensure_dir(os.path.join(OUT_DIR, "forecasts_test_only", "ctx_512", "covariate"))
TEST_GT_DIR = ensure_dir(os.path.join(OUT_DIR, "gt_test_only"))

# copy final forecasts
for sid in test_ids:
    src = os.path.join(CORR_DIR, f"forecast_store_{sid}.csv")
    if os.path.exists(src):
        pd.read_csv(src).to_csv(os.path.join(TEST_FORECASTS_ROOT, f"forecast_store_{sid}.csv"), index=False)

# copy gt
for sid in test_ids:
    gp = os.path.join(GT_DIR, f"ground_truth_store_{sid}.csv")
    if os.path.exists(gp):
        pd.read_csv(gp).to_csv(os.path.join(TEST_GT_DIR, f"ground_truth_store_{sid}.csv"), index=False)

records, text_report = compute_wql_per_store(
    forecasts_root=os.path.join(OUT_DIR, "forecasts_test_only"),
    gt_dir=TEST_GT_DIR,
    include_store_lines=False,
)

per_store_path, _, summary_path, grouped_df, filter_note = summarize_wql(
    records=records,
    reports_dir=REPORTS_DIR,
    apply_outlier_filter=True,
    outlier_threshold=0.5,
)

if grouped_df is not None and not grouped_df.empty:
    text_report.append(f"=== Summary {filter_note} ===")
    row = grouped_df.iloc[0]
    text_report.append(
        f"CTX {row['context_length']} {row['mode']}: "
        f"mean_wql={row['mean_wql']:.4f} std_wql={row['std_wql']:.4f} "
        f"mean_mae={row['mean_mae']:.2f} mean_rmse={row['mean_rmse']:.2f} "
        f"p10_under={row['mean_p10_under']:.2f} p90_over={row['mean_p90_over']:.2f} "
        f"n_stores={int(row['n_stores'])}"
    )

write_comparison_report(reports_dir=REPORTS_DIR, text_lines=text_report)

compute_mae_open_closed(
    forecasts_root=os.path.join(OUT_DIR, "forecasts_test_only"),
    gt_dir=TEST_GT_DIR,
    reports_dir=REPORTS_DIR,
)

print("Evaluation done.")

######################################################################

# 7) VISUALIZATION: plot 5 shops (median + CI) and save to visualization/extension3
import matplotlib.pyplot as plt

VIS_DIR = os.path.join("visualization", "extension3")
os.makedirs(VIS_DIR, exist_ok=True)

def plot_shop_forecast(sid: int, forecasts_path: str, gt_path: str, out_path: str, horizon: int):
    df_f = pd.read_csv(forecasts_path)
    df_g = pd.read_csv(gt_path)

    df_f["timestamp"] = pd.to_datetime(df_f["timestamp"])
    df_g["timestamp"] = pd.to_datetime(df_g["timestamp"])

    # merge and keep first horizon
    df = df_f.merge(df_g[["timestamp", "y_true"]], on="timestamp", how="inner").sort_values("timestamp").iloc[:horizon]

    if df.empty:
        print(f"[SKIP] Store {sid}: empty merged df")
        return

    plt.figure(figsize=(12, 4.5))
    plt.title("Covariate Forecast")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Sales")

    # median + confidence interval
    plt.plot(df["timestamp"], df["median"], label="Median")
    plt.fill_between(df["timestamp"], df["p10"], df["p90"], alpha=0.25, label="Confidence interval")

    # ground truth (optional)
    plt.plot(df["timestamp"], df["y_true"], linestyle="--", linewidth=1.5, label="Ground truth")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


available = []
for sid in test_ids:
    f_path = os.path.join(CORR_DIR, f"forecast_store_{sid}.csv")  # corrected forecasts from extension3
    g_path = os.path.join(GT_DIR, f"ground_truth_store_{sid}.csv")
    if os.path.exists(f_path) and os.path.exists(g_path):
        available.append(sid)

if len(available) == 0:
    print("[WARN] No available shops found for plotting (missing forecast or gt files).")
else:
    random.seed(42)
    chosen = available[:5] if len(available) < 5 else random.sample(available, 5)
    print("[INFO] Plotting shops:", chosen)

    for sid in chosen:
        f_path = os.path.join(CORR_DIR, f"forecast_store_{sid}.csv")
        g_path = os.path.join(GT_DIR, f"ground_truth_store_{sid}.csv")
        out_path = os.path.join(VIS_DIR, f"forecast_store_{sid}.png")
        plot_shop_forecast(sid, f_path, g_path, out_path, HORIZON)

