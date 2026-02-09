import os, glob, re, json, random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from config import HORIZON

from models.residual_head import QuantileResidualHead, pinball_loss, enforce_monotonic

# for evaluation
from evaluation.io import ensure_dir
from evaluation.compare_results import compute_wql_per_store, summarize_wql, write_comparison_report
from evaluation.metrics import compute_mae_open_closed

# ---- PATHS ----
FORECASTS_DIR = r"outputs/extension1/forecasts/ctx_512/covariate"
GT_DIR        = r"outputs/extension1/ground_truth"
SHOP_EMB_CSV  = r"data/extension3/shop_embeddings_with_cluster.csv"
CLUST_EMB_CSV = r"data/extension3/cluster_embeddings_mean.csv"
OUT_DIR       = r"outputs/extension3"

# ---- PARAMS ----
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTILES = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32, device=DEVICE)
FORECAST_COLS = ["p10", "median", "p90"]

os.makedirs(OUT_DIR, exist_ok=True)
CORR_DIR = os.path.join(OUT_DIR, "forecasts", "ctx_512", "covariate")
os.makedirs(CORR_DIR, exist_ok=True)

# ---- LOAD EMBEDDINGS ----
df_shop = pd.read_csv(SHOP_EMB_CSV)
emb_cols = [c for c in df_shop.columns if c not in ["shop_id", "cluster"]]
shop_emb = {int(r.shop_id): r[emb_cols].values.astype(np.float32) for _, r in df_shop.iterrows()}
shop_cluster = {int(r.shop_id): int(r.cluster) for _, r in df_shop.iterrows()}

df_cluster = pd.read_csv(CLUST_EMB_CSV)
cluster_emb = {int(r.cluster): r[emb_cols].values.astype(np.float32) for _, r in df_cluster.iterrows()}

# ---- LOAD DATASET  ----
examples = []
for fp in glob.glob(os.path.join(FORECASTS_DIR, "forecast_store_*.csv")):
    sid = int(re.search(r"store_(\d+)\.csv", fp).group(1))
    gp = os.path.join(GT_DIR, f"ground_truth_store_{sid}.csv")
    if not os.path.exists(gp):
        continue

    df_f = pd.read_csv(fp)
    df_g = pd.read_csv(gp)

    df_f["_dt_"] = pd.to_datetime(df_f["timestamp"])
    df_g["_dt_"] = pd.to_datetime(df_g["timestamp"])

    df_m = df_f.merge(df_g[["_dt_", "y_true"]], on="_dt_").sort_values("_dt_")

    df_m = df_m.iloc[:HORIZON]
    y_ch = df_m[FORECAST_COLS].values.T.astype(np.float32)  # (3, H)
    y_true = df_m["y_true"].values.astype(np.float32)       # (H,)

    z_shop = shop_emb[sid]
    z_cluster = cluster_emb[shop_cluster[sid]]
    x_head = np.concatenate([z_shop, z_cluster]).astype(np.float32)

    examples.append((sid, x_head, y_ch, y_true))

H = HORIZON
d_in = examples[0][1].shape[0]

# ---- SPLIT ----
store_ids = [e[0] for e in examples]
train_ids, test_ids = train_test_split(store_ids, test_size=TEST_SIZE, random_state=42)
train_ex = [e for e in examples if e[0] in train_ids]
test_ex  = [e for e in examples if e[0] in test_ids]

# ---- MODEL ----
head = QuantileResidualHead(d_in=d_in, horizon=H).to(DEVICE)
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ---- TRAIN ----
for epoch in range(1, EPOCHS + 1):
    random.shuffle(train_ex)
    head.train()
    losses = []

    for _, x_head, y_ch, y_true in train_ex:
        x = torch.tensor(x_head, device=DEVICE).unsqueeze(0)
        y_ch_t = torch.tensor(y_ch, device=DEVICE).unsqueeze(0)
        y_true_t = torch.tensor(y_true, device=DEVICE).unsqueeze(0)

        delta = head(x)
        y_final = enforce_monotonic(y_ch_t + delta)

        loss = pinball_loss(y_true_t, y_final, QUANTILES)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"[Epoch {epoch:03d}] train_pinball={np.mean(losses):.5f}")


# ---- SAVE MODEL ----
torch.save(head.state_dict(), os.path.join(OUT_DIR, "quantile_residual_head.pt"))

# ---- APPLY AND SAVE CORRECTED FORECASTS ----

corr_dir = os.path.join(OUT_DIR, "forecasts", "ctx_512", "covariate")
os.makedirs(corr_dir, exist_ok=True)

test_set = set(test_ids)
for sid, x_head, y_ch, _ in examples:
    if sid not in test_set:
        continue
    x = torch.tensor(x_head, device=DEVICE).unsqueeze(0)
    delta = head(x).squeeze(0).detach().cpu().numpy()  # (3, H)
    y_final = np.sort(y_ch + delta, axis=0)

    in_path = os.path.join(FORECASTS_DIR, f"forecast_store_{sid}.csv")
    df_f = pd.read_csv(in_path)
    df_f[FORECAST_COLS] = y_final.T
    df_f.to_csv(os.path.join(corr_dir, f"forecast_store_{sid}.csv"), index=False)

print("DONE")

# ---- EVALUATION ----

REPORTS_DIR = ensure_dir(os.path.join("reports", "extension3"))

records, text_report = compute_wql_per_store(
    forecasts_root=os.path.join(OUT_DIR, "forecasts"),
    gt_dir=GT_DIR,
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
    row = grouped_df.iloc[0]  # single context, single mode
    text_report.append(
        f"CTX {row['context_length']} {row['mode']}: "
        f"mean_wql={row['mean_wql']:.4f} std_wql={row['std_wql']:.4f} "
        f"mean_mae={row['mean_mae']:.2f} mean_rmse={row['mean_rmse']:.2f} "
        f"p10_under={row['mean_p10_under']:.2f} p90_over={row['mean_p90_over']:.2f} "
        f"n_stores={int(row['n_stores'])}"
    )

write_comparison_report(reports_dir=REPORTS_DIR, text_lines=text_report)

compute_mae_open_closed(
    forecasts_root=os.path.join(OUT_DIR, "forecasts"),
    gt_dir=GT_DIR,
    reports_dir=REPORTS_DIR,
)
