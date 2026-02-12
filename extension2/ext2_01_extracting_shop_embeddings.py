import math
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.chronos import load_model
from extension1.ext1_01_data_prep import read_valid_store_ids
from extension1.run_extension1 import read_processed_store, extension1_covariate_sets, ensure_dayofweek, temporal_split, PROCESSED_DIR_EXT1, CTX_LEN, HORIZON


def extract_past_embedding(
    pipeline,
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    horizon: int = 30,
    target_index: int = 0,
    pooling: str = "mean",  
    verbose: bool = True,
):
    """
    Extracts the shop embedding from the Chronos-2 encoder (past-only, target index 0),
    with a detailed trace of tensor shapes through key layers.

    Returns:
    shop_emb: torch.Tensor        [D]
    past_tokens: torch.Tensor     [Nc, D]
    info: dict (Nc, Nf, P, shapes, trace)
    """
    
    model = pipeline.inner_model
    model.eval()

    cache = {"enc": []}
    handles = []
    trace = []

    def _log(where, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        if torch.is_tensor(x):
            trace.append({
                "where": where,
                "shape": tuple(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
            })
        else:
            trace.append({"where": where, "shape": str(type(x))})

    # --- hooks ---
    def hook_final_ln(m, inp, out):
        # _log("encoder.final_layer_norm.out", out)
        cache["enc"].append(out.detach().cpu())  # [V, L, D]

    # def hook_enc_dropout(m, inp, out):
    #     _log("encoder.dropout.in", inp[0])
    #     _log("encoder.dropout.out", out)

    # def hook_out_patch_in(m, inp, out):
    #     _log("output_patch_embedding.in", inp[0])

    # def hook_out_patch_out(m, inp, out):
    #     _log("output_patch_embedding.out", out)

    # def hook_hidden_layer(m, inp, out):
    #     _log("output_patch_embedding.hidden_layer.in", inp[0])
    #     _log("output_patch_embedding.hidden_layer.out", out)

    # register hooks
    handles.append(model.encoder.final_layer_norm.register_forward_hook(hook_final_ln)) #we take it from this layer
    # handles.append(model.encoder.dropout.register_forward_hook(hook_enc_dropout))
    # handles.append(model.output_patch_embedding.register_forward_hook(hook_out_patch_in))
    # handles.append(model.output_patch_embedding.register_forward_hook(hook_out_patch_out))
    # handles.append(model.output_patch_embedding.hidden_layer.register_forward_hook(hook_hidden_layer))

    # run inference
    with torch.inference_mode():
        _ = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

    for h in handles:
        h.remove()

    # encoder output
    enc = torch.cat(cache["enc"], dim=0)  # [V, L, D]

    # token math
    P = pipeline.model_output_patch_size
    Nf = math.ceil(horizon / P)
    L = enc.shape[1]
    Nc = L - 1 - Nf  # [past | REG | future]

    # target variate -> past tokens
    target_enc = enc[target_index]   # [L, D]
    past_tokens = target_enc[:Nc, :] # [Nc, D]

    # pooling
    if pooling == "mean":
        shop_emb = past_tokens.mean(dim=0)
    elif pooling == "last":
        shop_emb = past_tokens[-1]
    else:
        raise ValueError("pooling must be 'mean' or 'last'")

    info = {
        "enc_shape": tuple(enc.shape),
        "past_tokens_shape": tuple(past_tokens.shape),
        "shop_emb_shape": tuple(shop_emb.shape),
        "P": int(P),
        "Nf": int(Nf),
        "Nc": int(Nc),
        "target_index": int(target_index),
        "pooling": pooling,
        "trace": trace,
    }

    if verbose:
        for row in trace:
            print(
                f"{row['where']:40s} {row['shape']} "
                f"{row.get('dtype','')} {row.get('device','')}"
            )
        print("INFO:", {k: v for k, v in info.items() if k != "trace"})

    return shop_emb, past_tokens, info

print("===================================================")
print(f"=== EXTRACTING SHOP EMBEDDINGS START  ===")
print("===================================================")

store_ids = read_valid_store_ids()
pipeline = load_model("amazon/chronos-2")


INCLUDE_EMA = True
INCLUDE_CHG = True
INCLUDE_ROLLING = False


PAST_ONLY_COVS, FUTURE_KNOWN_COVS = extension1_covariate_sets(
    include_ema=INCLUDE_EMA,
    include_chg=INCLUDE_CHG,
    include_rolling=INCLUDE_ROLLING,
)

rows = []
embed_dim = None

for k, sid in enumerate(store_ids):

    df = read_processed_store(sid, processed_dir=PROCESSED_DIR_EXT1)
    df = ensure_dayofweek(df)
    df_past, df_test = temporal_split(df, test_size=HORIZON)

    if len(df_past) > CTX_LEN:
        df_past = df_past.iloc[-CTX_LEN:].reset_index(drop=True)

    df_past = ensure_dayofweek(df_past)
    df_test = ensure_dayofweek(df_test)

    needed_ctx = ["id", "timestamp", "target"] + PAST_ONLY_COVS + FUTURE_KNOWN_COVS
    needed_fut = ["id", "timestamp"] + FUTURE_KNOWN_COVS

    missing_ctx = [c for c in needed_ctx if c not in df_past.columns]
    missing_fut = [c for c in needed_fut if c not in df_test.columns]

    if missing_ctx or missing_fut:
        print(f"[SKIP] Store {sid}: missing cov columns ctx={missing_ctx} fut={missing_fut}")
        continue

    context_cov = df_past[needed_ctx].copy()
    future_cov = df_test[needed_fut].copy()

    if "Open" in future_cov.columns and "Promo" in future_cov.columns:
        fut_open = pd.to_numeric(future_cov["Open"], errors="coerce").fillna(1)
        future_cov.loc[fut_open.eq(0), "Promo"] = 0
    if "Open" in context_cov.columns and "Promo" in context_cov.columns:
        ctx_open = pd.to_numeric(context_cov["Open"], errors="coerce").fillna(1)
        context_cov.loc[ctx_open.eq(0), "Promo"] = 0

    if context_cov.isna().any().any() or future_cov.isna().any().any():
        print(f"[SKIP] Store {sid}: NaN in covariates")
        continue

    # log only for the first shop
    verbose = (k == 0)
    shop_emb, past_tokens, info = extract_past_embedding(
        pipeline,
        context_cov,
        future_cov,
        horizon=30,
        target_index=0,
        pooling="mean",
        verbose=verbose,
    )

    emb = shop_emb.detach().cpu().numpy().astype(np.float32)  # [D]

    if embed_dim is None:
        embed_dim = emb.shape[0]
        feature_cols = [f"i{i}" for i in range(embed_dim)]
    else:
        if emb.shape[0] != embed_dim:
            print(f"[SKIP] Store {sid}: embedding dim mismatch {emb.shape[0]} != {embed_dim}")
            continue

    row = {"shop_id": sid}
    row.update({feature_cols[i]: float(emb[i]) for i in range(embed_dim)})
    rows.append(row)

print(f"Collected embeddings for {len(rows)} shops.")
emb_df = pd.DataFrame(rows)

# Load embeddings from CSV file
# csv_path = "data/extension3/shop_embeddings_chronos2.csv"
# emb_df = pd.read_csv(csv_path)

### SPLITTING EMBEDDINGS INTO TRAIN AND TEST
train_df, test_df = train_test_split(
    emb_df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Define output directory
base_path = "data/extension3/shop_embeddings"
os.makedirs(base_path, exist_ok=True)

# Define output file paths
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")

# Save train and test datasets
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Train saved to:", train_path, "shape:", train_df.shape)
print("Test saved to:", test_path, "shape:", test_df.shape)

print("===================================================")
print(f"=== EXTRACTING SHOP EMBEDDINGS FINISHED  ===")
print("===================================================")
