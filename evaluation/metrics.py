import numpy as np
import os
import pandas as pd

from .io import ensure_dir, load_csv, list_context_dirs, detect_store_ids


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

import numpy as np


def weighted_quantile_loss(
    y_true: np.ndarray,
    preds: dict[float, np.ndarray],
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> float:
    denom = float(np.sum(np.abs(y_true)) + 1e-9)
    total = 0.0
    for q in quantiles:
        diff = y_true - preds[q]
        loss = np.maximum(q * diff, (q - 1.0) * diff)
        total += float(np.sum(loss))
    return float(2.0 * total / denom / len(quantiles))


# helper to solve different path problems for extension and for baseline
def _candidate_pred_paths(ctx_dir: str, mode: str, sid: int) -> list[str]:
    fname = f"forecast_store_{sid}.csv"
    return [
        os.path.join(ctx_dir, mode, "predictions", fname),
        os.path.join(ctx_dir, mode, fname),
        os.path.join(ctx_dir, fname),
    ]

def compute_mae_open_closed(
    forecasts_root: str,
    gt_dir: str,
    reports_dir: str,
) -> str | None:
    ensure_dir(reports_dir)
    ctx_dirs = list_context_dirs(forecasts_root)
    if not ctx_dirs:
        return None

    rows: list[dict] = []

    for ctx_len, ctx_dir in ctx_dirs:
        store_ids = detect_store_ids(ctx_dir)
        if not store_ids:
            continue

        for sid in store_ids:
            gt_path = os.path.join(gt_dir, f"ground_truth_store_{sid}.csv")
            gt = load_csv(gt_path)
            if gt is None or "y_true" not in gt.columns:
                continue

            y_true = pd.to_numeric(gt["y_true"], errors="coerce").to_numpy(dtype=float)

            if "Open" in gt.columns:
                open_series = pd.to_numeric(gt["Open"], errors="coerce").fillna(1).to_numpy(dtype=float)
            else:
                open_series = np.ones(len(y_true), dtype=float)

            # compute for whichever modes exist
            for mode in ["univariate", "covariate"]:
                pred = None
                pred_path_used = None

                for pth in _candidate_pred_paths(ctx_dir, mode, sid):
                    pred = load_csv(pth)
                    if pred is not None:
                        pred_path_used = pth
                        break

                if pred is None:
                    continue

                if "median" in pred.columns:
                    y_hat = pd.to_numeric(pred["median"], errors="coerce").to_numpy(dtype=float)
                elif "p50" in pred.columns:
                    y_hat = pd.to_numeric(pred["p50"], errors="coerce").to_numpy(dtype=float)
                elif "0.5" in pred.columns:
                    y_hat = pd.to_numeric(pred["0.5"], errors="coerce").to_numpy(dtype=float)
                else:
                    continue

                n = min(len(y_true), len(y_hat), len(open_series))
                if n <= 0:
                    continue

                yt = y_true[:n]
                yh = y_hat[:n]
                os_ = open_series[:n]

                open_mask = os_ == 1
                closed_mask = os_ == 0

                rows.append(
                    {
                        "store_id": sid,
                        "context_len": ctx_len,
                        "mode": mode,
                        "mae_all": float(np.mean(np.abs(yt - yh))),
                        "mae_open": float(np.mean(np.abs(yt[open_mask] - yh[open_mask]))) if open_mask.any() else None,
                        "mae_closed": float(np.mean(np.abs(yt[closed_mask] - yh[closed_mask]))) if closed_mask.any() else None,
                        "n_open": int(open_mask.sum()),
                        "n_closed": int(closed_mask.sum()),
                        "pred_path": pred_path_used,
                    }
                )

    if not rows:
        return None

    out_path = os.path.join(reports_dir, "mae_open_closed.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path