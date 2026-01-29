import os
import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def list_context_dirs(forecasts_root: str) -> list[tuple[int, str]]:
    ctx_dirs: list[tuple[int, str]] = []
    if not os.path.exists(forecasts_root):
        return ctx_dirs

    for name in os.listdir(forecasts_root):
        p = os.path.join(forecasts_root, name)
        if os.path.isdir(p) and name.startswith("ctx_"):
            try:
                ctx = int(name.replace("ctx_", ""))
                ctx_dirs.append((ctx, p))
            except ValueError:
                continue

    return sorted(ctx_dirs, key=lambda x: x[0])


def detect_store_ids(ctx_dir: str) -> list[int]:
    candidate_dirs = [
        os.path.join(ctx_dir, "univariate", "predictions"),
        os.path.join(ctx_dir, "covariate", "predictions"),
        os.path.join(ctx_dir, "univariate"),
        os.path.join(ctx_dir, "covariate"),
        ctx_dir,
    ]

    files = []
    for d in candidate_dirs:
        if not os.path.exists(d):
            continue
        listed = os.listdir(d)
        matched = [f for f in listed if f.startswith("forecast_store_") and f.endswith(".csv")]
        files.extend(matched)

    ids: list[int] = []
    for f in set(files):
        s = f.replace("forecast_store_", "").replace(".csv", "")
        try:
            ids.append(int(s))
        except ValueError:
            print(f"[DBG detect_store_ids] cannot parse id from filename: {f}")
            continue

    out = sorted(set(ids))
    return out
