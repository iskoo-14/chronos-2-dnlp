import os
import pandas as pd


def select_best_context(reports_dir: str) -> list[dict]:
    path = os.path.join(reports_dir, "wql_by_context.csv")
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    summary: list[dict] = []
    for mode in ["covariate", "univariate"]:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        best = sub.loc[sub["mean_wql"].idxmin()]
        summary.append(
            {
                "mode": mode,
                "best_context": int(best["context_length"]),
                "mean_wql": float(best["mean_wql"]),
                "std_wql": float(best["std_wql"]),
                "n_stores": int(best["n_stores"]),
            }
        )
    return summary


def print_best_context(summary: list[dict]) -> None:
    if not summary:
        print("[WARN] No modes found in wql_by_context.csv.")
        return

    print("Best context per mode (by mean WQL):")
    for row in summary:
        print(
            f"- {row['mode']}: ctx={row['best_context']} "
            f"(mean_wql={row['mean_wql']:.4f}, std_wql={row['std_wql']:.4f}, n={row['n_stores']})"
        )
