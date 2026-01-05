"""
Select the best context length per mode (covariate/univariate) based on WQL.

Inputs:
- reports/wql_by_context.csv (produced by compare_results.py)

Outputs:
- prints a short summary to stdout
"""
import os
import pandas as pd


def main():
    path = os.path.join("reports", "wql_by_context.csv")
    if not os.path.exists(path):
        print(f"[WARN] Missing {path}. Run compare_results.py first.")
        return

    df = pd.read_csv(path)
    if df.empty:
        print("[WARN] wql_by_context.csv is empty.")
        return

    summary = []
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

    if not summary:
        print("[WARN] No modes found in wql_by_context.csv.")
        return

    print("Best context per mode (by mean WQL):")
    for row in summary:
        print(
            f"- {row['mode']}: ctx={row['best_context']} "
            f"(mean_wql={row['mean_wql']:.4f}, std={row['std_wql']:.4f}, n={row['n_stores']})"
        )


if __name__ == "__main__":
    main()
