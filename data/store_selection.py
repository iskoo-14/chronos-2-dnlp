import pandas as pd

def max_consecutive_daily_run(dates: pd.Series) -> int:
    if dates is None:
        return 0
    d = pd.to_datetime(dates).dropna().drop_duplicates().sort_values()
    if len(d) == 0:
        return 0
    diffs = d.diff().dt.days
    segments = diffs.ne(1).cumsum()
    return int(d.groupby(segments).size().max())

def has_continuous_recent_window(df_store, date_col="Date", target_col="Sales", window_length=256) -> bool:
    if len(df_store) == 0:
        return False

    g = df_store.copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    end_date = g[date_col].max()
    start_date = end_date - pd.Timedelta(days=window_length - 1)
    expected_range = pd.date_range(start_date, end_date, freq="D")

    g_recent = (
        g.drop_duplicates(subset=date_col)
        .set_index(date_col)
        .reindex(expected_range)
        .reset_index()
        .rename(columns={"index": date_col})
    )

    if len(g_recent) != window_length:
        return False

    return g_recent[target_col].notna().all()


def build_store_validity_report(
    df,
    store_col="Store",
    date_col="Date",
    target_col="Sales",
    min_run=256,
    recent_window_length=None,
    min_obs=None,
    min_mean_target=None,
    covariate_cols=None,
    check_recent_covariates=False,
    zero_tail_open_max=None,
    zero_tail_open_share=None,
    check_future_covariates=None,
) -> pd.DataFrame:
    if covariate_cols is None:
        covariate_cols = []

    records = []
    for store_id, g in df.groupby(store_col):
        g = g.copy()
        g[date_col] = pd.to_datetime(g[date_col])
        g = g.sort_values(date_col)

        observed = g[g[target_col].notna()]
        n_obs = len(observed)
        start_date = observed[date_col].min() if n_obs > 0 else pd.NaT
        start_date_all = g[date_col].min()
        end_date_observed = observed[date_col].max() if n_obs > 0 else pd.NaT
        end_date_all = g[date_col].max()
        max_run = max_consecutive_daily_run(observed[date_col])

        is_valid = True
        reasons = []
        recent_ok = True
        cov_ok = True
        zero_tail_open_ok = True
        zero_open_count = 0
        open_count = 0
        zero_open_share = 0.0
        max_zero_open_run = 0
        future_na_cols = []

        if n_obs == 0:
            is_valid = False
            reasons.append("no_target")

        # max_run gate only if recent window check is disabled
        if recent_window_length is None and max_run < min_run:
            is_valid = False
            reasons.append(f"max_run<{min_run}")

        if min_obs is not None and n_obs < min_obs:
            is_valid = False
            reasons.append(f"n_obs<{min_obs}")

        if min_mean_target is not None:
            mean_target = observed[target_col].mean()
            if pd.isna(mean_target) or mean_target < min_mean_target:
                is_valid = False
                reasons.append(f"mean_target<{min_mean_target}")

        if recent_window_length is not None:
            recent_ok = has_continuous_recent_window(
                g, date_col=date_col, target_col=target_col, window_length=recent_window_length
            )
            if not recent_ok:
                is_valid = False
                reasons.append("recent_gap")

        if check_recent_covariates and covariate_cols and recent_window_length is not None:
            end_date_cov = g[date_col].max()
            start_date_cov = end_date_cov - pd.Timedelta(days=recent_window_length - 1)
            expected_range_cov = pd.date_range(start_date_cov, end_date_cov, freq="D")
            g_recent_cov = (
                g.drop_duplicates(subset=date_col)
                .set_index(date_col)
                .reindex(expected_range_cov)
            )
            cov_na = g_recent_cov[covariate_cols].isna().any(axis=1).any()
            cov_ok = not cov_na
            if cov_na:
                is_valid = False
                reasons.append("recent_cov_na")

        
        if check_future_covariates and recent_window_length is not None:
            future_cols = check_future_covariates
            end_date_future = g[date_col].max()
            start_date_future = end_date_future - pd.Timedelta(days=recent_window_length - 1)
            expected_range_future = pd.date_range(start_date_future, end_date_future, freq="D")
            g_recent_future = (
                g.drop_duplicates(subset=date_col)
                .set_index(date_col)
                .reindex(expected_range_future)
            )
            na_mask = g_recent_future[future_cols].isna()
            future_na_cols = [c for c in future_cols if na_mask[c].any()]
            if future_na_cols:
                is_valid = False
                reasons.append(f"known_future_na[{','.join(future_na_cols)}]")

        if recent_window_length is not None and "Open" in g.columns:
            end_date_tail = g[date_col].max()
            start_date_tail = end_date_tail - pd.Timedelta(days=recent_window_length - 1)
            expected_range_tail = pd.date_range(start_date_tail, end_date_tail, freq="D")
            g_recent_tail = (
                g.drop_duplicates(subset=date_col)
                .set_index(date_col)
                .reindex(expected_range_tail)
            )
            open_vals = pd.to_numeric(g_recent_tail["Open"], errors="coerce")
            open_mask = open_vals == 1
            zeros_open = (g_recent_tail[target_col] == 0) & open_mask
            open_count = int(open_mask.sum())
            zero_open_count = int(zeros_open.sum())
            zero_open_share = float(zero_open_count / open_count) if open_count > 0 else 0.0
            segments = zeros_open.ne(zeros_open.shift()).cumsum()
            max_zero_open_run = int(zeros_open.groupby(segments).sum().max() or 0)

            if zero_tail_open_max is not None and open_count > 0 and max_zero_open_run > zero_tail_open_max:
                zero_tail_open_ok = False
                is_valid = False
                reasons.append(f"inconsistent_zero_open_run>{zero_tail_open_max}")

            if zero_tail_open_share is not None and open_count > 0 and zero_open_share > zero_tail_open_share:
                zero_tail_open_ok = False
                is_valid = False
                reasons.append(f"inconsistent_zero_open_share>{zero_tail_open_share}")

        records.append(
            {
                "store_id": store_id,
                "n_obs": n_obs,
                "start_date": start_date,
                "start_date_all": start_date_all,
                "end_date_observed": end_date_observed,
                "end_date_all": end_date_all,
                "max_consecutive_daily_run": max_run,
                "recent_window_ok": bool(recent_ok),
                "recent_cov_ok": bool(cov_ok),
                "zero_tail_open_ok": bool(zero_tail_open_ok),
                "zero_open_count": int(zero_open_count),
                "open_count": int(open_count),
                "zero_open_share": float(zero_open_share),
                "max_zero_open_run": int(max_zero_open_run),
                "n_total_days": int((end_date_all - start_date_all).days + 1)
                if pd.notna(end_date_all) and pd.notna(start_date_all)
                else None,
                "missing_target_days": int((end_date_all - start_date_all).days + 1 - n_obs)
                if pd.notna(end_date_all) and pd.notna(start_date_all)
                else None,
                "future_na_cols": ",".join(future_na_cols) if future_na_cols else "",
                "is_valid": bool(is_valid),
                "reasons": ";".join(reasons) if reasons else "",
            }
        )

    return pd.DataFrame(records).sort_values("store_id").reset_index(drop=True)


def filter_valid_stores(
    df,
    store_col="Store",
    date_col="Date",
    target_col="Sales",
    min_run=256,
    recent_window_length=None,
    min_obs=None,
    min_mean_target=None,
    covariate_cols=None,
    check_recent_covariates=False,
    zero_tail_open_max=None,
    zero_tail_open_share=None,
    check_future_covariates=None,
):
    report_df = build_store_validity_report(
        df,
        store_col=store_col,
        date_col=date_col,
        target_col=target_col,
        min_run=min_run,
        recent_window_length=recent_window_length,
        min_obs=min_obs,
        min_mean_target=min_mean_target,
        covariate_cols=covariate_cols,
        check_recent_covariates=check_recent_covariates,
        zero_tail_open_max=zero_tail_open_max,
        zero_tail_open_share=zero_tail_open_share,
        check_future_covariates=check_future_covariates,
    )
    valid_ids = report_df.loc[report_df["is_valid"], "store_id"].tolist()
    df_filtered = df[df[store_col].isin(valid_ids)].copy()
    return df_filtered, report_df, valid_ids


def reasons_summary(report_df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for reason_str in report_df.loc[~report_df["is_valid"], "reasons"].fillna(""):
        for r in [x for x in str(reason_str).split(";") if x]:
            counts[r] = counts.get(r, 0) + 1
    return pd.DataFrame([{"reason": r, "count": c} for r, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))])
