#!/usr/bin/env python3
"""
Machine-learning portfolios for N-En based on diag_all.csv and features_and_dh_returns.parquet.

- Merge diag_all and features_and_dh_returns on (date, y_target = y_price).
- Uses yhat_N-En as the sorting variable.
- Groups by calendar date (from features_and_dh_returns.date).
- Forms value-weighted quantile portfolios (terciles by default, adjustable).
- Computes daily portfolio returns and H-L, and summary tables.
- Exports tables, raw data used for graphs, and figures.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

DIAG_PATH = r"C:\Users\kawah\Documents\bitcoining\big_results\IBMCINTERACTIONS\diag_all.csv"
FEATURES_PATH = r"C:\Users\kawah\Documents\bitcoining\hope\features_and_dh_returns3.parquet"
OUTPUT_DIR = Path("big-figs/pf")

# Number of quantile buckets (e.g. 3 for terciles, 5 for quintiles, 10 for deciles)
N_BUCKETS = 3

# Number of last calendar dates to keep for evaluation (set to None to use all)
N_LAST_EVAL_DATES = None  # set to None if you want to use all merged dates

# Column names
PRED_COL = "yhat_N-En"   # predicted value used for sorting (from diag_all)
RET_COL = "dh_ret"       # realised delta-hedged return (from features file)
DATE_COL = "date"        # date column in features_and_dh_returns
LOG_OI_OPT_COL = "log_open_interest_opt"  # used for portfolio weights if present


# =========================
# Helper functions
# =========================

def value_weighted_mean(x: pd.Series, w: pd.Series) -> float:
    """Compute value-weighted mean, robust to NaNs."""
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(w_arr)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x_arr[mask], weights=w_arr[mask]))


def assign_quantile_buckets(df: pd.DataFrame,
                            group_col: str,
                            pred_col: str,
                            n_buckets: int,
                            bucket_col: str = "bucket") -> pd.DataFrame:
    """
    Within each group (e.g. day), assign observations to quantile buckets based on pred_col.

    Uses pd.qcut where possible; falls back gracefully if there are fewer unique predictions than buckets.
    Lowest predictions get bucket 1, highest bucket n_buckets.
    """
    df = df.copy()

    def _assign_for_group(g: pd.DataFrame) -> pd.DataFrame:
        preds = g[pred_col]
        nunique = preds.nunique(dropna=True)
        if nunique == 0:
            # all NaN predictions -> put everything in bucket 1
            g[bucket_col] = 1
            return g
        n_eff = min(n_buckets, nunique)
        try:
            buckets = pd.qcut(
                preds,
                q=n_eff,
                labels=range(1, n_eff + 1),
                duplicates="drop"
            )
        except ValueError:
            # fallback: rank then cut
            ranks = preds.rank(method="first")
            buckets = pd.qcut(
                ranks,
                q=n_eff,
                labels=range(1, n_eff + 1),
                duplicates="drop"
            )
        buckets = buckets.astype(int)
        g[bucket_col] = buckets
        return g

    df = df.groupby(group_col, group_keys=False).apply(_assign_for_group)
    return df


def compute_daily_portfolio_returns(df: pd.DataFrame,
                                    group_col: str,
                                    bucket_col: str,
                                    ret_col: str,
                                    weight_col: str) -> pd.DataFrame:
    """
    Compute value-weighted daily portfolio returns for each bucket.

    Returns a DataFrame with columns:
        [group_col, bucket_col, 'vw_ret', 'n_obs', 'sum_weight']
    """
    def _agg(g: pd.DataFrame) -> pd.Series:
        if weight_col in g.columns:
            w = g[weight_col]
        else:
            w = pd.Series(1.0, index=g.index)
        vw_ret = value_weighted_mean(g[ret_col], w)
        return pd.Series(
            {
                "vw_ret": vw_ret,
                "n_obs": int(len(g)),
                "sum_weight": float(np.nansum(w))
            }
        )

    grouped = df.groupby([group_col, bucket_col], as_index=False)
    port_daily = grouped.apply(_agg)
    return port_daily


def summarise_bucket_returns(port_daily: pd.DataFrame,
                             group_col: str,
                             bucket_col: str) -> pd.DataFrame:
    """
    Summarise daily portfolio returns by bucket:
    mean, median, std, count, hit ratio, daily Sharpe (mean/std).
    """
    grouped = port_daily.groupby(bucket_col)["vw_ret"]
    stats = grouped.agg(["mean", "median", "std", "count"]).rename(columns={"count": "n_days"})
    stats["hit_ratio"] = grouped.apply(lambda s: (s > 0).mean())
    stats["sharpe_daily"] = stats["mean"] / stats["std"]
    return stats.reset_index()


def compute_hl_series(port_daily: pd.DataFrame,
                      group_col: str,
                      bucket_col: str) -> pd.DataFrame:
    """
    Compute daily High-Low (highest bucket minus lowest bucket) portfolio returns.
    Returns DataFrame with columns [group_col, 'hl_ret'] plus bucket returns as columns.
    """
    pivot = port_daily.pivot(index=group_col, columns=bucket_col, values="vw_ret")
    pivot.columns = [f"Q{int(c)}" for c in pivot.columns]  # rename for clarity
    if pivot.shape[1] < 2:
        raise ValueError("Need at least two buckets to form H-L.")
    low_col = pivot.columns[0]
    high_col = pivot.columns[-1]
    pivot["hl_ret"] = pivot[high_col] - pivot[low_col]
    return pivot.reset_index()


def summarise_hl(hl_df: pd.DataFrame,
                 group_col: str) -> pd.Series:
    """
    Summarise H-L daily returns: mean, median, std, Sharpe, hit ratio.
    """
    hl = hl_df["hl_ret"]
    mean = hl.mean()
    median = hl.median()
    std = hl.std()
    sharpe = mean / std if std != 0 else np.nan
    hit_ratio = (hl > 0).mean()
    return pd.Series(
        {
            "mean": mean,
            "median": median,
            "std": std,
            "sharpe_daily": sharpe,
            "hit_ratio": hit_ratio,
            "n_days": len(hl)
        }
    )

def compute_composition(df: pd.DataFrame,
                        bucket_col: str,
                        weight_col: str,
                        group_col: str) -> pd.DataFrame:
    """
    Simple composition statistics by bucket, value-weighted across all days:
    - number of observations and total weight
    - share of calls (is_call)
    - mean DTE (dte)
    - mean |log_moneyness|
    - mean option relative spread (opt_rel_spread_final), if available
    """
    df = df.copy()

    # Build weights
    if weight_col in df.columns:
        df["_w"] = df[weight_col].astype(float)
    else:
        df["_w"] = 1.0

    records = []

    for b, g in df.groupby(bucket_col):
        w = g["_w"]

        rec = {
            bucket_col: b,
            "n_obs": int(len(g)),
            "sum_weight": float(np.nansum(w)),
        }

        # Value-weighted share of calls
        if "is_call" in g.columns:
            rec["call_share"] = value_weighted_mean(g["is_call"].astype(float), w)

        # Value-weighted mean DTE
        if "dte" in g.columns:
            rec["dte_mean"] = value_weighted_mean(g["dte"], w)

        # Value-weighted mean |log_moneyness|
        if "log_moneyness" in g.columns:
            rec["abs_log_moneyness"] = value_weighted_mean(g["log_moneyness"].abs(), w)

        # Value-weighted mean option relative spread
        if "opt_rel_spread_final" in g.columns:
            rec["opt_rel_spread_mean"] = value_weighted_mean(g["opt_rel_spread_final"], w)

        records.append(rec)

    if not records:
        return pd.DataFrame()

    comp = pd.DataFrame.from_records(records)

    # Sort by bucket for nicer output
    comp = comp.sort_values(by=bucket_col).reset_index(drop=True)
    return comp



def plot_monotonicity(bucket_stats: pd.DataFrame,
                      bucket_col: str,
                      out_path: Path) -> None:
    """Bar plot of mean daily returns by bucket."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = bucket_stats[bucket_col].values
    y = bucket_stats["mean"].values
    ax.bar(x, y)
    ax.set_xlabel("Bucket (sorted by predicted yhat_N-En)")
    ax.set_ylabel("Mean daily realised return (dh_ret)")
    ax.set_title("Mean daily realised return by prediction-sorted bucket")
    ax.axhline(0.0, linestyle="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_hl_timeseries(hl_df: pd.DataFrame,
                       group_col: str,
                       out_path: Path) -> None:
    """Line plot of daily H-L returns over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(hl_df[group_col])
    y = hl_df["hl_ret"]
    ax.plot(x, y, marker="o")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("H-L daily return (High bucket - Low bucket)")
    ax.set_title("Daily H-L portfolio return (yhat_N-En sorted)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =========================
# Main pipeline
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "data").mkdir(exist_ok=True)

    # Load data
    diag = pd.read_csv(DIAG_PATH)
    feats = pd.read_parquet(FEATURES_PATH)

    if PRED_COL not in diag.columns:
        raise KeyError(f"Prediction column '{PRED_COL}' not found in diag_all.csv.")
    if "y_target" not in diag.columns:
        raise KeyError("Column 'y_target' not found in diag_all.csv.")
    if "y_price" not in feats.columns:
        raise KeyError("Column 'y_price' not found in features_and_dh_returns.parquet.")
    if RET_COL not in feats.columns:
        raise KeyError(f"Return column '{RET_COL}' not found in features_and_dh_returns.parquet.")
    if DATE_COL not in feats.columns:
        raise KeyError(f"DATE_COL '{DATE_COL}' not found in features_and_dh_returns.parquet.")

    # Build merge keys: date (as calendar day) and price
    # diag: use 'group' as date indicator
    if "group" not in diag.columns:
        raise KeyError("Column 'group' (date/group) not found in diag_all.csv.")

    diag = diag.copy()
    feats = feats.copy()

    diag["date_key"] = pd.to_datetime(diag["group"]).dt.date
    feats[DATE_COL] = pd.to_datetime(feats[DATE_COL])
    feats["date_key"] = feats[DATE_COL].dt.date

    # Merge on (date_key, y_target = y_price)
    merged = pd.merge(
        feats,
        diag,
        left_on=["date_key", "y_price"],
        right_on=["date_key", "y_target"],
        how="inner",
        suffixes=("", "_diag")
    )

    if len(merged) == 0:
        raise ValueError(
            "Merge on (date, y_price = y_target) produced zero rows. "
            "Check that y_target in diag_all and y_price in features_and_dh_returns are aligned."
        )

    # Optional: sanity check, not enforced
    if len(merged) != len(diag):
        print(
            f"Warning: merged rows ({len(merged)}) != diag_all rows ({len(diag)}). "
            f"This may be due to unmatched or duplicated (date, price) combinations."
        )

    # Define evaluation date
    merged["eval_date"] = merged["date_key"]

    # # Option weights: default to exp(log_open_interest_opt), else equal weights
    # if LOG_OI_OPT_COL in merged.columns:
    #     merged["oi_weight"] = np.exp(merged[LOG_OI_OPT_COL])
    # else:
    merged["oi_weight"] = 1.0

    # Keep only rows with non-missing prediction and return
    merged = merged[np.isfinite(merged[PRED_COL]) & np.isfinite(merged[RET_COL])].copy()

    # Restrict to last N_LAST_EVAL_DATES distinct eval_date if specified
    if N_LAST_EVAL_DATES is not None:
        unique_dates = sorted(pd.unique(merged["eval_date"]))
        if len(unique_dates) < N_LAST_EVAL_DATES:
            print(
                f"Warning: requested {N_LAST_EVAL_DATES} evaluation dates, "
                f"but only found {len(unique_dates)}. Using all."
            )
            eval_dates = unique_dates
        else:
            eval_dates = unique_dates[-N_LAST_EVAL_DATES:]
        merged = merged[merged["eval_date"].isin(eval_dates)].copy()

    # Assign quantile buckets within each eval_date based on predicted yhat_N-En
    merged = assign_quantile_buckets(
        df=merged,
        group_col="eval_date",
        pred_col=PRED_COL,
        n_buckets=N_BUCKETS,
        bucket_col="bucket"
    )

    # Compute daily portfolio returns
    port_daily = compute_daily_portfolio_returns(
        df=merged,
        group_col="eval_date",
        bucket_col="bucket",
        ret_col=RET_COL,
        weight_col="oi_weight"
    )

    # Summarise bucket returns
    bucket_stats = summarise_bucket_returns(
        port_daily=port_daily,
        group_col="eval_date",
        bucket_col="bucket"
    )

    # Compute H-L series and summary
    hl_df = compute_hl_series(
        port_daily=port_daily,
        group_col="eval_date",
        bucket_col="bucket"
    )
    hl_summary = summarise_hl(hl_df=hl_df, group_col="eval_date")

    # Composition table
    comp_table = compute_composition(
        df=merged,
        bucket_col="bucket",
        weight_col="oi_weight",
        group_col="eval_date"
    )

    # =========================
    # Export tables and data
    # =========================

    port_daily.to_csv(OUTPUT_DIR / "data" / "daily_portfolio_returns.csv", index=False)
    hl_df.to_csv(OUTPUT_DIR / "data" / "daily_hl_series.csv", index=False)
    merged[["eval_date", "bucket", PRED_COL, RET_COL, "oi_weight"]].to_csv(
        OUTPUT_DIR / "data" / "bucket_assignment_and_core_vars.csv", index=False
    )

    bucket_stats.to_csv(OUTPUT_DIR / "tables" / "bucket_return_stats.csv", index=False)
    comp_table.to_csv(OUTPUT_DIR / "tables" / "bucket_composition.csv", index=False)
    hl_summary.to_frame().T.to_csv(
        OUTPUT_DIR / "tables" / "hl_summary.csv",
        index=False
    )

    # =========================
    # Export figures
    # =========================

    plot_monotonicity(
        bucket_stats=bucket_stats,
        bucket_col="bucket",
        out_path=OUTPUT_DIR / "figures" / "monotonicity_mean_return_by_bucket.png"
    )

    plot_hl_timeseries(
        hl_df=hl_df,
        group_col="eval_date",
        out_path=OUTPUT_DIR / "figures" / "hl_daily_timeseries.png"
    )

    print("Finished. Results written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
