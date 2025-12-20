#!/usr/bin/env python3
"""
Machine-learning portfolios for N-En.

Updates:
- Fixes the "Theta = 0.00" issue.
- Gamma: SCALED (Bali et al.) -> Change in Delta per 1% price move.
- Vega/Theta: RAW -> Kept in dollar/points terms so they remain visible.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# Configuration
# =========================

DIAG_PATH = r"C:\Users\kawah\prettygood\cleanrepo\results\dh_ret\all-all\IBMCTINTERACTIONS\predictions_expanding.csv"
FEATURES_PATH = r"C:\Users\kawah\prettygood\cleanrepo\spread0.8\dhinput.parquet"
FUTURES_PATH = r"data\futures_cme-long.csv" # <--- CHECK THIS PATH
OUTPUT_DIR = Path("results/dh_ret/figures/pfUS_Tables")

N_BUCKETS = 3
PRED_COL = "yhat_N-En"
RET_COL = "dh_ret"
DATE_COL = "date"
WEIGHT_COL = "oi_weight" 
UNDERLYING_COL = "adj close"

# =========================
# Helper functions
# =========================

def value_weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(w_arr)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x_arr[mask], weights=w_arr[mask]))

def assign_quantile_buckets(df: pd.DataFrame, group_col: str, pred_col: str, n_buckets: int, bucket_col: str = "bucket") -> pd.DataFrame:
    df = df.copy()
    def _assign_for_group(g: pd.DataFrame) -> pd.DataFrame:
        preds = g[pred_col]
        nunique = preds.nunique(dropna=True)
        if nunique == 0:
            g[bucket_col] = 1
            return g
        n_eff = min(n_buckets, nunique)
        try:
            buckets = pd.qcut(preds, q=n_eff, labels=range(1, n_eff + 1), duplicates="drop")
        except ValueError:
            ranks = preds.rank(method="first")
            buckets = pd.qcut(ranks, q=n_eff, labels=range(1, n_eff + 1), duplicates="drop")
        g[bucket_col] = buckets.astype(int)
        return g
    df = df.groupby(group_col, group_keys=False).apply(_assign_for_group)
    return df

def compute_daily_portfolio_returns(df: pd.DataFrame, group_col: str, bucket_col: str, ret_col: str, weight_col: str) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        w = g[weight_col] if weight_col in g.columns else pd.Series(1.0, index=g.index)
        vw_ret = value_weighted_mean(g[ret_col], w)
        return pd.Series({"vw_ret": vw_ret})
    return df.groupby([group_col, bucket_col], as_index=False).apply(_agg)

def compute_hl_series(port_daily: pd.DataFrame, group_col: str, bucket_col: str) -> pd.DataFrame:
    pivot = port_daily.pivot(index=group_col, columns=bucket_col, values="vw_ret")
    if pivot.shape[1] < 2: 
        return pd.DataFrame() 
    low_col, high_col = pivot.columns[0], pivot.columns[-1]
    pivot["High - Low"] = pivot[high_col] - pivot[low_col]
    return pivot.reset_index()

# =========================
# Table Generation Functions
# =========================

def generate_performance_table(merged_df: pd.DataFrame, port_daily: pd.DataFrame, hl_daily: pd.DataFrame) -> pd.DataFrame:
    stats = []
    buckets = sorted(port_daily["bucket"].unique())
    
    for b in buckets:
        raw_b = merged_df[merged_df["bucket"] == b]
        daily_b = port_daily[port_daily["bucket"] == b]["vw_ret"]
        mean_ret = daily_b.mean()
        std_ret = daily_b.std()
        
        row = {
            "Portfolio": f"Bucket {b}",
            "Mean pred.": raw_b[PRED_COL].mean(),
            "Med. pred.": raw_b[PRED_COL].median(),
            "Mean ret.": mean_ret,
            "Med. ret.": daily_b.median(),
            "SD": std_ret,
            "SR": mean_ret / std_ret if std_ret != 0 else np.nan
        }
        stats.append(row)
        
    if "High - Low" in hl_daily.columns:
        hl_series = hl_daily["High - Low"]
        mean_ret = hl_series.mean()
        std_ret = hl_series.std()
        high_pred = stats[-1]["Mean pred."]
        low_pred = stats[0]["Mean pred."]
        high_med = stats[-1]["Med. pred."]
        low_med = stats[0]["Med. pred."]

        row = {
            "Portfolio": "High - Low",
            "Mean pred.": high_pred - low_pred,
            "Med. pred.": high_med - low_med,
            "Mean ret.": mean_ret,
            "Med. ret.": hl_series.median(),
            "SD": std_ret,
            "SR": mean_ret / std_ret if std_ret != 0 else np.nan
        }
        stats.append(row)
    
    label_map = {f"Bucket {buckets[0]}": "Low", f"Bucket {buckets[-1]}": "High"}
    if len(buckets) == 3:
        label_map[f"Bucket {buckets[1]}"] = "Mid"
        
    df_out = pd.DataFrame(stats)
    df_out["Portfolio"] = df_out["Portfolio"].replace(label_map)
    return df_out.set_index("Portfolio")

def generate_characteristics_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()
    
    if UNDERLYING_COL not in df.columns:
        print(f"WARNING: '{UNDERLYING_COL}' not found. Gamma will not be scaled.")
        S = 1.0
    else:
        S = df[UNDERLYING_COL]

    # --- HYBRID SCALING STRATEGY ---
    
    # 1. Gamma: SCALE IT. 
    # Raw Gamma is tiny (e.g. 0.00001). Scaling by S/100 gives "Delta change per 1% move".
    # This makes the number readable (e.g., 0.05).
    if "gamma_model" in df.columns:
        df["Gamma (1%)"] = df["gamma_model"] * S / 100.0
    
    # 2. Vega & Theta: KEEP RAW.
    # Raw Vega/Theta are usually readable (e.g., 2000, -25).
    # Scaling them by S makes them 0.00. We keep them raw for the table.
    if "vega_model" in df.columns:
        df["Vega"] = df["vega_model"]
        
    if "theta_model" in df.columns:
        df["Theta"] = df["theta_model"]
        
    if "delta_model" in df.columns:
        df["Delta"] = df["delta_model"].abs()

    def agg_chars(g):
        n_contracts = g.groupby("eval_date").size().sum()
        mny = g["moneyness_std"].abs().mean() if "moneyness_std" in g.columns else np.nan
        
        sign_pred = np.sign(g[PRED_COL])
        sign_real = np.sign(g[RET_COL])
        hitrate = (sign_pred == sign_real).mean()
        
        pct_call = g["is_call"].mean() if "is_call" in g.columns else np.nan
        spread = g["opt_rel_spread_final"].mean() if "opt_rel_spread_final" in g.columns else np.nan
        
        res = {
            "n": n_contracts,
            "|m|": mny,
            "Hit ratio": hitrate,
            "% Calls": pct_call,
            "Spread": spread,
        }
        
        # Add Greeks
        for col in ["Delta", "Gamma (1%)", "Vega", "Theta"]:
            if col in g.columns:
                res[col] = g[col].mean()
        return pd.Series(res)

    df_out = df.groupby("bucket").apply(agg_chars)
    
    buckets = sorted(merged_df["bucket"].unique())
    label_map = {buckets[0]: "Low", buckets[-1]: "High"}
    if len(buckets) == 3:
        label_map[buckets[1]] = "Mid"
    df_out.index = df_out.index.map(lambda x: label_map.get(x, f"B{x}"))
    
    # Order: Note "Gamma (1%)" is the new name
    desired_order = ["n", "|m|", "Hit ratio", "% Calls", "Spread", "Delta", "Gamma (1%)", "Vega", "Theta"]
    available_cols = [c for c in desired_order if c in df_out.columns]
    return df_out[available_cols]

# =========================
# Main Logic
# =========================

def process_subset(subset_df, label, output_dir):
    print(f"Processing subset: {label} ({len(subset_df)} rows)...")
    sub_dir = output_dir / label
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    subset_df = assign_quantile_buckets(subset_df, "eval_date", PRED_COL, N_BUCKETS, "bucket")
    port_daily = compute_daily_portfolio_returns(subset_df, "eval_date", "bucket", RET_COL, WEIGHT_COL)
    hl_daily = compute_hl_series(port_daily, "eval_date", "bucket")
    
    perf_table = generate_performance_table(subset_df, port_daily, hl_daily)
    perf_table.to_csv(sub_dir / "Table1_Performance.csv", float_format="%.2f")
    
    char_table = generate_characteristics_table(subset_df)
    char_table.to_csv(sub_dir / "Table2_Characteristics.csv", float_format="%.2f")
    print(f"Saved tables for {label}")

def main():
    print("Loading main data...")
    diag = pd.read_csv(DIAG_PATH)
    feats = pd.read_parquet(FEATURES_PATH)

    print(f"Loading futures data from {FUTURES_PATH}...")
    futures = pd.read_csv(FUTURES_PATH)
    
    # Parse dates
    futures["date_parsed"] = pd.to_datetime(futures["date"], format="%b %d %Y")
    futures["date_key"] = futures["date_parsed"].dt.date
    futures_clean = futures[["date_key", "adj close"]].copy()

    diag["date_key"] = pd.to_datetime(diag["group"]).dt.date
    feats[DATE_COL] = pd.to_datetime(feats[DATE_COL])
    feats["date_key"] = feats[DATE_COL].dt.date

    merged = pd.merge(
        feats, diag,
        left_on=["date_key", "y_price"],
        right_on=["date_key", "y_target"],
        how="inner", suffixes=("", "_diag")
    )

    merged = pd.merge(merged, futures_clean, on="date_key", how="left")
    
    missing_px = merged[UNDERLYING_COL].isnull().sum()
    if missing_px > 0:
        print(f"Warning: {missing_px} rows missing futures price. Gamma scaling will be off for those.")

    merged["eval_date"] = merged["date_key"]
    merged[WEIGHT_COL] = 1.0
    merged = merged[np.isfinite(merged[PRED_COL]) & np.isfinite(merged[RET_COL])].copy()
    
    subsets = {
        "Ultra-Short": merged[(merged["dte"] >= 1) & (merged["dte"] <= 7)].copy(),
        "Short": merged[(merged["dte"] >= 8) & (merged["dte"] <= 31)].copy(),
        "All": merged.copy()
    }
    
    for label, df in subsets.items():
        if not df.empty:
            process_subset(df, label, OUTPUT_DIR)

if __name__ == "__main__":
    main()