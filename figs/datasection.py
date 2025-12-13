"""
Data-section tables and figures generator.

Usage option 1 (standalone script):
    - Set CSV_PATH and OUT_DIR at the bottom of this file.
    - Run the script (e.g. from your IDE).

Usage option 2 (from a notebook):
    from make_data_tables_figures import run_all
    run_all("features_and_dh_returnsclean.parquet", "output_data_section")
"""

import os
from typing import Sequence, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Non-interactive backend so it works in scripts
plt.switch_backend("Agg")


def classify_moneyness_bucket(df: pd.DataFrame,
                              m_col: str = "moneyness_std",
                              is_call_col: str = "is_call") -> pd.Series:
    """
    Classify contracts into ITM/ATM/OTM based on standardised moneyness m*
    and option side, using:

        ATM: |m*| <= 1
        Calls: ITM if m* < -1, OTM if m* > 1
        Puts : ITM if m* >  1, OTM if m* < -1
    """
    if m_col not in df.columns:
        raise KeyError(f"Column '{m_col}' not found in DataFrame.")
    if is_call_col not in df.columns:
        raise KeyError(f"Column '{is_call_col}' not found in DataFrame.")

    m = df[m_col].astype(float)
    is_call = df[is_call_col].astype(int)

    bucket = pd.Series(index=df.index, dtype="object")

    # ATM first
    atm_mask = m.abs() <= 1
    bucket.loc[atm_mask] = "ATM"

    # Calls
    call_mask = is_call == 1
    bucket.loc[call_mask & (m < -1)] = "ITM"
    bucket.loc[call_mask & (m > 1)] = "OTM"

    # Puts
    put_mask = is_call == 0
    bucket.loc[put_mask & (m > 1)] = "ITM"
    bucket.loc[put_mask & (m < -1)] = "OTM"

    return bucket


def make_side_moneyness_table(df: pd.DataFrame,
                              out_dir: str,
                              m_bucket_col: str = "m_bucket",
                              side_col: str = "side") -> pd.DataFrame:
    """
    Cross-tab of option side (Call/Put) and moneyness bucket (ITM/ATM/OTM)
    with row/column totals (Table 1 style).
    """
    if m_bucket_col not in df.columns:
        raise KeyError(f"Column '{m_bucket_col}' not found in DataFrame.")
    if side_col not in df.columns:
        raise KeyError(f"Column '{side_col}' not found in DataFrame.")

    ct = pd.crosstab(df[side_col], df[m_bucket_col])

    # Standard ordering (will leave as NaN if bucket is missing)
    ct = ct.reindex(index=["Call", "Put"])
    ct = ct.reindex(columns=["ITM", "ATM", "OTM"])

    # Replace NaN with zeros before integer formatting
    ct = ct.fillna(0).astype(int)

    ct["All"] = ct.sum(axis=1)
    total_row = ct.sum(axis=0).to_frame().T
    total_row.index = ["All"]
    ct = pd.concat([ct, total_row], axis=0)

    # Save LaTeX
    table_path = os.path.join(out_dir, "table_side_moneyness.tex")
    ct.to_latex(
        table_path,
        index=True,
        na_rep="",
        formatters={c: "{:d}".format for c in ct.columns},
    )
    return ct


def make_dh_return_moments_table(df: pd.DataFrame,
                                 out_dir: str,
                                 ret_col: str = "dh_ret") -> pd.DataFrame:
    """
    Summary moments for delta-hedged returns: mean, std, quantiles, skew, kurtosis, JB.
    """
    if ret_col not in df.columns:
        raise KeyError(f"Column '{ret_col}' not found in DataFrame.")

    r = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    n = r.shape[0]
    if n == 0:
        raise ValueError("No valid delta-hedged return observations found.")

    mean = r.mean()
    std = r.std(ddof=1)
    q01, q05, q50, q95, q99 = r.quantile([0.01, 0.05, 0.5, 0.95, 0.99])

    # pandas skew/kurt are sample versions; kurt is excess by default
    skew = r.skew()
    ex_kurt = r.kurt()

    # Jarque–Bera statistic (no scipy dependency)
    jb_stat = n / 6.0 * (skew ** 2 + (ex_kurt ** 2) / 4.0)

    moments = pd.DataFrame(
        {
            "N": [n],
            "Mean": [mean],
            "Std": [std],
            "p1": [q01],
            "p5": [q05],
            "p50": [q50],
            "p95": [q95],
            "p99": [q99],
            "Skewness": [skew],
            "Excess kurtosis": [ex_kurt],
            "JB": [jb_stat],
        }
    )

    table_path = os.path.join(out_dir, "table_dh_returns_moments.tex")
    moments.to_latex(table_path, index=False, float_format="%.4f")
    return moments


def make_tail_mass_table(df: pd.DataFrame,
                         out_dir: str,
                         ret_col: str = "dh_ret",
                         thresholds: Sequence[float] = (2.0, 3.0, 4.0, 5.0, 8.0, 10.0)
                         ) -> pd.DataFrame:
    """
    Fraction of observations beyond several |r^Δ| thresholds (tail mass).
    """
    if ret_col not in df.columns:
        raise KeyError(f"Column '{ret_col}' not found in DataFrame.")

    r = pd.to_numeric(df[ret_col], errors="coerce").dropna()
    if r.empty:
        raise ValueError("No valid delta-hedged return observations found.")

    rows = []
    for thr in thresholds:
        mask = r.abs() > thr
        frac = mask.mean()
        rows.append({"|r^Δ| > x": thr, "Fraction": frac, "Count": int(mask.sum())})

    tail_df = pd.DataFrame(rows)

    table_path = os.path.join(out_dir, "table_dh_returns_tailmass.tex")
    tail_df.to_latex(table_path, index=False, float_format="%.4f")
    return tail_df


def make_contract_characteristics_table(df: pd.DataFrame,
                                        out_dir: str,
                                        cols: Dict[str, str] = None) -> pd.DataFrame:
    """
    Summary statistics for key option characteristics: DTE, moneyness, IV, |delta|.
    `cols` maps logical names to column names in df.
    """
    if cols is None:
        cols = {
            "DTE": "dte",
            "Std. moneyness": "moneyness_std",
            "Implied vol": "iv",
            "Abs. delta": "abs_delta",
        }

    rows = []
    for label, col in cols.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' for '{label}' not found in DataFrame.")

        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if x.empty:
            raise ValueError(f"No valid observations for '{label}' (column '{col}').")

        n = x.shape[0]
        mean = x.mean()
        std = x.std(ddof=1)
        q01, q05, q50, q95, q99 = x.quantile([0.01, 0.05, 0.5, 0.95, 0.99])

        rows.append({
            "Variable": label,
            "N": n,
            "Mean": mean,
            "Std": std,
            "p1": q01,
            "p5": q05,
            "p50": q50,
            "p95": q95,
            "p99": q99,
        })

    stats_df = pd.DataFrame(rows)

    table_path = os.path.join(out_dir, "table_contract_characteristics.tex")
    stats_df.to_latex(table_path, index=False, float_format="%.4f")
    stats_df.to_csv(os.path.join(out_dir, "table_contract_characteristics.csv"), index=False)

    return stats_df


def plot_moneyness_hist(df: pd.DataFrame,
                        out_dir: str,
                        m_col: str = "moneyness_std") -> None:
    """
    Histogram of standardised moneyness m*.
    """
    if m_col not in df.columns:
        raise KeyError(f"Column '{m_col}' not found in DataFrame.")

    m = pd.to_numeric(df[m_col], errors="coerce").dropna()
    if m.empty:
        raise ValueError("No valid moneyness observations found.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(m, bins=50, density=True, edgecolor="black", alpha=0.7)
    ax.set_xlabel(r"Standardised moneyness $m_t^\ast$")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of standardised moneyness")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.axvline(1.0, linestyle=":", linewidth=1)
    ax.axvline(-1.0, linestyle=":", linewidth=1)
    ax.set_xlim(-10, 10)
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "fig_moneyness_distribution.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def plot_dte_hist(df: pd.DataFrame,
                  out_dir: str,
                  dte_col: str = "dte") -> None:
    """
    Histogram of days-to-expiry (DTE) in the final sample.
    """
    if dte_col not in df.columns:
        raise KeyError(f"Column '{dte_col}' not found in DataFrame.")

    dte = pd.to_numeric(df[dte_col], errors="coerce").dropna()
    if dte.empty:
        raise ValueError("No valid DTE observations found.")

    fig, ax = plt.subplots(figsize=(6, 4))
    # Integer bins centred on days
    bins = np.arange(dte.min() - 0.5, dte.max() + 1.5, 1.0)
    ax.hist(dte, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Days to expiry (DTE)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of days to expiry in the final sample")
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "fig_dte_distribution.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def plot_contract_counts_per_day(df: pd.DataFrame,
                                 out_dir: str,
                                 date_col: str = "date") -> None:
    """
    Time-series of the number of contracts per day.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")

    counts = df.groupby(date_col).size().sort_index()
    if counts.empty:
        raise ValueError("No observations to aggregate by date.")

    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.asarray(mdates.date2num(counts.index.to_pydatetime()), dtype="float64")
    y = np.asarray(counts.values, dtype="float64")
    ax.plot(x, y, marker="o", linewidth=1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of contracts")
    ax.set_title("Number of contracts per day")
    ax.xaxis_date()
    fig.autofmt_xdate()
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "fig_contracts_per_day.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def plot_contract_counts_hist(df: pd.DataFrame,
                              out_dir: str,
                              date_col: str = "date") -> None:
    """
    Histogram of the number of contracts per day (cross-sectional size).
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")

    counts = df.groupby(date_col).size()
    if counts.empty:
        raise ValueError("No observations to aggregate by date.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts.values, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of contracts per day")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of daily cross-sectional size")
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "fig_contracts_per_day_hist.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def plot_daily_median_iqr(df: pd.DataFrame,
                          out_dir: str,
                          ret_col: str = "dh_ret",
                          date_col: str = "date") -> None:
    """
    Daily cross-sectional median and IQR of delta-hedged returns (pooled sample).
    """
    if ret_col not in df.columns:
        raise KeyError(f"Column '{ret_col}' not found in DataFrame.")
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")

    agg = df.groupby(date_col)[ret_col].agg(
        median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    )
    tmp = agg.reset_index()
    if tmp.empty:
        raise ValueError("No daily return aggregates computed.")

    x = np.asarray(mdates.date2num(tmp[date_col].dt.to_pydatetime()), dtype="float64")
    median = np.asarray(tmp["median"].values, dtype="float64")
    p25 = np.asarray(tmp["p25"].values, dtype="float64")
    p75 = np.asarray(tmp["p75"].values, dtype="float64")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, median, linewidth=1.2)
    ax.fill_between(x, p25, p75, alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel(r"Daily cross-sectional $r^\Delta$ (median / IQR)")
    ax.set_title("Daily median and interquartile range of delta-hedged returns")
    ax.xaxis_date()
    fig.autofmt_xdate()
    fig.tight_layout()

    fig_path = os.path.join(out_dir, "fig_dh_ret_daily_median_iqr.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def plot_daily_median_iqr_by_group(df: pd.DataFrame,
                                   out_dir: str,
                                   group_col: str,
                                   ret_col: str = "dh_ret",
                                   date_col: str = "date",
                                   prefix: str = "dte") -> None:
    """
    Daily median/IQR of delta-hedged returns, stratified by a discrete group
    such as DTE or moneyness bucket. One file per group.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")
    if ret_col not in df.columns:
        raise KeyError(f"Column '{ret_col}' not found in DataFrame.")
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")

    for g, dfg in df.groupby(group_col):
        if pd.isna(g):
            continue

        agg = dfg.groupby(date_col)[ret_col].agg(
            median="median",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
        )
        tmp = agg.reset_index()

        if tmp.empty:
            continue

        x = np.asarray(mdates.date2num(tmp[date_col].dt.to_pydatetime()), dtype="float64")
        median = np.asarray(tmp["median"].values, dtype="float64")
        p25 = np.asarray(tmp["p25"].values, dtype="float64")
        p75 = np.asarray(tmp["p75"].values, dtype="float64")

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(x, median, linewidth=1.2)
        ax.fill_between(x, p25, p75, alpha=0.3)
        ax.set_xlabel("Date")
        ax.set_ylabel(r"Daily $r^\Delta$ (median / IQR)")
        ax.set_title(f"Daily median/IQR of delta-hedged returns – {group_col} = {g}")
        ax.xaxis_date()
        fig.autofmt_xdate()
        fig.tight_layout()

        safe_g = str(g).replace(" ", "_")
        fig_path = os.path.join(out_dir, f"fig_dh_ret_daily_median_iqr_{prefix}_{safe_g}.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)


def run_all(csv_path: str, out_dir: str) -> None:
    """
    High-level function: read data, build all tables and figures needed for data section.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Despite name csv_path this is a parquet file in your setup
    df = pd.read_parquet(csv_path)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise KeyError("Expected a 'date' column in the data.")

    # Side indicator
    if "is_call" not in df.columns:
        raise KeyError("Expected an 'is_call' column (1=call,0=put).")
    df["side"] = np.where(df["is_call"] == 1, "Call", "Put")

    # Standardised moneyness buckets
    df["m_bucket"] = classify_moneyness_bucket(df)

    # === Tables ===
    side_mny_tab = make_side_moneyness_table(df, out_dir)
    dh_moments_tab = make_dh_return_moments_table(df, out_dir)
    tailmass_tab = make_tail_mass_table(df, out_dir)
    char_stats_tab = make_contract_characteristics_table(df, out_dir)

    # Also export as CSV (optional)
    side_mny_tab.to_csv(os.path.join(out_dir, "table_side_moneyness.csv"))
    dh_moments_tab.to_csv(os.path.join(out_dir, "table_dh_returns_moments.csv"), index=False)
    tailmass_tab.to_csv(os.path.join(out_dir, "table_dh_returns_tailmass.csv"), index=False)
    # char_stats_tab already exports CSV inside the function

    # === Figures ===
    plot_moneyness_hist(df, out_dir)
    plot_contract_counts_per_day(df, out_dir)
    plot_contract_counts_hist(df, out_dir)

    if "dte" in df.columns:
        plot_dte_hist(df, out_dir)

    plot_daily_median_iqr(df, out_dir)

    # Stratified daily median/IQR by DTE and by moneyness bucket
    if "dte" in df.columns:
        # Note: if dte takes many distinct values this produces many files.
        # If you want bins instead, discretise dte first.
        plot_daily_median_iqr_by_group(df, out_dir, group_col="dte", prefix="dte")

    plot_daily_median_iqr_by_group(df, out_dir, group_col="m_bucket", prefix="mny_bucket")


# --------- EDIT THESE LINES AND RUN THE FILE (no command line needed) ---------
if __name__ == "__main__":
    # Path to your parquet file
    CSV_PATH = "inputs/short.parquet"
    # Output directory where tables/figures will be saved
    OUT_DIR = "figs/output/data"

    run_all(CSV_PATH, OUT_DIR)
