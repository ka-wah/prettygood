import os
from math import erf, sqrt

import numpy as np
import pandas as pd

# --- user settings ---------------------------------------------------------
# root directory where the results live
BASE_DIR = r"C:\Users\kawah\Downloads\results\results-mac\y_price\all-all"

# mapping: x-axis label -> folder name
INFO_SETS = [
    ("I", "I"),
    ("IB", "IB"),
    ("IBM", "IBM"),
    ("IBMC", "IBMC"),
    ("full", "IBMCTINTERACTIONS"),   # treat this as the full information set
]

# IMPORTANT: must match the yhat_* columns in diag_all.csv, e.g. yhat_N-En, yhat_N-En#1, ...
TARGET_MODEL = "NL-En"

DIAG_FILE = "diag_all.csv"
Y_COL = "y_target"  # or "y_raw" if you prefer per-dollar return


# --- helpers ---------------------------------------------------------------
def dm_newey_west_from_d(d, max_lags=None):
    """
    Compute DM t-stat with Newey-West long-run variance for a 1D series d_t.
    d_t should be the cross-sectional mean loss differential per day.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 3:
        return np.nan

    d_mean = d.mean()
    d_c = d - d_mean

    if max_lags is None:
        max_lags = int(np.floor(T ** (1 / 3)))
        max_lags = max(1, min(max_lags, T - 1))

    gamma0 = np.sum(d_c * d_c) / T
    gammas = []
    for k in range(1, max_lags + 1):
        gam_k = np.sum(d_c[k:] * d_c[:-k]) / T
        gammas.append(gam_k)

    weights = [1 - k / (max_lags + 1) for k in range(1, max_lags + 1)]
    nw_var = gamma0 + 2.0 * np.sum(w * g for w, g in zip(weights, gammas))
    if nw_var <= 0:
        # fallback to plain variance if NW variance is non-positive due to numerical issues
        nw_var = np.var(d, ddof=1)

    dm_t = d_mean / np.sqrt(nw_var / T)
    return float(dm_t)


def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def star_code(p):
    if pd.isna(p):
        return ""
    if p <= 0.01:
        return "***"
    elif p <= 0.05:
        return "**"
    elif p <= 0.10:
        return "*"
    else:
        return ""


def load_forecasts(label, folder):
    """
    Load forecasts for the target model from a given info-set folder.
    Returns DataFrame with columns [group, Y_COL, forecast] or None if missing.
    """
    path = os.path.join(BASE_DIR, folder, DIAG_FILE)
    if not os.path.exists(path):
        print(f"# WARNING: missing file, skipping: {path}")
        return None

    df = pd.read_csv(path)
    if Y_COL not in df.columns:
        print(f"# WARNING: {Y_COL} not found in {path}; skipping")
        return None

    # We expect columns like yhat_N-En#1, yhat_N-En#2, or yhat_N-En
    base_model = TARGET_MODEL.split("#")[0]

    # primary: ensemble over yhat_{base_model}#k if present
    pred_cols = [c for c in df.columns if c.startswith(f"yhat_{base_model}#")]

    if not pred_cols:
        exact = f"yhat_{TARGET_MODEL}"
        no_num = f"yhat_{base_model}"
        if exact in df.columns:
            pred_cols = [exact]
        elif no_num in df.columns:
            pred_cols = [no_num]

    if not pred_cols:
        print(f"# WARNING: model {TARGET_MODEL} not found in {path}; skipping")
        return None

    df["forecast"] = df[pred_cols].mean(axis=1)
    return df[["group", Y_COL, "forecast"]]


# --- load forecasts for each information set ------------------------------
data = {}
for label, folder in INFO_SETS:
    forecasts = load_forecasts(label, folder)
    if forecasts is not None:
        data[label] = forecasts

if len(data) < 2:
    raise RuntimeError("Need at least two information sets with data to run DM and correlations.")

# align on common groups (e.g. dates)
common_groups = set.intersection(*(set(df["group"]) for df in data.values()))
if not common_groups:
    raise RuntimeError("No overlapping groups across information sets.")
common_groups = sorted(common_groups)

# restrict each DataFrame to common groups only
for label in list(data.keys()):
    df = data[label]
    data[label] = df[df["group"].isin(common_groups)].copy()

# group by group identifier for per-day cross sections
grouped = {
    label: {g: sub for g, sub in df.groupby("group")}
    for label, df in data.items()
}

labels = list(data.keys())


# --- Diebold-Mariano across info sets -------------------------------------
dm_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
p_df = pd.DataFrame(index=labels, columns=labels, dtype=float)
dm_with_stars = pd.DataFrame(index=labels, columns=labels, dtype=object)

for i in labels:
    for j in labels:
        if i == j:
            dm_df.loc[i, j] = np.nan
            p_df.loc[i, j] = np.nan
            dm_with_stars.loc[i, j] = ""
            continue

        d_list = []
        for g in common_groups:
            sub_i = grouped[i][g]
            sub_j = grouped[j][g]

            y_i = sub_i[Y_COL].to_numpy()
            y_j = sub_j[Y_COL].to_numpy()
            yhat_i = sub_i["forecast"].to_numpy()
            yhat_j = sub_j["forecast"].to_numpy()

            # align by row position within group; this assumes consistent ordering
            n = min(len(y_i), len(y_j))
            if n == 0:
                continue

            # cross-sectional mean difference in squared errors for this day
            d_t = np.mean((y_i[:n] - yhat_i[:n]) ** 2 - (y_j[:n] - yhat_j[:n]) ** 2)
            d_list.append(d_t)

        t_stat = dm_newey_west_from_d(np.asarray(d_list), max_lags=None)
        dm_df.loc[i, j] = t_stat

        if np.isnan(t_stat):
            p = np.nan
        else:
            p = 2.0 * (1.0 - norm_cdf(abs(t_stat)))
        p_df.loc[i, j] = p

        s = star_code(p)
        dm_with_stars.loc[i, j] = f"{t_stat:.2f}{s}" if not np.isnan(t_stat) else ""

# --- Bali-style forecast error correlations (no daily averaging) ----------
# Build, for each info set, a single stacked vector of forecast errors
# over the common evaluation sample, aligned across information sets.

# First, determine for each group the minimum cross-sectional size across all info sets
min_len_per_group = {}
for g in common_groups:
    lengths = [len(grouped[label][g]) for label in labels]
    n_g = min(lengths)
    if n_g > 0:
        min_len_per_group[g] = n_g

if not min_len_per_group:
    raise RuntimeError("No groups with positive cross-sectional size across all information sets.")

# Optionally, restrict common_groups to those with positive min length
common_groups = sorted(min_len_per_group.keys())

# Build stacked forecast error vectors per information set
errors = {}
for label in labels:
    parts = []
    for g in common_groups:
        sub = grouped[label][g]
        n_g = min_len_per_group[g]

        y = sub[Y_COL].to_numpy()[:n_g]
        yhat = sub["forecast"].to_numpy()[:n_g]
        parts.append(y - yhat)  # forecast error

    if parts:
        errors[label] = np.concatenate(parts)
    else:
        errors[label] = np.array([])

# Sanity: all non-empty error vectors should have equal length now
non_empty_lengths = {label: len(vec) for label, vec in errors.items() if len(vec) > 0}
if len(set(non_empty_lengths.values())) > 1:
    print("# WARNING: error vectors have differing lengths; correlation matrix may be inconsistent.")

# Compute pairwise correlations
fe_corr = pd.DataFrame(index=labels, columns=labels, dtype=float)

for i in labels:
    for j in labels:
        e_i = errors[i]
        e_j = errors[j]

        if i == j:
            if len(e_i) == 0:
                fe_corr.loc[i, j] = np.nan
            else:
                fe_corr.loc[i, j] = 1.0
            continue

        if len(e_i) == 0 or len(e_j) == 0:
            fe_corr.loc[i, j] = np.nan
            continue

        std_i = e_i.std(ddof=1)
        std_j = e_j.std(ddof=1)
        if std_i == 0 or std_j == 0:
            fe_corr.loc[i, j] = np.nan
            continue

        fe_corr.loc[i, j] = np.corrcoef(e_i, e_j)[0, 1]


# --- save outputs ----------------------------------------------------------
dm_df.to_csv("figs/outputs/infosets_dm_tstats.csv", float_format="%.4f")
p_df.to_csv("figs/outputs/infosets_dm_pvalues.csv", float_format="%.4g")
dm_with_stars.to_csv("figs/outputs/infosets_dm_tstats_with_stars.csv")
fe_corr.to_csv("figs/outputs/infosets_forecast_error_correlations.csv", float_format="%.4f")

print("Saved DM matrices and Bali-style forecast error correlations for info sets.")
