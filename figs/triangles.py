import numpy as np
import pandas as pd
from math import erf, sqrt

# ------------------------------------------------------------
# 1. Newey–West DM for a given d_t series
# ------------------------------------------------------------
def dm_newey_west_from_d(d, max_lags=None):
    """
    d: 1D array-like, time series of cross-sectional loss differentials
       d_t = avg[(e_row^2 - e_col^2)] so that DM>0 -> column better than row.
    Returns DM t-stat with NW long-run variance.
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 3:
        return np.nan

    d_mean = d.mean()
    d_c = d - d_mean

    if max_lags is None:
        # Bali don't give an explicit lag here; use a simple NW choice
        max_lags = int(np.floor(T ** (1/3)))
        max_lags = max(1, min(max_lags, T - 1))

    # gamma_0
    gamma0 = np.sum(d_c * d_c) / T

    # gamma_k and Bartlett weights
    gammas = []
    for k in range(1, max_lags + 1):
        gam_k = np.sum(d_c[k:] * d_c[:-k]) / T
        gammas.append(gam_k)

    weights = [1 - k / (max_lags + 1) for k in range(1, max_lags + 1)]
    nw_var = gamma0 + 2.0 * np.sum(w * g for w, g in zip(weights, gammas))

    # fallback if numerical issues
    if nw_var <= 0:
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


# ------------------------------------------------------------
# 2. Load data and set up models
# ------------------------------------------------------------
df = pd.read_csv(r'C:\Users\kawah\Documents\bitcoining\big_results\all-all\IBMCINTERACTIONS\diag_all.csv', parse_dates=["group"])

# choose target: "y_target" (your y_vega) or "y_raw" (per-dollar return)
y_col = "y_target"

model_cols = {
    "Ridge": ["yhat_ridge#1", "yhat_ridge#2"],
    "Lasso": ["yhat_lasso#1", "yhat_lasso#2"],
    "ENet":  ["yhat_elasticnet#1", "yhat_elasticnet#2"],
    "PCR":   ["yhat_pcr#1", "yhat_pcr#2"],
    "PLS":   ["yhat_pls#1", "yhat_pls#2"],
    "L-En":  ["yhat_L-En"],
    "GBR":   ["yhat_lgbm_gbdt#1", "yhat_lgbm_gbdt#2"],
    "RF":    ["yhat_rf#1", "yhat_rf#2"],
    "Dart":  ["yhat_lgbm_dart#1", "yhat_lgbm_dart#2"],
    "FFN":   ["yhat_ffn#1", "yhat_ffn#2"],
    "N-En":  ["yhat_N-En"],
}

models = ["Ridge", "Lasso", "ENet", "PCR", "PLS",
          "L-En", "GBR", "RF", "Dart", "FFN", "N-En"]

# average across #1/#2 variants per model
for m, cols in model_cols.items():
    df[m] = df[cols].mean(axis=1)

# group by day (cross-section within each day)
groups = list(df.groupby("group"))  # list of (date, subdf)


# ------------------------------------------------------------
# 3. Build DM matrix using cross-sectional loss differentials
# ------------------------------------------------------------
dm_df = pd.DataFrame(index=models, columns=models, dtype=float)
p_df = pd.DataFrame(index=models, columns=models, dtype=float)
dm_with_stars = pd.DataFrame(index=models, columns=models, dtype=object)

# full-length forecast errors for correlations (same test set length for all models)
errors_full = {m: (df[y_col] - df[m]).to_numpy() for m in models}

for i in models:      # row model
    for j in models:  # column model
        if i == j:
            dm_df.loc[i, j] = np.nan
            p_df.loc[i, j] = np.nan
            dm_with_stars.loc[i, j] = ""
            continue

        # build time series d_t^{(i,j)} = avg[(e_i^2 - e_j^2)] over days
        d_list = []
        for _, sub in groups:
            y = sub[y_col].to_numpy()
            e_i = y - sub[i].to_numpy()
            e_j = y - sub[j].to_numpy()
            # cross-sectional mean difference in squared errors
            d_t = np.mean(e_i**2 - e_j**2)
            d_list.append(d_t)

        d_arr = np.asarray(d_list, dtype=float)
        t_stat = dm_newey_west_from_d(d_arr, max_lags=6)
        dm_df.loc[i, j] = t_stat

        # two-sided p-value (Bali-style)
        if np.isnan(t_stat):
            p = np.nan
        else:
            p = 2.0 * (1.0 - norm_cdf(abs(t_stat)))
        p_df.loc[i, j] = p

        # attach stars only based on significance; interpretation still uses sign
        s = star_code(p)
        dm_with_stars.loc[i, j] = f"{t_stat:.2f}{s}" if not np.isnan(t_stat) else ""

# forecast error correlations using all individual errors (no daily averaging)
err_df = pd.DataFrame(errors_full)
err_corr = err_df.corr()

# ------------------------------------------------------------
# 4. Save matrices for Excel
# ------------------------------------------------------------
dm_df.to_csv("big-figs/trdm_crosssec_neweywest_tstats.csv", float_format="%.4f")
p_df.to_csv("big-figs/trdm_crosssec_neweywest_pvalues.csv", float_format="%.4g")
dm_with_stars.to_csv("big-figs/trdm_crosssec_neweywest_tstats_with_stars.csv")
err_corr.to_csv("big-figs/trforecast_error_correlations.csv", float_format="%.4f")
