import pandas as pd
import numpy as np
import os
import sys

# ---------------------------------------------------------
# 1. Metric Functions (As provided)
# ---------------------------------------------------------

def r2_os(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard unweighted R2_OS (1 - MSE_model / MSE_zero)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0: return np.nan
    
    mse_model = np.mean((y_true - y_pred)**2)
    mse_naive = np.mean(y_true**2)
    
    if mse_naive == 0: return np.nan
    return 1.0 - mse_model / mse_naive

def r2_os_weighted(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Calculates R2_OS where errors are weighted by 'weights' (e.g. price)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(weights, dtype=float)
    
    good = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w) & (w > 0)
    y_true, y_pred, w = y_true[good], y_pred[good], w[good]
    
    if w.sum() == 0: return np.nan
    
    # Weighted MSE of Model
    mse_model = np.average((y_true - y_pred)**2, weights=w)
    
    # Weighted MSE of Naive Prediction (Zero Return)
    mse_naive = np.average(y_true**2, weights=w)
    
    if mse_naive == 0: return np.nan
    return 1.0 - mse_model / mse_naive

def r2_os_xs_weighted(y_true: np.ndarray, y_pred: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> float:
    """Calculates Cross-Sectional R2 (R2_XS) with sample weights."""
    df = pd.DataFrame({'y': y_true, 'yhat': y_pred, 'g': ids, 'w': weights})
    df = df[np.isfinite(df['y']) & np.isfinite(df['yhat']) & np.isfinite(df['w']) & (df['w'] > 0)]
    if df.empty: return np.nan
    
    # 1. Calculate weighted means per group
    df['wy'] = df['w'] * df['y']
    df['wyhat'] = df['w'] * df['yhat']
    
    g = df.groupby('g')
    sums = g[['wy', 'wyhat', 'w']].sum()
    
    # Avoid division by zero
    sums = sums[sums['w'] > 0]
    sums['y_bar'] = sums['wy'] / sums['w']
    sums['yhat_bar'] = sums['wyhat'] / sums['w']
    
    # Map back
    df['y_bar'] = df['g'].map(sums['y_bar'])
    df['yhat_bar'] = df['g'].map(sums['yhat_bar'])
    
    # Drop rows where group sum(w) was zero
    df = df.dropna(subset=['y_bar', 'yhat_bar'])
    
    # 2. Center Data
    df['y_cs'] = df['y'] - df['y_bar']
    df['yhat_cs'] = df['yhat'] - df['yhat_bar']
    
    # 3. Weighted R2 on centered data
    num = np.average((df['y_cs'] - df['yhat_cs'])**2, weights=df['w'])
    den = np.average(df['y_cs']**2, weights=df['w'])
    
    if den == 0: return np.nan
    return 1.0 - num/den

def clark_west_t_weighted(y_true: np.ndarray, y_pred: np.ndarray, ids: np.ndarray, weights: np.ndarray, hac_lags: int = 5) -> float:
    """Calculates CW_t on Weighted Data (Time Series)."""
    y = np.asarray(y_true, dtype=float)
    ya = np.asarray(y_pred, dtype=float)
    w = np.asarray(weights, dtype=float)
    ids = np.asarray(ids)
    
    mask = np.isfinite(y) & np.isfinite(ya) & np.isfinite(w) & (w > 0)
    y, ya, w, ids = y[mask], ya[mask], w[mask], ids[mask]
    
    if len(y) == 0: return np.nan

    # 1. Difference in Squared Errors (Null=0 vs Alt=Ya)
    d_row = y**2 - (y - ya)**2
    
    # 2. Aggregate to Daily Weighted Mean
    df = pd.DataFrame({'d': d_row, 'w': w, 'g': ids})
    
    df['dw'] = df['d'] * df['w']
    g = df.groupby('g')
    sums = g[['dw', 'w']].sum()
    
    sums = sums[sums['w'] > 0]
    daily_d = sums['dw'] / sums['w']
    d_vals = daily_d.values
    
    if len(d_vals) < 2: return np.nan

    # 3. HAC T-Test
    try:
        import statsmodels.api as sm
        X = np.ones((len(d_vals), 1))
        # Handle cases with extremely small sample size
        lags = min(hac_lags, len(d_vals) // 2)
        res = sm.OLS(d_vals, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
        return float(res.tvalues[0])
    except Exception:
        return float(np.mean(d_vals) / (np.std(d_vals, ddof=1) / np.sqrt(len(d_vals))))

def clark_west_t_xs_weighted(y_true: np.ndarray, y_pred: np.ndarray, ids: np.ndarray, weights: np.ndarray, hac_lags: int = 5) -> float:
    """Calculates CW_t on Weighted Cross-Sectional Data."""
    df = pd.DataFrame({'y': y_true, 'yhat': y_pred, 'g': ids, 'w': weights})
    df = df[np.isfinite(df['y']) & np.isfinite(df['yhat']) & np.isfinite(df['w']) & (df['w'] > 0)]
    if df.empty: return np.nan
    
    # 1. Calculate weighted means per group
    df['wy'] = df['w'] * df['y']
    df['wyhat'] = df['w'] * df['yhat']
    
    g = df.groupby('g')
    sums = g[['wy', 'wyhat', 'w']].sum()
    
    sums = sums[sums['w'] > 0]
    sums['y_bar'] = sums['wy'] / sums['w']
    sums['yhat_bar'] = sums['wyhat'] / sums['w']
    
    df['y_bar'] = df['g'].map(sums['y_bar'])
    df['yhat_bar'] = df['g'].map(sums['yhat_bar'])
    df = df.dropna(subset=['y_bar', 'yhat_bar'])
    
    # 2. Center Data
    df['y_cs'] = df['y'] - df['y_bar']
    df['yhat_cs'] = df['yhat'] - df['yhat_bar']
    
    # 3. Calculate Difference in Squared Errors (XS)
    df['d_row'] = df['y_cs']**2 - (df['y_cs'] - df['yhat_cs'])**2
    
    # 4. Weighted Daily Aggregation of d_row
    df['dw'] = df['d_row'] * df['w']
    g_d = df.groupby('g')
    sums_d = g_d[['dw', 'w']].sum()
    sums_d = sums_d[sums_d['w'] > 0]
    
    daily_d = sums_d['dw'] / sums_d['w']
    d_vals = daily_d.values
    
    if len(d_vals) < 2: return np.nan

    # 5. HAC T-Test
    try:
        import statsmodels.api as sm
        X = np.ones((len(d_vals), 1))
        lags = min(hac_lags, len(d_vals) // 2)
        res = sm.OLS(d_vals, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
        return float(res.tvalues[0])
    except Exception:
        return float(np.mean(d_vals) / (np.std(d_vals, ddof=1) / np.sqrt(len(d_vals))))

# ---------------------------------------------------------
# 2. Main Processing Logic
# ---------------------------------------------------------

def main():
    # SETTINGS
    INPUT_FILE = r'C:\Users\kawah\prettygood\cleanrepo\results\dh_ret\withoutmidandspread\IBMCTINTERACTIONS\predictions_expanding.csv'   # Change this to your actual filename
    OUTPUT_FILE = "results/dh_ret/withoutmidandspread/metrics_weighted.csv"
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        # If running in an environment where you paste data, create dummy data:
        print("Please ensure your CSV file is in the same directory.")
        return

    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Configuration for columns
    target_col = 'y_target'
    group_col = 'group'
    weight_col = 'mid_t'

    # Filter for relevant prediction columns:
    # 1. Starts with "yhat_"
    # 2. Does NOT contain "usd" or "per_dollar" (auxiliary metrics)
    pred_cols = [
        c for c in df.columns 
        if c.startswith('yhat_') 
        and 'usd' not in c 
        and 'per_dollar' not in c
    ]

    print(f"Found {len(pred_cols)} prediction columns: {pred_cols}")
    print(f"Calculating metrics weighted by '{weight_col}'...")

    results = []

    # Pre-extract common arrays to speed up loop
    y_true = df[target_col].values
    ids = df[group_col].values
    weights = df[weight_col].values
    # Add this to my script ONLY if you use clipping in the main pipeline
    weights = np.clip(weights, 10, 5000.0) # Example values

    for model in pred_cols:
        y_pred = df[model].values
        
        metrics = {
            "model": model,
            "R2_OS_w": r2_os_weighted(y_true, y_pred, weights),
            "R2_OS_w_XS": r2_os_xs_weighted(y_true, y_pred, ids, weights),
            "CW_t_w": clark_west_t_weighted(y_true, y_pred, ids, weights),
            "CW_t_w_XS": clark_west_t_xs_weighted(y_true, y_pred, ids, weights),
            "R2_OS_Raw": r2_os(y_true, y_pred)
        }
        results.append(metrics)

    # Create DataFrame and Export
    res_df = pd.DataFrame(results)
    
    # Sort by R2_OS_w descending
    res_df = res_df.sort_values("R2_OS_w", ascending=False)
    
    # Round for display
    res_df = res_df.round(6)

    print("-" * 60)
    print(res_df.head())
    print("-" * 60)
    
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved metrics to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()