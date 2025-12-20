import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURATION & FEATURES
# ==========================================
features = [
    'moneyness_std', 'tau', 'is_call', 'delta_model', 'theta_model', 
    'gamma_model', 'vega_model', 'iv', 'baspread_chg', 
    'log_open_interest_opt', 'd_log_open_interest_opt',
    'atm_iv', 'smile_slope', 'convexity_proxy', 'rr25_proxy', 'bf25_proxy',
    'ivrv_ratio_pred', 'd_iv_atm', 'atm_ivchg_proxy', 'dist_to_wall',
    'ddist_to_wall', 'gex_proxy', 'dgex_proxy', 'ddist_to_wall_date',
    'dgex_proxy_date', 'oi_herf_date', 'log_pcr_oi_opt', 'd_log_pcr_oi_opt',
    'rv_chg', 'realskew_chg', 'fut_baspread_chg', 'fut_baspread_z',
    'log_taker_buy_sell_ratio', 'log1p_taker_buy_volume', 'log1p_long_liquidations_usd',
    'log1p_short_liquidations_usd', 'd_log_taker_buy_sell_ratio', 'd_log1p_taker_buy_volume',
    'd_log1p_long_liquidations_usd', 'd_log1p_short_liquidations_usd', 'basis', 'd_basis',
    'funding_rates', 'd_funding_rates', 'log_estimated_leverage_ratio',
    'd_log_estimated_leverage_ratio', 'log_open_interest', 'd_log_open_interest',
    'log_pcr_oi', 'd_log_pcr_oi', 'futmom_5', 'futmom_10', 'futmom_21',
    'log1p_spot_inflow_total', 'log1p_spot_outflow_total', 'asinh_spot_netflow_total',
    'log1p_spot_reserve_usd', 'log1p_spot_transactions_count_inflow',
    'log1p_spot_transactions_count_outflow', 'addresses_count_active',
    'd_log1p_spot_inflow_total', 'd_log1p_spot_outflow_total', 'd_asinh_spot_netflow_total',
    'd_log1p_spot_reserve_usd', 'd_log1p_spot_transactions_count_inflow',
    'd_log1p_spot_transactions_count_outflow', 'log1p_der_inflow_total',
    'd_log1p_der_inflow_total', 'log1p_der_outflow_total', 'd_log1p_der_outflow_total',
    'asinh_der_netflow_total', 'd_asinh_der_netflow_total',
    'z_gt', 'reddit_pos_z', 'reddit_neg_z', 'reddit_neu_z'
]

# Hyperparameters (Your best found params)
params_dart = {"bagging_freq": 1, "colsample_bytree": 0.6, "drop_rate": 0.5, "learning_rate": 0.1, "max_bin": 127, "max_depth": 6, "max_drop": 10, "min_child_samples": 200, "min_child_weight": 0.01, "min_split_gain": 0.1, "n_estimators": 100, "num_leaves": 15, "skip_drop": 0.5, "subsample": 0.8, "verbosity": -1, "reg_alpha": 3.5639, "reg_lambda": 1.8718, "boosting_type": "dart"}
params_rf = {"bootstrap": True, "max_depth": 8, "max_features": "sqrt", "max_samples": 0.8, "min_samples_leaf": 20, "n_estimators": 200, "n_jobs": -1, "random_state": 42}
params_gbdt = {"bagging_freq": 1, "colsample_bytree": 0.8, "learning_rate": 0.1, "max_bin": 127, "max_depth": 6, "min_child_samples": 200, "n_estimators": 200, "num_leaves": 31, "subsample": 0.6, "verbosity": -1, "reg_alpha": 0.2372, "reg_lambda": 2.6941, "boosting_type": "gbdt"}

# ==========================================
# 2. DATA LOADING & PREP
# ==========================================
df = pd.read_parquet('spread0.8/dhinput.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['month_idx'] = df['date'].dt.to_period('M').factorize()[0] + 1

for col in features:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# Helper for weights
def get_weights(sub_df):
    dates = pd.to_datetime(sub_df['date']).dt.normalize()
    counts = dates.value_counts()
    w = 1.0 / dates.map(counts)
    return (w / w.mean()).values

# ==========================================
# 3. TRAINING & POOLED SHAP
# ==========================================
model_types = ['All', 'Call', 'Put']
ensemble_results = {}

for m_type in model_types:
    print(f"--- Processing N-Ensemble: {m_type} ---")
    
    # TRAIN: Strictly on Months 1-38
    df_tr = df[df['month_idx'] <= 38].copy()
    if m_type == 'Call': df_tr = df_tr[df_tr['is_call'] == 1]
    elif m_type == 'Put': df_tr = df_tr[df_tr['is_call'] == 0]
    
    w_tr = get_weights(df_tr)
    m_gbdt = lgb.LGBMRegressor(**params_gbdt).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)
    m_dart = lgb.LGBMRegressor(**params_dart).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)
    m_rf   = RandomForestRegressor(**params_rf).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)

    # EXPLAIN: Pooled sample (Months 1-44) for better density in plots
    print(f"Generating SHAP values (Pooled 1-44) for {m_type}...")
    df_pool = df.copy()
    if m_type == 'Call': df_pool = df_pool[df_pool['is_call'] == 1]
    elif m_type == 'Put': df_pool = df_pool[df_pool['is_call'] == 0]
    
    X_pool = df_pool[features]
    s_gbdt = shap.TreeExplainer(m_gbdt).shap_values(X_pool)
    s_dart = shap.TreeExplainer(m_dart).shap_values(X_pool)
    s_rf   = shap.TreeExplainer(m_rf).shap_values(X_pool)
    
    ensemble_results[m_type] = {
        'shap': (s_gbdt + s_dart + s_rf) / 3.0,
        'X_df': df_pool.reset_index(drop=True)
    }

# ==========================================
# 4. PLOTTING WITH LOCAL PERCENTILES
# ==========================================

def plot_split_dependence(var_name):
    # sharey=True synchronizes the vertical axis scale
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    buckets = {'Ultra-Short (< 7D)': (0, 7/365), 'Short (7-31D)': (7/365, 31/365)}
    
    line_configs = [
        {'label': 'ATM (All)', 'model': 'All', 'mask_fn': lambda d: d['moneyness_std'].abs() <= 1.0, 'color': 'black'},
        {'label': 'ITM Call',  'model': 'Call', 'mask_fn': lambda d: d['moneyness_std'] < -1.0, 'color': 'tab:red'},
        {'label': 'OTM Call',  'model': 'Call', 'mask_fn': lambda d: d['moneyness_std'] > 1.0, 'color': 'tab:orange'},
        {'label': 'ITM Put',   'model': 'Put', 'mask_fn': lambda d: d['moneyness_std'] > 1.0, 'color': 'tab:blue'},
        {'label': 'OTM Put',   'model': 'Put', 'mask_fn': lambda d: d['moneyness_std'] < -1.0, 'color': 'tab:cyan'},
    ]

    feat_idx = features.index(var_name)
    
    print(f"\nPOOLED (1-44) OBSERVATION SUMMARY FOR: {var_name}")
    print("-" * 65)

    for i, (title, (low, high)) in enumerate(buckets.items()):
        ax = axes[i]
        for conf in line_configs:
            res = ensemble_results[conf['model']]
            df_sub = res['X_df']
            shap_sub = res['shap']
            
            mask = (df_sub['tau'] >= low) & (df_sub['tau'] < high) & conf['mask_fn'](df_sub)
            n_obs = mask.sum()
            
            print(f"{title:<20} | {conf['label']:<12} | n={n_obs}")

            # Bali et al. threshold: only plot if we have enough data to be credible
            if n_obs < 30: continue
            
            # Local Percentiles (0 to 100 for this specific line)
            x_raw = df_sub.loc[mask, var_name].values
            y_pts = shap_sub[mask.values, feat_idx]
            x_pct = pd.Series(x_raw).rank(pct=True) * 100

            sns.regplot(x=x_pct, y=y_pts, lowess=True, scatter=False, ax=ax, 
                        color=conf['color'], label=conf['label'], line_kws={'linewidth': 3})
            
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel(f"{var_name} (Local Percentile)", fontsize=12)
        ax.set_xlim(0, 100)
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.1)
        if i == 0: ax.set_ylabel("SHAP Value (Impact on Returns)", fontsize=12); ax.legend(fontsize=9)

    plt.suptitle(f"Bali et al. (2021) Dependence Plots: {var_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def run_and_save_data(df, output_filename="shap_plotting_data.pkl"):
    model_types = ['All', 'Call', 'Put']
    save_payload = {}

    for m_type in model_types:
        print(f"--- Training {m_type} ---")
        df_tr = df[df['month_idx'] <= 38].copy()
        if m_type == 'Call': df_tr = df_tr[df_tr['is_call'] == 1]
        elif m_type == 'Put': df_tr = df_tr[df_tr['is_call'] == 0]
        
        # Training
        w_tr = get_weights(df_tr)
        m_gbdt = lgb.LGBMRegressor(**params_gbdt).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)
        m_dart = lgb.LGBMRegressor(**params_dart).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)
        m_rf   = RandomForestRegressor(**params_rf).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)

        # Explain Full Data (1-44) for maximum density
        df_pool = df.copy()
        if m_type == 'Call': df_pool = df_pool[df_pool['is_call'] == 1]
        elif m_type == 'Put': df_pool = df_pool[df_pool['is_call'] == 0]
        
        X_pool = df_pool[features]
        print(f"Generating SHAP for {m_type}...")
        s_gbdt = shap.TreeExplainer(m_gbdt).shap_values(X_pool)
        s_dart = shap.TreeExplainer(m_dart).shap_values(X_pool)
        s_rf   = shap.TreeExplainer(m_rf).shap_values(X_pool)
        
        # Store metadata and average SHAP for N-En
        save_payload[m_type] = {
            'shap': (s_gbdt + s_dart + s_rf) / 3.0,
            'X_df': df_pool[['moneyness_std', 'tau', 'is_call'] + features].reset_index(drop=True)
        }
    import pickle
    with open(output_filename, 'wb') as f:
        pickle.dump(save_payload, f)
    print(f"Done! Data saved to {output_filename}")

run_and_save_data(df)
            
# # Execution
# for v in ['futmom_10', 'vega_model', 'tau', 'reddit_neg_z', 'iv']:
#     plot_split_dependence(v)