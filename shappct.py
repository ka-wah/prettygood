import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import shap
import pickle
import os

# ==========================================
# 1. SETUP
# ==========================================

# The features you want to analyze in R
targets = [
    'futmom_10', 'vega_model', 'tau', 'iv', 'moneyness_std', 
    'reddit_neg_z', 'opt_rel_spread_final', 'log_mid_t', 
    'convexity_proxy', 'log1p_spot_reserve_usd'
]

# The full feature list for training
# NOTE: I uncommented log_mid_t and opt_rel_spread_final so they are included!
features = [
    'moneyness_std', 'tau', 'is_call', 'delta_model', 'theta_model', 
    'gamma_model', 'vega_model', 'iv', 'baspread_chg', 
    'log_mid_t', 'opt_rel_spread_final', # Uncommented these!
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

# Best Hyperparameters
params_dart = {"bagging_freq": 1, "colsample_bytree": 0.6, "drop_rate": 0.5, "learning_rate": 0.1, "max_bin": 127, "max_depth": 6, "max_drop": 10, "min_child_samples": 200, "min_child_weight": 0.01, "min_split_gain": 0.1, "n_estimators": 100, "num_leaves": 15, "skip_drop": 0.5, "subsample": 0.8, "verbosity": -1, "reg_alpha": 3.5639, "reg_lambda": 1.8718, "boosting_type": "dart"}
params_rf = {"bootstrap": True, "max_depth": 8, "max_features": "sqrt", "max_samples": 0.8, "min_samples_leaf": 20, "n_estimators": 200, "n_jobs": -1, "random_state": 42}
params_gbdt = {"bagging_freq": 1, "colsample_bytree": 0.8, "learning_rate": 0.1, "max_bin": 127, "max_depth": 6, "min_child_samples": 200, "n_estimators": 200, "num_leaves": 31, "subsample": 0.6, "verbosity": -1, "reg_alpha": 0.2372, "reg_lambda": 2.6941, "boosting_type": "gbdt"}

def get_weights(sub_df):
    dates = pd.to_datetime(sub_df['date']).dt.normalize()
    counts = dates.value_counts()
    w = 1.0 / dates.map(counts)
    return (w / w.mean()).values

# ==========================================
# 2. TRAIN & PICKLE
# ==========================================
def run_and_save_data(df, output_pkl="shap_plotting_data.pkl"):
    print("Preparing data...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['month_idx'] = df['date'].dt.to_period('M').factorize()[0] + 1
    
    # Handle Infs
    for col in features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    model_types = ['All', 'Call', 'Put']
    save_payload = {}

    for m_type in model_types:
        print(f"\n--- Training {m_type} Model ---")
        
        # 1. Define Train/Pool Data
        # Train on first 38 months
        df_tr = df[df['month_idx'] <= 38].copy()
        
        # Pool (Test/Viz) data: Use ALL 44 months for best plots
        df_pool = df.copy() 

        if m_type == 'Call': 
            df_tr = df_tr[df_tr['is_call'] == 1]
            df_pool = df_pool[df_pool['is_call'] == 1]
        elif m_type == 'Put': 
            df_tr = df_tr[df_tr['is_call'] == 0]
            df_pool = df_pool[df_pool['is_call'] == 0]
        
        # 2. Train Models
        w_tr = get_weights(df_tr)
        print("  Fitting GBDT...")
        m_gbdt = lgb.LGBMRegressor(**params_gbdt).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)
        print("  Fitting DART...")
        m_dart = lgb.LGBMRegressor(**params_dart).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)
        print("  Fitting RF...")
        m_rf   = RandomForestRegressor(**params_rf).fit(df_tr[features], df_tr['dh_ret'], sample_weight=w_tr)

        # 3. Generate SHAP
        print(f"  Calculating SHAP for {len(df_pool)} observations...")
        X_pool = df_pool[features]
        s_gbdt = shap.TreeExplainer(m_gbdt).shap_values(X_pool)
        s_dart = shap.TreeExplainer(m_dart).shap_values(X_pool)
        s_rf   = shap.TreeExplainer(m_rf).shap_values(X_pool)
        
        # 4. Store Results
        save_payload[m_type] = {
            'shap': (s_gbdt + s_dart + s_rf) / 3.0,
            'X_df': df_pool[features].reset_index(drop=True) 
        }

    print(f"\nSaving Pickle to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(save_payload, f)
    print("Pickle saved.")

# ==========================================
# 3. CONVERT PICKLE TO CSV (FOR R)
# ==========================================
def convert_to_csv(input_pkl="shap_plotting_data.pkl", output_csv="shap_r_ready_final.csv"):
    print(f"\nConverting {input_pkl} to {output_csv}...")
    
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    line_configs = [
        {'label': 'ATM (All)', 'model': 'All', 'mask_fn': lambda d: d['moneyness_std'].abs() <= 1.0},
        {'label': 'ITM Call',  'model': 'Call', 'mask_fn': lambda d: d['moneyness_std'] < -1.0},
        {'label': 'OTM Call',  'model': 'Call', 'mask_fn': lambda d: d['moneyness_std'] > 1.0},
        {'label': 'ITM Put',   'model': 'Put',  'mask_fn': lambda d: d['moneyness_std'] > 1.0},
        {'label': 'OTM Put',   'model': 'Put',  'mask_fn': lambda d: d['moneyness_std'] < -1.0},
    ]

    tidy_data_list = []

    for conf in line_configs:
        model_name = conf['model']
        label = conf['label']
        
        X_df = data[model_name]['X_df']
        # DEDUPLICATION (Safety check)
        X_df = X_df.loc[:, ~X_df.columns.duplicated()].copy()
        
        shap_vals = data[model_name]['shap']
        
        # Maturity logic (approx 0.019 years = 7 days)
        X_df['maturity'] = np.where(X_df['tau'] < 7/365, 'Ultra-Short (< 7D)', 'Short (7-31D)')
        
        # Apply mask
        mask = conf['mask_fn'](X_df).values
        sub_X = X_df.iloc[mask].reset_index(drop=True)
        sub_shap = shap_vals[mask]
        
        print(f"  Processing {label:<10} | n={len(sub_X)}")

        for var in targets:
            if var in sub_X.columns:
                var_idx = sub_X.columns.tolist().index(var)
                
                df_slice = pd.DataFrame({
                    'feature_name': var,
                    'feature_value': sub_X[var],
                    'shap_value': sub_shap[:, var_idx],
                    'category': label,
                    'maturity': sub_X['maturity']
                })
                tidy_data_list.append(df_slice)
            else:
                print(f"    WARNING: {var} missing in {label}")

    if tidy_data_list:
        tidy_df = pd.concat(tidy_data_list, ignore_index=True)
        tidy_df.to_csv(output_csv, index=False)
        print(f"Success! CSV saved to {output_csv}")
    else:
        print("Error: No data extracted.")

# ==========================================
# 4. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    print("Reading parquet...")
    df = pd.read_parquet('spread0.8/dhinput.parquet')
    
    # 2. Train Models and Create Pickle
    run_and_save_data(df, "shap_plotting_data.pkl")
    
    # 3. Create CSV for R
    convert_to_csv("shap_plotting_data.pkl", "shap_r_ready_final.csv")