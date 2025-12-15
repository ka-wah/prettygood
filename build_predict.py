from __future__ import annotations

import math
import hashlib
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, jarque_bera, probplot, skew, kurtosis
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ CONFIGURATION ------------------------------
FOLDER = "zerovolume-nonzero-oi"  # align with your YAML paths
SPREADMAX = 1.4

CONFIG = {
    "paths": {
        "standard_dir": Path("data"),
        "options": "options.csv",
        "google_trends": "google_trends-long.csv",
        "reddit": "reddit_daily.csv",
        "rtrs": "rtrs_sentiment.csv",
        "derivs_onchain": "derivs_onchain-long.csv",
        "spot": "spot_yahoo-long.csv",
        "futures_yahoo": "futures_cme-long.csv",
        "sofr": "sofr-long.csv",
        "bitw": "bitw.csv",
    },
    "outputs_dir": Path(f"{FOLDER}"),
    "diagnostics_dir": Path(f"{FOLDER}/diagnostics"),
    "panel_out": "dhinput.parquet",
    "feature_dict_csv": "feature_dictionary.csv",
    "feature_dict_md": "feature_dictionary.md",

    # Diagnostics
    "OUTLIERS_CSV": Path(f"{FOLDER}") / "yprice_outliers_before_winsor.csv",
    "QQ_MODEL_PNG": Path(f"{FOLDER}/diagnostics") / "y_price_qq_modeling.png",
    "PP_MODEL_PNG": Path(f"{FOLDER}/diagnostics") / "y_price_pp_modeling.png",
    "HIST_MODEL_PNG": Path(f"{FOLDER}/diagnostics") / "y_price_hist_modeling.png",
    "DIST_SUMMARY_CSV": Path(f"{FOLDER}") / "y_price_distribution_summary.csv",
    "INTEGRITY_CSV": Path(f"{FOLDER}") / "daily_integrity_modeling_panel.csv",

    "dte_min": 1,
    "dte_max": 31,
    "use_sofr": True,

    "dummy_sentinel": -99.99,
    "accept_exact_pennies": True,
    "DROP_PLACEHOLDER_LIKE": False,
    "placeholder_values": {1.0},

    "iv_lower": 1e-6,
    "iv_upper": 8.0,

    "atm_window_logmny": 0.25,
    "min_points_local_fit": 2,
    "zscore_method": "expanding",

    # Microstructure trims
    "USE_FIXED_MID_FLOOR": True,
    "MID_PRICE_FIXED_FLOOR": 0.20,
    "MID_FLOOR_CLIP": (0.01, 10.00),
    "MICRO_PCT_MID_FLOOR": 0.02,
    "MICRO_PCT_REL_SPREAD_CAP": 0.98,
    "REL_SPREAD_CAP_CLIP": (0.10, 1.95),
    "APPLY_SPREAD_CAP_TO_IMPUTED": False,
    "REQUIRE_REAL_BBO": False,

    "INTRINSIC_ALPHA": 0.02,
    "APPLY_FEATURE_WINSOR": False,
    "WINSOR_LOWER": 0.01,
    "WINSOR_UPPER": 0.99,
    "APPLY_TARGET_WINSOR": False,
    "TARGET_WLO": 0.01,
    "TARGET_WHI": 0.99,
    "ALLOW_MODELED_VTP1": False,

    # ATM smoothing
    "ATM_STD_USE_SMOOTHED": True,
    "ATM_SMOOTH_HALFLIFE": 5,
    "ATM_SMOOTH_MINP": 3,

    # moneyness_std caps - important to avoid extreme leverage on tails
    "APPLY_MNY_STD_CAP": False,
    "MNY_STD_CAP_LO": -10,
    "MNY_STD_CAP_HI": 10,

    # dh_ret caps - aggressive to reduce outlier influence
    "APPLY_DH_RET_CAP": False,
    "DH_RET_CAP_LO": -5.0,
    "DH_RET_CAP_HI": 10.0,

    "MNY_DIST_SUMMARY_CSV": Path(f"{FOLDER}") / "mny_distribution_summary.csv",
    "MNY_HIST_PNG": Path(f"{FOLDER}/diagnostics") / "mny_hist_modeling.png",
    "MNY_QQ_PNG": Path(f"{FOLDER}/diagnostics") / "mny_qq_modeling.png",
    "MNY_PP_PNG": Path(f"{FOLDER}/diagnostics") / "mny_pp_modeling.png",
}

# ------------------------------ UTILITIES ------------------------------
def _polyfit_safe(x, y, deg, w=None, *, min_unique=None, min_span=1e-6):
    """
    Safer polyfit wrapper to avoid RankWarning on thin/degenerate groups.
    Returns coefficients like np.polyfit (highest power first), or NaNs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        w = np.asarray(w, dtype=float)
        mask = mask & np.isfinite(w) & (w > 0)

    x = x[mask]
    y = y[mask]
    if w is not None:
        w = w[mask]

    if x.size < (deg + 1):
        return np.full(deg + 1, np.nan)

    # defaults
    if min_unique is None:
        min_unique = deg + 1

    ux = np.unique(x)
    if ux.size < min_unique:
        return np.full(deg + 1, np.nan)

    span = float(x.max() - x.min())
    if not np.isfinite(span) or span < min_span:
        return np.full(deg + 1, np.nan)

    # center/scale improves conditioning a lot
    mu = float(x.mean())
    sd = float(x.std())
    if (not np.isfinite(sd)) or sd == 0.0:
        return np.full(deg + 1, np.nan)

    xs = (x - mu) / sd

    try:
        if w is None:
            coef = np.polyfit(xs, y, deg)
        else:
            coef = np.polyfit(xs, y, deg, w=w)
    except Exception:
        return np.full(deg + 1, np.nan)

    # NOTE: coef is in scaled-x basis.
    # If you only use fitted values or relative slopes in the same basis, you're fine.
    # If you need coefficients in original-x basis, ask and I’ll give the exact transform you need.
    return coef


def _slope_safe(x, y, w=None, *, min_unique=2, min_span=1e-6):
    c = _polyfit_safe(x, y, 1, w=w, min_unique=min_unique, min_span=min_span)
    return float(c[0]) if np.isfinite(c[0]) else np.nan

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def safe_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return to_num(obj)

def safe_math_op(x: pd.Series, op_func) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        y = op_func(x)
        y[~np.isfinite(y)] = np.nan
        return y

def safe_log(x: pd.Series) -> pd.Series:
    return safe_math_op(x, np.log)

def safe_log1p(x: pd.Series) -> pd.Series:
    return safe_math_op(x, np.log1p)

def safe_asinh(x: pd.Series) -> pd.Series:
    return safe_math_op(x, np.arcsinh)

def logit(x: pd.Series) -> pd.Series:
    return safe_math_op(x, lambda v: np.log(v / (1 - v)))

def rolling_ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    return np.sqrt(returns.ewm(alpha=(1 - lam)).var(bias=False))

def cap_series(x: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = to_num(x)
    if lo is not None:
        x = x.clip(lower=lo)
    if hi is not None:
        x = x.clip(upper=hi)
    return x

def zscore_series(x: pd.Series, method: str = "expanding") -> pd.Series:
    if method == "full_sample":
        return (x - x.mean()) / x.std(ddof=0)
    mean = x.expanding(min_periods=5).mean()
    std = x.expanding(min_periods=5).std()
    return safe_math_op(x, lambda v: (v - mean) / std)

def _finite(s: pd.Series) -> pd.Series:
    return to_num(s).replace([np.inf, -np.inf], np.nan).notna()

def _winsorise_pooled(df: pd.DataFrame, cols: List[str], lo=0.01, hi=0.99) -> pd.DataFrame:
    if not cols:
        return df
    df = df.copy()
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            x = to_num(df[c])
            with np.errstate(invalid="ignore"):
                df[c] = np.clip(x, x.quantile(lo), x.quantile(hi))
    return df

def build_next_trading_day_map(dates: pd.Series) -> dict:
    d = pd.to_datetime(pd.Series(dates).dropna().unique()).sort_values()
    d = pd.Index(d.date)
    return pd.Series(d[1:], index=d[:-1]).to_dict()

def _compute_intrinsic_today(df: pd.DataFrame) -> pd.Series:
    F = to_num(df.get("F_t", np.nan))
    K = to_num(df.get("strike", np.nan))
    is_call = to_num(df.get("is_call", np.nan)).fillna(0).astype(int)
    return np.where(is_call == 1, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))

def _compute_adaptive_micro_thresholds(df: pd.DataFrame) -> Tuple[float, Optional[float]]:
    cfg = CONFIG
    lo_clip, hi_clip = cfg["MID_FLOOR_CLIP"]

    if cfg.get("USE_FIXED_MID_FLOOR", True):
        mid_floor = cfg.get("MID_PRICE_FIXED_FLOOR", 0.50)
    else:
        mid = to_num(df.get("mid_t", np.nan))
        mid_pos = mid[mid > 0]
        mid_floor = lo_clip if mid_pos.empty else np.nanpercentile(mid_pos, cfg["MICRO_PCT_MID_FLOOR"] * 100.0)
    mid_floor = float(np.clip(mid_floor, lo_clip, hi_clip))

    rel_series = pd.Series(dtype=float)
    for col in ["opt_rel_spread_raw", "opt_rel_spread_final"]:
        if col in df.columns:
            rel_series = to_num(df[col]).dropna()
            if not rel_series.empty:
                break

    if not rel_series.empty:
        q = np.nanpercentile(rel_series, cfg["MICRO_PCT_REL_SPREAD_CAP"] * 100.0)
        rel_cap = float(np.clip(q, *cfg["REL_SPREAD_CAP_CLIP"]))
    else:
        rel_cap = cfg["REL_SPREAD_CAP_CLIP"][1]

    return mid_floor, rel_cap

def _y_distribution_diagnostics(y: pd.Series) -> Dict[str, float]:
    y = to_num(y).dropna()
    if y.empty:
        return {}
    jb = jarque_bera(y)
    jb_stat = float(jb.statistic) if hasattr(jb, "statistic") else float(jb[0])
    jb_p = float(jb.pvalue) if hasattr(jb, "pvalue") else float(jb[1])
    out = {
        "n": int(len(y)),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)) if y.std(ddof=1) > 0 else np.nan,
        "skew": float(skew(y, bias=False)) if len(y) > 3 else np.nan,
        "excess_kurtosis": float(kurtosis(y, fisher=True, bias=False)) if len(y) > 3 else np.nan,
        "jb_stat": jb_stat,
        "jb_pvalue": jb_p,
    }
    qs = y.quantile([0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995]).to_dict()
    out.update({f"q{int(k*1000):03d}": float(v) for k, v in qs.items()})
    return out

def _save_distribution_plots(y: pd.Series, hist_path: Path, qq_path: Path, pp_path: Path):
    y = to_num(y).dropna()
    if len(y) < 10:
        return
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    plt.hist(y, bins=80, density=True, alpha=0.6)
    mu, sd = float(y.mean()), float(y.std(ddof=1))
    if np.isfinite(sd) and sd > 0:
        xs = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
        plt.plot(xs, (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2), linewidth=2)
    plt.title("y_price distribution (pooled winsor, modeling panel)")
    plt.savefig(hist_path, dpi=140)
    plt.close(fig)

    fig = plt.figure()
    probplot(y, dist="norm", plot=plt)
    plt.title("Q–Q plot of y_price (modeling panel)")
    plt.savefig(qq_path, dpi=140)
    plt.close(fig)

    fig = plt.figure()
    y_sorted = np.sort(y.values)
    u = np.linspace(1 / (len(y) + 1), len(y) / (len(y) + 1), len(y))
    if np.isfinite(sd) and sd > 0:
        plt.plot(norm.cdf((y_sorted - mu) / sd), u, marker=".", linestyle="none")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Normal CDF (theoretical)")
        plt.ylabel("Empirical CDF")
        plt.title("P–P plot")
        plt.savefig(pp_path, dpi=140)
    plt.close(fig)

# ---- Futures Corwin–Schultz spread (daily OHLC)
def corwin_schultz_spread(df_hlc: pd.DataFrame, overnight_adjust: bool = True) -> pd.Series:
    if not {"high", "low", "close"}.issubset(df_hlc.columns):
        return pd.Series(np.nan, index=df_hlc.index)

    h = to_num(df_hlc["high"]).copy()
    l = to_num(df_hlc["low"]).copy()
    c_prev = to_num(df_hlc["close"]).shift(1)

    if overnight_adjust:
        up, dn = (c_prev - h).clip(lower=0), (l - c_prev).clip(lower=0)
        h += up - dn
        l += up - dn

    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / l)
        beta = hl.pow(2) + hl.shift(1).pow(2)
        gamma = np.log(
            pd.concat([h, h.shift(1)], axis=1).max(axis=1)
            / pd.concat([l, l.shift(1)], axis=1).min(axis=1)
        ).pow(2)
        den = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return s.clip(lower=0)

# ------------------------------ BLACK-76 ------------------------------
@dataclass
class B76Inputs:
    F: float
    K: float
    tau: float
    r: float
    is_call: bool

def _d1_d2(F, K, sigma, tau):
    if sigma <= 0 or tau <= 0 or F <= 0 or K <= 0:
        return np.nan, np.nan
    vol_sqrt = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt
    return d1, d1 - vol_sqrt

def b76_price(inputs: B76Inputs, sigma: float) -> float:
    F, K, tau, r, is_call = inputs.F, inputs.K, inputs.tau, inputs.r, inputs.is_call
    if tau <= 0 or F <= 0 or K <= 0:
        return max(F - K, 0.0) if is_call else max(K - F, 0.0)
    d1, d2 = _d1_d2(F, K, sigma, tau)
    if np.isnan(d1):
        return np.nan
    DF = math.exp(-r * tau)
    return DF * (F * norm.cdf(d1) - K * norm.cdf(d2)) if is_call else DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def b76_greeks(inputs: B76Inputs, sigma: float) -> Dict[str, float]:
    F, K, tau, r, is_call = inputs.F, inputs.K, inputs.tau, inputs.r, inputs.is_call
    DF = math.exp(-r * tau) if tau > 0 else 1.0

    if tau <= 0 or F <= 0 or K <= 0 or sigma <= 0:
        delta = 1.0 if (is_call and F > K) else (-1.0 if ((not is_call) and F < K) else 0.0)
        return {"delta_model": delta * DF, "gamma_model": 0.0, "vega_model": 0.0, "theta_model": 0.0, "rho_model": 0.0}

    d1, d2 = _d1_d2(F, K, sigma, tau)
    pdf = norm.pdf(d1)

    if is_call:
        price, delta = DF * (F * norm.cdf(d1) - K * norm.cdf(d2)), DF * norm.cdf(d1)
    else:
        price, delta = DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1)), -DF * norm.cdf(-d1)

    return {
        "delta_model": delta,
        "gamma_model": DF * pdf / (F * sigma * math.sqrt(tau)),
        "vega_model": DF * F * pdf * math.sqrt(tau),
        "theta_model": -DF * (F * pdf * sigma) / (2 * math.sqrt(tau)) + r * price,
        "rho_model": -tau * price,
    }

def implied_vol_b76(inputs: B76Inputs, price_obs: float, lo=1e-6, hi=8.0) -> Optional[float]:
    if not (np.isfinite(price_obs) and price_obs > 0):
        return np.nan
    try:
        return brentq(lambda sig: b76_price(inputs, sig) - price_obs, lo, hi, maxiter=200, xtol=1e-8)
    except Exception:
        return np.nan

# ------------------------------ DATA LOADING ------------------------------
def load_standard_csv(name: str) -> pd.DataFrame:
    p = CONFIG["paths"]["standard_dir"] / name
    if not p.exists():
        print(f"[WARN] Missing standardized file: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)

def load_sofr() -> pd.DataFrame:
    if not CONFIG["use_sofr"]:
        return pd.DataFrame()
    p = CONFIG["paths"]["standard_dir"] / CONFIG["paths"]["sofr"]
    if not p.exists():
        print(f"[WARN] SOFR not found -> using r=0 fallback")
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "r_annualised"]].drop_duplicates()

# ------------------------------ FUTURES SERIES (FIXED) ------------------------------
def build_futures_series_robust(opt: pd.DataFrame) -> pd.DataFrame:
    """
    Build a futures level series (F_t) for the option trading dates. DO NOT calendar-reindex to all days.
    We construct F_t for each trading date and map F_tp1 using next trading date (based on option dates).
    """
    # from options vendor settlement
    ft_opt = (
        opt.groupby("date", as_index=False)["futuresettlementprice"]
        .median()
        .rename(columns={"futuresettlementprice": "F_t_opt"})
    )

    fy = load_standard_csv(CONFIG["paths"]["futures_yahoo"])
    if not fy.empty:
        fy.columns = [c.strip().lower() for c in fy.columns]
        fy["date"] = pd.to_datetime(fy["date"]).dt.date
        col = "close" if "close" in fy.columns else ("adj close" if "adj close" in fy.columns else None)
        fy = fy[["date", col]].rename(columns={col: "F_t_y"}) if col else pd.DataFrame(columns=["date", "F_t_y"])
        fy = fy.drop_duplicates("date", keep="last")
    else:
        fy = pd.DataFrame(columns=["date", "F_t_y"])

    F = ft_opt.merge(fy, on="date", how="outer")
    F["F_t"] = np.where(np.isfinite(F.get("F_t_y")), F["F_t_y"], F.get("F_t_opt"))
    F = F[["date", "F_t"]].sort_values("date").dropna(subset=["F_t"]).drop_duplicates("date", keep="last")

    if F.empty:
        raise ValueError("No futures levels found.")

    # Map tp1 using trading dates from the options universe (not calendar)
    trade_dates = pd.Index(pd.to_datetime(opt["date"]).dropna().dt.date.unique()).sort_values()
    cal = pd.DataFrame({"date": trade_dates})
    cal = cal.merge(F, on="date", how="left").sort_values("date")
    cal["F_t"] = cal["F_t"].ffill()

    next_td = build_next_trading_day_map(cal["date"])
    cal["date_tp1"] = cal["date"].map(next_td)
    cal = cal.merge(cal[["date", "F_t"]].rename(columns={"date": "date_tp1", "F_t": "F_tp1"}), on="date_tp1", how="left")
    return cal[["date", "F_t", "F_tp1"]]

def attach_futures_next_trading_day(opt_df, fut_df, date_col="date", f_col="F_t"):
    trade_dates = pd.Index(pd.unique(opt_df[date_col].dropna())).sort_values()
    fut_eod = fut_df[[date_col, f_col]].dropna().sort_values(date_col).groupby(date_col, as_index=False).last()
    fut_cal = fut_eod.set_index(date_col).reindex(trade_dates, method="ffill").rename_axis(date_col).reset_index()
    next_td = build_next_trading_day_map(fut_cal[date_col])
    fut_cal["date_tp1"] = fut_cal[date_col].map(next_td)
    fut_cal = fut_cal.merge(
        fut_cal[[date_col, f_col]].rename(columns={date_col: "date_tp1", f_col: f"{f_col}_tp1"}),
        on="date_tp1",
        how="left",
    )
    return opt_df.merge(fut_cal[[date_col, f"{f_col}_tp1"]], on=date_col, how="left", validate="many_to_one")

# ------------------------------ COMPLEX FEATURES ------------------------------
def add_nextday_mid_by_signature(df: pd.DataFrame, next_td: dict) -> pd.Series:
    sig_cols = ["expiration", "strike", "callput", "exchange"]
    sig = df[sig_cols].fillna(np.nan).astype(str).agg("|".join, axis=1)

    dfn = df[["date", "mid_t"]].copy()
    dfn["sig"] = sig.values
    dfn = dfn.sort_values(["sig", "date"])

    dfn["date_tp1_sig"] = dfn.groupby("sig", sort=False)["date"].shift(-1)
    dfn["mid_tp1_sig"] = dfn.groupby("sig", sort=False)["mid_t"].shift(-1)

    dfn["next_trading_day"] = dfn["date"].map(next_td)
    ok = dfn["date_tp1_sig"] == dfn["next_trading_day"]
    dfn.loc[~ok, "mid_tp1_sig"] = np.nan

    out = df[["date"]].copy()
    out["sig"] = sig.values
    return out.merge(dfn[["date", "sig", "mid_tp1_sig"]], on=["date", "sig"], how="left")["mid_tp1_sig"]

def model_vtp1_from_today_iv(row) -> float:
    # FIX (2): tau decay uses actual gap to next trading day, not 1/365.
    if not (np.isfinite(row.get("F_tp1")) and np.isfinite(row.get("iv"))
            and np.isfinite(row.get("tau")) and np.isfinite(row.get("strike"))):
        return np.nan

    dt_days = row.get("dt_days_tp1", np.nan)
    if not np.isfinite(dt_days) or dt_days <= 0:
        dt_days = 1.0

    tau_next = max(row["tau"] - float(dt_days) / 365.0, 0.0)
    if tau_next <= 0.0:
        return np.nan

    inp = B76Inputs(F=row["F_tp1"], K=row["strike"], tau=tau_next, r=row.get("r_annualised", 0.0), is_call=bool(row["is_call"]))
    try:
        return b76_price(inp, float(row["iv"]))
    except Exception:
        return np.nan

def build_rr_bf_25(df_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    dates = sorted(df_rows["date"].dropna().unique())
    for d in dates:
        sub = df_rows.loc[df_rows["date"] == d]
        if sub.empty or not {"delta_model", "iv", "atm_iv"}.issubset(sub.columns):
            out.append((d, np.nan, np.nan))
            continue
        atm = sub["atm_iv"].median()
        if not np.isfinite(atm):
            out.append((d, np.nan, np.nan))
            continue

        def pick_iv(is_call):
            s = sub[(sub["is_call"] == int(is_call)) & sub["iv"].notna() & sub["delta_model"].notna()].copy()
            if s.empty:
                return np.nan
            s["adelta"] = s["delta_model"].abs()
            for band in [(0.20, 0.30), (0.15, 0.35), (0.10, 0.40)]:
                cand = s[(s["adelta"] >= band[0]) & (s["adelta"] <= band[1])]
                if len(cand) >= 3:
                    return cand["iv"].median()
            s["dist"] = (s["adelta"] - 0.25).abs()
            cand = s.nsmallest(3, "dist")
            return cand["iv"].median() if not cand.empty else np.nan

        ivc, ivp = pick_iv(True), pick_iv(False)
        rr = (ivc - ivp) if (np.isfinite(ivc) and np.isfinite(ivp)) else np.nan
        bf = (0.5 * (ivc + ivp) - atm) if (np.isfinite(ivc) and np.isfinite(ivp)) else np.nan
        out.append((d, rr, bf))
    return pd.DataFrame(out, columns=["date", "rr25_proxy", "bf25_proxy"])

def build_daily_oi_wall_fallback(df: pd.DataFrame) -> pd.DataFrame:
    out_dates = sorted(pd.to_datetime(df["date"]).dropna().dt.date.unique())
    if not out_dates:
        return pd.DataFrame(columns=["date", "dist_to_wall_fallback", "gex_proxy_fallback"])

    fb_wall = pd.DataFrame({"date": out_dates, "dist_to_wall_fallback": np.nan})
    if {"date", "strike", "F_t", "oi_eff"}.issubset(df.columns):
        tmp = df[["date", "strike", "oi_eff", "F_t"]].copy()
        tmp["strike"] = to_num(tmp["strike"])
        tmp["oi_eff"] = to_num(tmp["oi_eff"]).fillna(0.0)
        oi_ds = tmp.dropna(subset=["date", "strike"]).groupby(["date", "strike"], as_index=False)["oi_eff"].sum()
        if not oi_ds.empty:
            top = oi_ds.sort_values(["date", "oi_eff"], ascending=[True, False]).groupby("date").head(5)
            def _wavg_wall(g):
                s, w = g["strike"].values, g["oi_eff"].values
                m = np.isfinite(s) & np.isfinite(w)
                return np.average(s[m], weights=w[m]) if m.any() and np.nansum(w) > 0 else np.nan
            wall = top.groupby("date").apply(_wavg_wall).rename("K_wall_day").reset_index()
            Fd = tmp.dropna(subset=["F_t"]).groupby("date", as_index=False)["F_t"].median()
            fb_wall = wall.merge(Fd, on="date", how="left")
            with np.errstate(all="ignore"):
                fb_wall["dist_to_wall_fallback"] = np.log(fb_wall["K_wall_day"] / fb_wall["F_t"])

    fb_gex = pd.DataFrame({"date": out_dates, "gex_proxy_fallback": np.nan})
    if {"date", "is_call", "gamma_model", "oi_eff"}.issubset(df.columns):
        gx = df[["date", "is_call", "gamma_model", "oi_eff"]].copy()
        gx["sign"] = np.where(to_num(gx["is_call"]) == 1, 1.0, -1.0)
        gx["val"] = gx["sign"] * to_num(gx["oi_eff"]).fillna(0) * to_num(gx["gamma_model"]).fillna(0)
        fb_gex = gx.groupby("date")["val"].sum().rename("gex_proxy_fallback").reset_index()

    return (
        pd.DataFrame({"date": out_dates})
        .merge(fb_wall[["date", "dist_to_wall_fallback"]], on="date", how="left")
        .merge(fb_gex, on="date", how="left")
    )

def apply_low_risk_missingness_fixes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    if {"open_interest_opt", "date"}.issubset(df.columns):
        df["log_open_interest_opt"] = safe_log1p(df["open_interest_opt"])
        df["d_log_open_interest_opt"] = df["log_open_interest_opt"].diff()

    if {"date", "dist_to_wall", "gex_proxy"}.issubset(df.columns):
        dlev = df.groupby("date", as_index=False).agg(
            dist=("dist_to_wall", "median"),
            gex=("gex_proxy", "median"),
        ).sort_values("date")
        df = df.merge(
            dlev.assign(
                ddist_to_wall_date=dlev["dist"].diff(),
                dgex_proxy_date=dlev["gex"].diff(),
            )[["date", "ddist_to_wall_date", "dgex_proxy_date"]],
            on="date",
            how="left",
        )

    if {"vega_model", "d_iv_atm", "vega_ivchg_proxy"}.issubset(df.columns):
        df["vega_atm_ivchg_proxy"] = df["vega_model"] * df["d_iv_atm"]
        df["vega_ivchg_proxy_filled"] = df["vega_ivchg_proxy"].combine_first(df["vega_atm_ivchg_proxy"])

    if "d_iv_level_filled" in df.columns:
        df["ivchg_proxy"] = df["d_iv_level_filled"]
    elif "d_iv_level" in df.columns:
        df["ivchg_proxy"] = df["d_iv_level"]
    if "d_iv_atm" in df.columns:
        df["atm_ivchg_proxy"] = df["d_iv_atm"]

    if {"date", "strike", "oi_eff"}.issubset(df.columns):
        def _herf(g):
            s = g.groupby("strike")["oi_eff"].sum()
            tot = s.sum()
            return np.sum((s / tot) ** 2) if tot > 0 and not s.empty else np.nan
        df = df.merge(df.groupby("date").apply(_herf).rename("oi_herf_date").reset_index(), on="date", how="left")

    for c in ["ddist_to_wall", "dgex_proxy", "oi_herf", "vega_ivchg_proxy"]:
        if c in df.columns:
            df[f"{c}_is_na"] = df[c].isna().astype(int)
    return df

# ------------------------------ MAIN PANEL ------------------------------
def build_panel() -> pd.DataFrame:
    opt = load_standard_csv(CONFIG["paths"]["options"])
    n_raw_opt = len(opt)
    if opt.empty:
        raise FileNotFoundError("Options file not found.")
    print("[INFO] Loaded options:", n_raw_opt, "rows")

#     opt.replace(-99.99, np.nan, inplace=True)
#     print("[INFO] -99.99 values dropped:", len(opt), "rows")

# # 3. Drop rows where Bid or Offer is missing
# # You cannot trade if one side is missing.
#     opt.dropna(subset=['bid', 'offer'], inplace=True)
#     print("[INFO] missing bid/offer dropped:", len(opt), "rows")

    opt.columns = [c.strip().lower() for c in opt.columns]

    for c in ["date", "expiration", "lasttradedate"]:
        if c in opt.columns:
            opt[c] = pd.to_datetime(opt[c])

    if "dte" not in opt.columns:
        if "date" in opt.columns and "expiration" in opt.columns:
            opt["dte"] = (opt["expiration"] - opt["date"]).dt.days
        else:
            print("[WARN] 'dte', 'date', or 'expiration' missing. DTE calculation failed.")
            opt["dte"] = np.nan

    for c in ["date", "expiration", "lasttradedate"]:
        if c in opt.columns:
            opt[c] = opt[c].dt.date

    opt.rename(columns={c: f"{c}_src" for c in ["delta", "gamma", "vega", "theta", "rho"] if c in opt.columns}, inplace=True)
    opt.replace(CONFIG["dummy_sentinel"], np.nan, inplace=True)

    # Simplified version
    mask = (
        (opt["dte"] >= CONFIG["dte_min"]) & 
        (opt["dte"] <= CONFIG["dte_max"]) & 
        (
            (safe_numeric_col(opt, "volume") > 0) | 
            (safe_numeric_col(opt, "openinterest") > 0)
        )
    )
    opt = opt[mask].copy()
    n_after_dte = len(opt)

    # futures series (FIXED)
    futures_core = build_futures_series_robust(opt)  # date, F_t, F_tp1
    def construct_mid(row):
        b, o = row.get("bid", np.nan), row.get("offer", np.nan)
        settle = row.get("settlementprice", np.nan)
        
        # 1. Calculate Candidate Mid from Quotes
        mid_quote = np.nan
        if np.isfinite(b) and np.isfinite(o) and b > 0 and o >= b:
            mid_quote = 0.5 * (b + o)
            
            # --- NEW SANITY CHECK ---
            # If Quote Mid deviates > 20% from Settlement, assume quotes are stale/fake.
            # Trust Settlement instead (if valid).
            if np.isfinite(settle) and settle > 0:
                diff = abs(mid_quote - settle) / settle
                if diff > 0.20:  # 20% threshold (adjust as needed)
                    return settle
            # ------------------------
            
            return mid_quote

        # 2. Fallback to Settlement / Close
        for px in [settle, row.get("closeprice")]:
            if np.isfinite(px) and px > 0:
                return px
                
        return np.nan

    opt["mid_t"] = opt.apply(construct_mid, axis=1)

    if CONFIG["accept_exact_pennies"]:
        opt["is_placeholder_like"] = opt["mid_t"].isin(CONFIG["placeholder_values"]).astype(int)
        opt["is_penny_price"] = (opt["mid_t"] <= 2.0).astype(int)
    else:
        opt["is_placeholder_like"] = 0
        opt["is_penny_price"] = 0

    sofr = load_sofr()
    if not sofr.empty:
        fut_series = futures_core.merge(sofr, on="date", how="left")
    else:
        fut_series = futures_core.assign(r_annualised=0.0)
    fut_series["r_annualised"] = fut_series["r_annualised"].ffill().fillna(0.0)

    df = opt.merge(fut_series[["date", "F_t", "r_annualised"]], on="date", how="left")

    df["tau"] = df["dte"] / 365.0
    df["is_call"] = (df["callput"].str.upper() == "C").astype(int)

    # FIX (1): correct moneyness sign
    df["log_moneyness"] = np.log(df["F_t"] / df["strike"])

    iv0 = df.get("impliedvolatility", pd.Series(np.nan, index=df.index))
    df["iv_raw"] = iv0.where(np.isfinite(iv0) & (iv0 > 0))

    def invert_row(row):
        if np.isfinite(row["iv_raw"]):
            return row["iv_raw"]
        inp = B76Inputs(F=row["F_t"], K=row["strike"], tau=row["tau"], r=row["r_annualised"], is_call=bool(row["is_call"]))
        return implied_vol_b76(inp, row["mid_t"], lo=CONFIG["iv_lower"], hi=CONFIG["iv_upper"])

    df["iv"] = df.apply(invert_row, axis=1)

    def greeks_row(row):
        inp = B76Inputs(F=row["F_t"], K=row["strike"], tau=row["tau"], r=row["r_annualised"], is_call=bool(row["is_call"]))
        sigma = row["iv"] if np.isfinite(row["iv"]) and row["iv"] > 0 else np.nan
        if not np.isfinite(sigma):
            return pd.Series({k: np.nan for k in ["delta_model", "gamma_model", "vega_model", "theta_model", "rho_model"]})
        g = b76_greeks(inp, sigma)
        g["theta_model"] /= 365.0
        return pd.Series(g)

    df = pd.concat([df, df.apply(greeks_row, axis=1)], axis=1)

    idcol = "optionid" if "optionid" in df.columns else "optionsymbol"
    df = df.sort_values([idcol, "date"])
    for pfx in ["_src", "_model"]:
        for base in ["delta", "gamma", "vega", "theta", "rho"]:
            col_name = f"{base}{pfx}"
            if col_name in df.columns:
                df[f"d_{base}{pfx}"] = df.groupby(idcol)[col_name].diff()

    df["gamma_scaled"] = df["gamma_model"] * (0.01 * df["F_t"]) ** 2
    df["vega_1volpt"] = 0.01 * df["vega_model"]

    def price_model_today(row):
        if not all(np.isfinite(row.get(k)) and row[k] > 0 for k in ["F_t", "strike", "tau", "iv"]):
            return np.nan
        inp = B76Inputs(F=row["F_t"], K=row["strike"], tau=row["tau"], r=row.get("r_annualised", 0.0), is_call=bool(row["is_call"]))
        return b76_price(inp, float(row["iv"]))

    df["price_model_t"] = df.apply(price_model_today, axis=1)

    df["mispricing_raw"] = to_num(df.get("mid_t")) - df["price_model_t"]
    v = to_num(df.get("vega_1volpt"))
    m = to_num(df.get("mid_t"))
    df["mispricing_vega"] = np.where(np.isfinite(v) & (np.abs(v) > 1e-8), df["mispricing_raw"] / v, np.nan).clip(-5, 5)
    df["mispricing_price"] = np.where(np.isfinite(m) & (m > 1e-8), df["mispricing_raw"] / m, np.nan).clip(-1, 1)

    if "date" in df.columns:
        for c in ["mispricing_vega", "mispricing_price"]:
            df[f"{c}_xs"] = (df[c] - df.groupby("date")[c].transform("median")).clip(lower=df[c].min(), upper=df[c].max())

    df["mispricing_price_bps"] = np.where(np.isfinite(m) & (m > 1e-10), (df["mispricing_raw"] / m) * 1e4, np.nan)

    # --- ATM IV Forward + Smile ---
    def build_atm_iv_from_forward(rows):
        out = []
        for (d, e), g in rows.groupby(["date", "expiration"], sort=False):
            F_med, tau_md, r_md = g["F_t"].median(), g["tau"].median(), g.get("r_annualised", 0.0).median()
            if not (np.isfinite(F_med) and tau_md > 0):
                out.append((d, e, np.nan))
                continue

            near = g[g["log_moneyness"].abs() <= 0.25].copy()
            if "opt_rel_spread_final" in near.columns:
                rs = to_num(near["opt_rel_spread_final"])
                near = near[rs <= rs.quantile(0.95)]
                wts = 1.0 / (1.0 + rs.clip(lower=0))
            else:
                wts = pd.Series(1.0, index=near.index)

            if "is_placeholder_like" in near.columns:
                near = near[near["is_placeholder_like"] != 1]
                wts = wts.loc[near.index]

            near = near[np.isfinite(near["mid_t"]) & (near["mid_t"] > 0)]
            if near.empty:
                out.append((d, e, np.nan))
                continue

            def interp_side(df_side):
                if df_side.empty:
                    return np.nan
                x, y, w = df_side["strike"].values, df_side["mid_t"].values, wts.loc[df_side.index].values
                mm = np.isfinite(x) & np.isfinite(y)
                x, y, w = x[mm], y[mm], w[mm]
                if x.size == 0:
                    return np.nan
                idx = np.argsort(x)
                x, y, w = x[idx], y[idx], w[idx]
                if F_med > x[0] and F_med < x[-1]:
                    r = np.searchsorted(x, F_med, side="right")
                    l = r - 1
                    alpha = (F_med - x[l]) / (x[r] - x[l])
                    y_lin = y[l] * (1 - alpha) + y[r] * alpha
                    if w[l] + w[r] > 0:
                        beta = w[l] / (w[l] + w[r])
                        y_w = y[l] * beta + y[r] * (1 - beta)
                        return 0.5 * y_lin + 0.5 * y_w
                    return y_lin
                return y[np.argmin(np.abs(x - F_med))]

            vols = []
            for px, is_call in [(interp_side(near[near["is_call"] == 1]), True), (interp_side(near[near["is_call"] == 0]), False)]:
                if np.isfinite(px) and px > 0:
                    vimp = implied_vol_b76(B76Inputs(F_med, F_med, tau_md, r_md, is_call), px, CONFIG.get("iv_lower", 1e-6), CONFIG.get("iv_upper", 8.0))
                    if np.isfinite(vimp) and vimp > 0:
                        vols.append(vimp)
            out.append((d, e, np.median(vols) if vols else np.nan))
        return pd.DataFrame(out, columns=["date", "expiration", "atm_iv_forward"])

    def local_smile(group):
        out = group.copy()
        base = group[np.isfinite(group["iv"]) & np.isfinite(group["log_moneyness"])]
        if base.empty:
            return out

        for bw in [0.10, 0.15, 0.20, CONFIG["atm_window_logmny"]]:
            g = base[base["log_moneyness"].abs() <= bw]
            if len(g) < max(3, CONFIG["min_points_local_fit"]):
                continue

            x, y = g["log_moneyness"].values, g["iv"].values
            if "opt_rel_spread_final" in g.columns:
                rs = to_num(g["opt_rel_spread_final"])
                mask_rs = rs <= np.nanquantile(rs, 0.95)
                if mask_rs.sum() >= 3:
                    x, y = x[mask_rs], y[mask_rs]

            h = max(float(bw), 1e-6)
            w = np.maximum(0.0, 1.0 - (np.abs(x) / h) ** 3) ** 3

            try:
                c2, c1, c0 = _polyfit_safe(x, y, 2, w=w, min_unique=4, min_span=1e-4)
            except Exception:
                c2, c1, c0 = np.polyfit(x, y, 2)

            r = y - (c2 * x * x + c1 * x + c0)
            mad = np.median(np.abs(r - np.median(r))) + 1e-12
            mask2 = np.abs(r) <= 3.5 * 1.4826 * mad
            if mask2.sum() >= 3:
                x2, y2 = x[mask2], y[mask2]
                w2 = np.maximum(0.0, 1.0 - (np.abs(x2) / h) ** 3) ** 3
                try:
                    c2, c1, c0 = _polyfit_safe(x2, y2, 2, w=w2, min_unique=4, min_span=1e-4)
                except Exception:
                    c2, c1, c0 = np.polyfit(x2, y2, 2)

            out["atm_iv"], out["smile_slope"], out["convexity_proxy"] = float(c0), float(c1), float(c2)
            return out

        g = base[base["log_moneyness"].abs() <= CONFIG["atm_window_logmny"]]
        if len(g) == 2:
            b1 = _slope_safe(g["log_moneyness"], g["iv"], min_unique=2, min_span=1e-4)
            out["atm_iv"], out["smile_slope"], out["convexity_proxy"] = float(np.median(g["iv"])), float(b1), np.nan
        elif len(g) == 1:
            out["atm_iv"], out["smile_slope"], out["convexity_proxy"] = float(g["iv"].iloc[0]), np.nan, np.nan
        return out

    for c in ["atm_iv", "smile_slope", "convexity_proxy"]:
        df[c] = np.nan
    df = df.groupby(["date", "expiration"], group_keys=False).apply(local_smile).reset_index(drop=True)

    df = df.merge(build_atm_iv_from_forward(df), on=["date", "expiration"], how="left")
    df["atm_iv"] = df["atm_iv_forward"].combine_first(df["atm_iv"])
    df.drop(columns=["atm_iv_forward"], inplace=True)

    near_atm = df[np.isfinite(df["iv"]) & (df["log_moneyness"].abs() <= 0.08)]
    df = df.merge(near_atm.groupby("date")["iv"].median().rename("atm_iv_date").reset_index(), on="date", how="left")
    df["atm_iv"] = df["atm_iv"].fillna(df["atm_iv_date"])
    df.drop(columns=["atm_iv_date"], inplace=True)

    atm_date = df.groupby("date")["atm_iv"].median().rename("atm_iv_date_med").reset_index().sort_values("date")
    atm_date["atm_iv_smooth"] = (
        atm_date["atm_iv_date_med"]
        .ewm(halflife=CONFIG.get("ATM_SMOOTH_HALFLIFE", 5), min_periods=CONFIG.get("ATM_SMOOTH_MINP", 3))
        .mean()
        .ffill()
        # NOTE: removed .bfill() to avoid look-ahead bias
    )
    atm_date["atm_iv_smooth"] = atm_date["atm_iv_smooth"].where(atm_date["atm_iv_smooth"] > 0, np.nan).ffill().bfill()
    df = df.merge(atm_date[["date", "atm_iv_smooth"]], on="date", how="left")
    print(f"[ATM smooth] using halflife={CONFIG.get('ATM_SMOOTH_HALFLIFE',5)}")

    mask_na = df["iv"].isna() & np.isfinite(df["atm_iv"]) & (df["log_moneyness"].abs() <= 0.05)
    df.loc[mask_na, "iv"] = df.loc[mask_na, "atm_iv"]
    if mask_na.any():
        sub = df.loc[mask_na]
        recalc = sub.apply(greeks_row, axis=1)
        df.update(recalc)

    def _slope(g):
        x, y = g["tau"].values, g["atm_iv"].values
        mm = np.isfinite(x) & np.isfinite(y)
        x, y = x[mm], y[mm]
        if x.size < 2 or np.allclose(x, x[0]):
            return np.nan
        try:
            return float(np.polyfit(x, y, 1)[0])
        except Exception:
            return np.nan

    ts = df[["date", "tau", "atm_iv"]].dropna().drop_duplicates().sort_values(["date", "tau"])
    df = df.merge(ts.groupby("date", observed=True).apply(lambda g: pd.Series({"atm_term_slope": _slope(g)})).reset_index(), on="date", how="left")
    s = df.groupby("date", as_index=False)["atm_term_slope"].median().sort_values("date")
    s["slope_ff"] = s["atm_term_slope"].ffill()
    df = df.merge(s[["date", "slope_ff"]], on="date", how="left")
    df["atm_term_slope"] = df["atm_term_slope"].fillna(df["slope_ff"])
    df.drop(columns=["slope_ff"], inplace=True)

    # IV changes
    df = df.sort_values([idcol, "date"])
    df["iv_prev"] = df.groupby(idcol)["iv"].shift(1)
    df["d_iv_level"] = df["iv"] - df["iv_prev"]
    df["vega_ivchg_proxy"] = df["vega_model"] * df["d_iv_level"]

    atm_d = df[["date", "atm_iv"]].drop_duplicates("date").sort_values("date")
    df = df.merge(atm_d.assign(atm_iv_prev=atm_d["atm_iv"].shift(1))[["date", "atm_iv_prev"]], on="date", how="left")
    df["d_iv_atm"] = df["atm_iv"] - df["atm_iv_prev"]
    df["vega_d_atm_iv"] = df["vega_model"] * df["d_iv_atm"]
    df["vega_atm_ivchg_proxy"] = df["vega_model"] * df["d_iv_atm"]
    df["vega_ivchg_proxy_filled"] = df["vega_ivchg_proxy"].combine_first(df["vega_atm_ivchg_proxy"])

    # Bucketed IV lag
    df["mny_bin"] = pd.cut(
        df["log_moneyness"],
        bins=[-np.inf, -0.75, -0.35, -0.10, 0.10, 0.35, 0.75, np.inf],
        labels=["DeepOTM_P", "OTM_P", "NearOTM_P", "ATM", "NearOTM_C", "OTM_C", "DeepOTM_C"],
    )
    q = df["mid_t"].quantile([0.25, 0.5, 0.75])
    df["price_bin"] = pd.cut(df["mid_t"], bins=[-np.inf, q[0.25], q[0.5], q[0.75], np.inf], labels=["low", "med", "high", "ultra"])

    med = (
        df.groupby(["expiration", "is_call", "mny_bin", "price_bin", "date"], observed=True)["iv"]
        .median()
        .rename("iv_med")
        .reset_index()
        .sort_values("date")
    )
    med["iv_med_prev"] = med.groupby(["expiration", "is_call", "mny_bin", "price_bin"], observed=True)["iv_med"].shift(1)
    df = df.merge(med.drop(columns="iv_med"), on=["expiration", "is_call", "mny_bin", "price_bin", "date"], how="left")
    df["iv_prev_bucketed"] = df["iv_med_prev"]
    df.drop(columns=["iv_med_prev"], inplace=True)
    df["iv_prev_filled"] = df["iv_prev"].combine_first(df["iv_prev_bucketed"])
    df["d_iv_level_filled"] = df["iv"] - df["iv_prev_filled"]

    if {"date", "mny_bin", "dte", "mispricing_price_bps"}.issubset(df.columns):
        grp = df.groupby(["date", "mny_bin", "dte"], observed=True)["mispricing_price_bps"]
        df["mispricing_bps_bucket_z"] = grp.transform(lambda s: ((s - s.mean()) / s.std(ddof=0)).replace([np.inf, -np.inf], np.nan)).clip(-5, 5)

    # Futures RV & Skew (use futures_core trading days)
    Ft = futures_core[["date", "F_t"]].sort_values("date").copy()
    Ft["ret_f"] = np.log(Ft["F_t"]).diff()
    for w in [3, 5]:
        Ft[f"rv_{w}d"] = Ft["ret_f"].rolling(w).std() * np.sqrt(365)
    Ft["rv_pred"] = rolling_ewma_vol(Ft["ret_f"]).shift(1) * np.sqrt(365)
    Ft["rv_chg"] = Ft["rv_pred"].diff()

    def rskew(x):
        xc = x - x.mean()
        m2 = (xc**2).mean()
        if m2 <= 0 or not np.isfinite(m2):
            return np.nan
        return (xc**3).mean() / (m2**1.5)

    Ft["realskew_chg"] = Ft["ret_f"].rolling(5).apply(rskew, raw=False).diff()
    df = df.merge(Ft[["date", "rv_pred", "rv_3d", "rv_5d", "rv_chg", "realskew_chg"]], on="date", how="left")
    df["ivrv_ratio_pred"] = df["atm_iv"] / df["rv_pred"]

    # BITW
    bitw = load_standard_csv(CONFIG["paths"]["bitw"])
    if not bitw.empty:
        bitw.columns = [c.strip().lower() for c in bitw.columns]
        bitw["date"] = pd.to_datetime(bitw["date"]).dt.date
        bitw = bitw.sort_values("date").drop_duplicates("date", keep="last")
        px_col = "adjusted_close" if "adjusted_close" in bitw.columns else "close"
        bitw["ret_bitw"] = np.log(to_num(bitw[px_col])).diff()

        Ft2 = futures_core[["date", "F_t"]].drop_duplicates().sort_values("date").copy()
        Ft2["ret_f"] = np.log(Ft2["F_t"]).diff()
        bitw = bitw.merge(Ft2, on="date", how="left")
        cov = bitw["ret_bitw"].rolling(60).cov(bitw["ret_f"])
        var_f = bitw["ret_f"].rolling(60).var()
        beta = cov / var_f
        bitw["ret_bitw_exbtc"] = bitw["ret_bitw"] - beta * bitw["ret_f"]

        for k in [5, 10, 21]:
            bitw[f"bitw_mom_{k}_1"] = bitw["ret_bitw"].shift(1).rolling(k).sum()
        bitw["bitw_mom_exbtc_10_1"] = bitw["ret_bitw_exbtc"].shift(1).rolling(10).sum()
        bitw["bitw_vol_10"] = bitw["ret_bitw"].shift(1).rolling(10).std() * np.sqrt(365)
        bitw["rs_bitw_btc_10"] = (np.log(to_num(bitw[px_col])) - np.log(bitw["F_t"])).diff().shift(1).rolling(10).sum()

        df = df.merge(
            bitw[
                [
                    "date",
                    "bitw_mom_5_1",
                    "bitw_mom_10_1",
                    "bitw_mom_21_1",
                    "bitw_mom_exbtc_10_1",
                    "bitw_vol_10",
                    "rs_bitw_btc_10",
                ]
            ],
            on="date",
            how="left",
        )

    # Microstructure (Futures)
    fy = load_standard_csv(CONFIG["paths"]["futures_yahoo"])
    if not fy.empty:
        fy.columns = [c.strip().lower() for c in fy.columns]
        if {"date", "high", "low", "close"}.issubset(fy.columns):
            fy["date"] = pd.to_datetime(fy["date"]).dt.date
            cs = corwin_schultz_spread(fy.sort_values("date").drop_duplicates("date", keep="last").set_index("date"))
            fs = pd.DataFrame({"date": cs.index, "fut_baspread": cs.values})
            fs["fut_baspread_chg"] = fs["fut_baspread"].diff()
            fs["fut_baspread_z"] = zscore_series(fs["fut_baspread"], CONFIG["zscore_method"])
            df = df.merge(fs, on="date", how="left")

    # Microstructure (Options)
    df["opt_rel_spread_raw"] = np.where(
        (df["bid"] > 0) & (df["offer"] > 0),
        (df["offer"] - df["bid"]) / ((df["offer"] + df["bid"]) * 0.5),
        np.nan,
    )


    spreads = df['opt_rel_spread_raw'].dropna()
    bins = np.arange(0, 2.1, 0.1)
    print(pd.cut(spreads, bins=bins).value_counts().sort_index())
    print(f"Spread >= 2.0: {(spreads >= 2.0).sum()}")

    # --- NUCLEAR OPTION: DROP WIDE SPREADS ---
    # If spread is > 40% of the price, the price is fake news. Drop it.
    initial_len = len(df)
    df = df[df["opt_rel_spread_raw"] < SPREADMAX].copy() 
    print(f"[TRIM] Dropped {initial_len - len(df)} rows with Rel Spread > 40%")


    med_d = df.groupby(["date", "dte", "is_call", "mny_bin", "price_bin"], observed=True)["opt_rel_spread_raw"].median().rename("med_d").reset_index()
    df = df.merge(med_d, on=["date", "dte", "is_call", "mny_bin", "price_bin"], how="left")
    df["opt_rel_spread_imputed"] = df["opt_rel_spread_raw"].fillna(df["med_d"])


    df = df.sort_values("date")
    med_exp = (
        df.groupby(["dte", "is_call", "mny_bin", "price_bin"], observed=True)["opt_rel_spread_raw"]
        .apply(lambda s: s.expanding().median().shift(1))
        .reset_index(drop=True)
    )
    df["opt_rel_spread_imputed"] = df["opt_rel_spread_imputed"].fillna(med_exp).fillna(df["opt_rel_spread_raw"].median())
    df["opt_rel_spread_final"] = df["opt_rel_spread_raw"].combine_first(df["opt_rel_spread_imputed"])
    df["opt_rel_spread_imputed_flag"] = df["opt_rel_spread_raw"].isna().astype(int)

    df = df.sort_values([idcol, "date"])
    df["opt_rel_spread_chg"] = df.groupby(idcol)["opt_rel_spread_final"].diff()
    df["baspread_chg"] = df["opt_rel_spread_chg"]

    # Effective OI
    df["volume"] = to_num(df.get("volume", np.nan))
    df = df.sort_values([idcol, "date"])
    df["vol_roll3"] = df.groupby(idcol)["volume"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    oi_raw = to_num(df.get("openinterest", np.nan))
    df["oi_eff"] = oi_raw.where(oi_raw.fillna(0) > 0, df["vol_roll3"]).fillna(0.0)

    # OI Totals
    oi_date = df.groupby(["date", "callput"])["oi_eff"].sum().unstack(fill_value=np.nan).rename(columns={"P": "OI_P", "C": "OI_C"}).reset_index().sort_values("date")
    oi_date["log_pcr_oi_opt"] = np.log((oi_date["OI_P"] + 1e-12) / (oi_date["OI_C"] + 1e-12))
    oi_date["d_log_pcr_oi_opt"] = oi_date["log_pcr_oi_opt"].diff()
    df = df.merge(oi_date, on="date", how="left")

    # Basis
    spot = load_standard_csv(CONFIG["paths"]["spot"])
    if not spot.empty:
        spot.columns = [c.strip().lower() for c in spot.columns]
        if "date" in spot.columns and "close" in spot.columns:
            spot["date"] = pd.to_datetime(spot["date"]).dt.date
            bd = df[["date", "F_t"]].drop_duplicates("date").merge(spot.rename(columns={"close": "spot_close"})[["date", "spot_close"]], on="date")
            bd["basis"] = bd["F_t"] / bd["spot_close"] - 1.0
            bd["d_basis"] = bd["basis"].diff()
            df = df.merge(bd[["date", "basis", "d_basis", "spot_close"]], on="date", how="left")

    OItot = df.groupby("date")["oi_eff"].sum().rename("open_interest_opt").reset_index().sort_values("date").drop_duplicates("date")
    OItot["log_open_interest_opt"] = np.log1p(OItot["open_interest_opt"])
    OItot["d_log_open_interest_opt"] = OItot["log_open_interest_opt"].diff()
    df = df.merge(OItot[["date", "log_open_interest_opt", "d_log_open_interest_opt"]], on="date", how="left")

    # Contract extras
    df = df.sort_values([idcol, "date"])
    if "openinterest" in df.columns:
        df["d_openinterest"] = df.groupby(idcol)["openinterest"].diff()
    if {"volume", "openinterest"}.issubset(df.columns):
        df["volume_to_oi"] = df["volume"] / (to_num(df["openinterest"]) + 1e-9)

    # Wall/GEX
    def wall_gex_calc(g):
        s = g.groupby("strike")["oi_eff"].sum().sort_values(ascending=False)
        if s.empty:
            return pd.Series([np.nan, np.nan, np.nan], index=["oi_herf", "dist_to_wall", "gex_proxy"])
        tot = s.sum()
        h = np.sum((s / tot).values**2) if tot > 0 else np.nan
        top = s.iloc[: max(int(len(s) * 0.05), 1)]
        sw, ww = to_num(top.index).values, top.values
        wall = np.average(sw, weights=ww) if np.isfinite(sw).any() and ww.sum() > 0 else np.nan
        Fm = g["F_t"].median()
        dist = np.log(wall / Fm) if np.isfinite(wall) and Fm > 0 else np.nan
        gex = np.nansum(np.where(g["is_call"] == 1, 1.0, -1.0) * g["oi_eff"] * g["gamma_model"])
        return pd.Series([h, dist, gex], index=["oi_herf", "dist_to_wall", "gex_proxy"])

    wg = df.groupby(["date", "expiration"]).apply(wall_gex_calc).reset_index().sort_values(["expiration", "date"])
    for c in ["gex_proxy", "dist_to_wall"]:
        wg[f"d{c}"] = wg.groupby("expiration")[c].diff()
    df = df.merge(wg, on=["date", "expiration"], how="left")

    # Fallback fills
    fb = build_daily_oi_wall_fallback(df)
    df = df.merge(fb, on="date", how="left")
    for c in ["dist_to_wall", "gex_proxy"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].fillna(df[f"{c}_fallback"])
    df.drop(columns=[c for c in df.columns if "_fallback" in c], inplace=True)

    df = apply_low_risk_missingness_fixes(df)

    # ------------------------------ Sentiment processing (unchanged) ------------------------------
    def process_sentiment(
        file_key: str,
        val_col: str | None = None,
        cols_to_z: dict[str, str] | None = None,
        date_col: str = "date",
        lag_days: int = 1,
        z_method: str = "rolling",
        window: int = 60,
        min_periods: int = 60,
        ddof: int = 0,
    ) -> pd.DataFrame:
        raw = load_standard_csv(CONFIG["paths"][file_key])
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = raw.copy()
        raw.columns = [c.strip().lower() for c in raw.columns]
        dcol = "datetime" if "datetime" in raw.columns else date_col
        if dcol not in raw.columns:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw[dcol], errors="coerce").dt.date
        raw = raw.dropna(subset=["date"])

        def _z_past_only(x: pd.Series) -> pd.Series:
            x = pd.to_numeric(x, errors="coerce")
            x = x.shift(lag_days) if lag_days else x

            if z_method == "expanding":
                mu = x.expanding(min_periods=min_periods).mean()
                sd = x.expanding(min_periods=min_periods).std(ddof=ddof)
            elif z_method == "rolling":
                mu = x.rolling(window=window, min_periods=min_periods).mean()
                sd = x.rolling(window=window, min_periods=min_periods).std(ddof=ddof)
            else:
                raise ValueError(f"Unknown z_method: {z_method}")

            z = (x - mu) / sd
            return z.replace([np.inf, -np.inf], np.nan)

        if val_col is not None:
            vcol = val_col.strip().lower()
            if vcol not in raw.columns:
                return pd.DataFrame()
            tmp = raw[["date", vcol]].rename(columns={vcol: "gt_level"})
            tmp["gt_level"] = pd.to_numeric(tmp["gt_level"], errors="coerce")
            agg = tmp.groupby("date", as_index=False)["gt_level"].mean().sort_values("date")
            out = pd.DataFrame({"date": agg["date"]})
            out["z_gt"] = _z_past_only(agg["gt_level"])
            return out

        if not cols_to_z:
            return pd.DataFrame()

        cols_to_z = {k.strip().lower(): v for k, v in cols_to_z.items()}
        existing = [c for c in cols_to_z.keys() if c in raw.columns]
        if not existing:
            return pd.DataFrame({"date": sorted(raw["date"].unique())})

        dd = raw[["date"] + existing].copy()
        for c in existing:
            dd[c] = pd.to_numeric(dd[c], errors="coerce")

        count_like = {"n_comments", "comment_count", "n_posts", "volume"}
        agg_rules = {c: ("sum" if c in count_like else "mean") for c in existing}
        agg = dd.groupby("date", as_index=False).agg(agg_rules).sort_values("date")

        out = pd.DataFrame({"date": agg["date"]})
        for c, alias in cols_to_z.items():
            if c not in agg.columns:
                continue
            base = alias or f"{file_key}_{c}"
            out[f"{base}_z"] = _z_past_only(agg[c])
        return out

    df = df.merge(
        process_sentiment(
            "reddit",
            cols_to_z={"mean_pos_prob": "reddit_pos", "mean_neg_prob": "reddit_neg", "mean_neu_prob": "reddit_neu"},
            z_method="rolling",
            window=60,
            min_periods=30,
            lag_days=1,
        ),
        on="date",
        how="left",
    )

    df = df.merge(
        process_sentiment(
            "google_trends",
            val_col="value",
            z_method="rolling",
            lag_days=1,
            window=60,
            min_periods=30,
        ),
        on="date",
        how="left",
    )

    # ------------------------------ CryptoQuant derivs/on-chain (unchanged) ------------------------------
    cq = load_standard_csv(CONFIG["paths"]["derivs_onchain"])
    if not cq.empty:
        cq.columns = [c.strip().lower() for c in cq.columns]
        d_col = "datetime" if "datetime" in cq.columns else "date"
        if d_col in cq.columns:
            cq["date"] = pd.to_datetime(cq[d_col]).dt.date

            agg_dict = {
                "addresses_count_active": "mean",
                "der_inflow_total": "sum",
                "der_netflow_total": "sum",
                "der_outflow_total": "sum",
                "der_transactions_count_inflow": "sum",
                "der_transactions_count_outflow": "sum",
                "estimated_leverage_ratio": "mean",
                "exchange_whale_ratio": "mean",
                "taker_buy_ratio": "mean",
                "taker_buy_sell_ratio": "mean",
                "taker_buy_volume": "sum",
                "funding_rates": "mean",
                "open_interest": "mean",
                "put_oi": "mean",
                "call_oi": "mean",
                "long_liquidations_usd": "sum",
                "short_liquidations_usd": "sum",
                "spot_inflow_total": "sum",
                "spot_netflow_total": "sum",
                "spot_outflow_total": "sum",
                "spot_reserve_usd": "mean",
                "spot_transactions_count_inflow": "sum",
                "spot_transactions_count_outflow": "sum",
            }
            actual_agg = {k: v for k, v in agg_dict.items() if k in cq.columns}
            agg = cq.groupby("date", as_index=False).agg(actual_agg).sort_values("date").drop_duplicates("date")

            transforms = [
                ("der_inflow_total", ["log1p"]),
                ("der_outflow_total", ["log1p"]),
                ("der_transactions_count_inflow", ["log1p"]),
                ("der_transactions_count_outflow", ["log1p"]),
                ("long_liquidations_usd", ["log1p"]),
                ("short_liquidations_usd", ["log1p"]),
                ("spot_inflow_total", ["log1p"]),
                ("spot_outflow_total", ["log1p"]),
                ("spot_reserve_usd", ["log1p"]),
                ("spot_transactions_count_inflow", ["log1p"]),
                ("spot_transactions_count_outflow", ["log1p"]),
                ("der_netflow_total", ["asinh"]),
                ("spot_netflow_total", ["asinh"]),
                ("estimated_leverage_ratio", ["log", "logit"]),
                ("exchange_whale_ratio", ["logit"]),
                ("taker_buy_ratio", ["logit"]),
                ("taker_buy_sell_ratio", ["log"]),
                ("taker_buy_volume", ["log1p"]),
                ("open_interest", ["log"]),
                ("put_oi", ["log"]),
                ("call_oi", ["log"]),
            ]

            cq_cols = ["date"]
            for col, ops in transforms:
                if col not in agg.columns:
                    continue
                base = to_num(agg[col])
                for op in ops:
                    new_name = f"{op}_{col}"
                    if op == "log":
                        agg[new_name] = safe_log(base)
                    elif op == "log1p":
                        agg[new_name] = safe_log1p(base)
                    elif op == "logit":
                        agg[new_name] = logit(base)
                    elif op == "asinh":
                        agg[new_name] = safe_asinh(base)
                    agg[f"d_{new_name}"] = agg[new_name].diff()
                    cq_cols.extend([new_name, f"d_{new_name}"])

            if "funding_rates" in agg.columns:
                agg["funding_rates"] = to_num(agg["funding_rates"])
                agg["d_funding_rates"] = agg["funding_rates"].diff()
                cq_cols.extend(["funding_rates", "d_funding_rates"])

            if "addresses_count_active" in agg.columns:
                agg["addresses_count_active"] = to_num(agg["addresses_count_active"])
                agg["d_addresses_count_active"] = agg["addresses_count_active"].diff()
                cq_cols.extend(["addresses_count_active", "d_addresses_count_active"])

            if {"put_oi", "call_oi"}.issubset(agg.columns):
                agg["log_pcr_oi"] = np.log(to_num(agg["put_oi"]) / (to_num(agg["call_oi"]) + 1e-9))
                agg["d_log_pcr_oi"] = agg["log_pcr_oi"].diff()
                cq_cols.extend(["log_pcr_oi", "d_log_pcr_oi"])

            df = df.merge(agg[cq_cols], on="date", how="left", validate="many_to_one")

    # ------------------------------ Next day valuation (FIXED tau gap) ------------------------------
    df = attach_futures_next_trading_day(df, futures_core.rename(columns={"F_t": "F_t"}), date_col="date", f_col="F_t").rename(columns={"F_t_tp1": "F_tp1"})

    df = df.sort_values([idcol, "date"])
    df["date_tp1"] = df.groupby(idcol)["date"].shift(-1)
    df["mid_tp1"] = df.groupby(idcol)["mid_t"].shift(-1)
    df["dte_tp1"] = df.groupby(idcol)["dte"].shift(-1)

    next_td = build_next_trading_day_map(df["date"])
    df["next_trading_day"] = df["date"].map(next_td)
    is_next_td = df["date_tp1"] == df["next_trading_day"]
    df.loc[~is_next_td, ["mid_tp1", "dte_tp1"]] = np.nan

    mid_sig = add_nextday_mid_by_signature(df, next_td)
    df["mid_tp1_combined"] = df["mid_tp1"].combine_first(mid_sig)

    # dt_days to next trading day for tau decay (FIX)
    dt_days = (pd.to_datetime(df["next_trading_day"]) - pd.to_datetime(df["date"])).dt.days
    df["dt_days_tp1"] = dt_days.where(dt_days.notna() & (dt_days > 0), 1.0)

    intr = np.where(df["is_call"] == 1, df["F_tp1"] - df["strike"], df["strike"] - df["F_tp1"]).clip(min=0)
    df["V_tp1_intrinsic"] = np.where(df["dte"] == 1, intr, np.nan)
    df["V_tp1_model"] = df.apply(model_vtp1_from_today_iv, axis=1)
    df["V_tp1"] = np.where(df["dte"] == 1, df["V_tp1_intrinsic"], df["mid_tp1_combined"])

    V = to_num(df["V_tp1"])
    F1 = to_num(df["F_tp1"])
    K = to_num(df["strike"])
    is_call = to_num(df["is_call"]).fillna(0).astype(int)

    bad_call = (is_call == 1) & np.isfinite(V) & np.isfinite(F1) & (V > 1.05 * F1)
    bad_put = (is_call == 0) & np.isfinite(V) & np.isfinite(K) & (V > 1.05 * K)
    df.loc[bad_call | bad_put, "V_tp1"] = np.nan

    need_model = df["V_tp1"].isna() & (df["dte"] > 1)
    df.loc[need_model, "V_tp1"] = df.loc[need_model, "V_tp1_model"]
    df["V_tp1_modeled_flag"] = (need_model & df["V_tp1"].notna()).astype(int)

    # Hedged returns
    df["dh_pnl"] = (df["V_tp1"] - df["mid_t"]) - df["delta_model"] * (df["F_tp1"] - df["F_t"])
    df["y_price"] = df["dh_pnl"] / df["mid_t"]
    df["dh_ret"] = df["y_price"]

    if CONFIG.get("APPLY_DH_RET_CAP", True):
        df["dh_ret"] = cap_series(df["dh_ret"], lo=CONFIG.get("DH_RET_CAP_LO", None), hi=CONFIG.get("DH_RET_CAP_HI", None))
        df["y_price"] = df["dh_ret"]

    if "date" in df.columns:
        df["y_price_xs"] = df["dh_ret"] - df.groupby("date")["dh_ret"].transform("mean")

    df["vega_1volpt"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.loc[df["vega_1volpt"].abs() < 1e-8, "vega_1volpt"] = np.nan
    df["y_vega"] = df["dh_pnl"] / df["vega_1volpt"]

    # Ratios & predictors
    v = to_num(df["vega_1volpt"])
    for col in ["theta_model", "gamma_model"]:
        c_name = col.replace("_model", "")
        num = df[col] if col == "theta_model" else df.get("gamma_model", df.get("gamma_scaled"))
        df[f"{c_name}_per_vega"] = np.where(np.isfinite(v) & (v.abs() > 1e-10), to_num(num) / v, np.nan)

    df["log_mid_t"] = np.log(np.maximum(to_num(df["mid_t"]), 1e-10))
    df["price_per_vega"] = np.where(np.isfinite(v) & (v.abs() > 1e-10), to_num(df["mid_t"]) / v, np.nan)
    if "date" in df.columns:
        df["price_per_vega_xs"] = df["price_per_vega"] - df.groupby("date")["price_per_vega"].transform("median")

    sigma_day = df["rv_pred"] / np.sqrt(365.0)
    S_gamma = 0.5 * df["gamma_model"] * (df["F_t"] ** 2) * (sigma_day**2)
    S_gamma.replace([np.inf, -np.inf], np.nan, inplace=True)
    S_gamma[np.abs(S_gamma) < 1e-16] = np.nan
    df["y_gamma"] = df["dh_pnl"] / S_gamma
    df["gamma_scaler"] = S_gamma

    df = df.merge(build_rr_bf_25(df), on="date", how="left")
    df["abs_delta"] = df["delta_model"].abs()
    df["signed_mny"] = (df["is_call"] * 2 - 1) * df["log_moneyness"]

    atm_std = df["atm_iv_smooth"] if CONFIG.get("ATM_STD_USE_SMOOTHED", True) and "atm_iv_smooth" in df.columns else df["atm_iv"]
    df["moneyness_std"] = df["log_moneyness"] / (atm_std * np.sqrt(df["tau"].clip(lower=1e-10))).replace(0, np.nan)

    if CONFIG.get("APPLY_MNY_STD_CAP", True):
        df["moneyness_std"] = cap_series(df["moneyness_std"], lo=CONFIG.get("MNY_STD_CAP_LO", None), hi=CONFIG.get("MNY_STD_CAP_HI", None))

    df["mny"] = df["moneyness_std"]

    interact_cols = ["d_funding_rates", "d_log1p_spot_inflow_total", "d_log_open_interest", "d_log_pcr_oi"]
    for c in interact_cols:
        if c in df.columns:
            df[f"{c}_x_mny"] = df[c] * df["mny"]
            df[f"{c}_x_tau"] = df[c] * df["tau"]

    def add_futures_momenta(df0, ks=(5, 10, 21), date_col="date", futures_col="F_t"):
        g = df0[[date_col, futures_col]].drop_duplicates().sort_values(date_col).copy()
        for k in ks:
            g[f"futmom_{k}"] = g[futures_col] / g[futures_col].shift(k) - 1.0
        return df0.merge(g[[date_col] + [f"futmom_{k}" for k in ks]], on=date_col, how="left")

    df = add_futures_momenta(df, ks=(5, 10, 21), date_col="date", futures_col="F_t")

    # ------------------------------ Output + trims (unchanged) ------------------------------
    has_mid_t = np.isfinite(df["mid_t"])
    has_iv = np.isfinite(df["iv"])
    has_F_tp1 = np.isfinite(df["F_tp1"])
    has_V_tp1 = np.isfinite(df["V_tp1"])
    has_greeks = df[["delta_model", "vega_model", "gamma_model", "theta_model"]].notna().all(axis=1)
    has_y_price = has_mid_t & has_V_tp1 & has_F_tp1 & np.isfinite(df["F_t"]) & df["delta_model"].notna()

    key_cols = ["date", "exchange", "optionid", "optionsymbol", "futuresymbol"]
    for k in key_cols:
        if k not in df.columns:
            df[k] = np.nan
    df["row_id"] = df[key_cols].astype(str).agg("|".join, axis=1).apply(lambda s: hashlib.md5(s.encode()).hexdigest())

    outdir = CONFIG["outputs_dir"]
    diagdir = CONFIG["diagnostics_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    diagdir.mkdir(parents=True, exist_ok=True)

    cols = key_cols + ["expiration", "securityid", "strike", "dte", "tau", "is_call", "log_moneyness", "F_t", "mid_t", "iv"]
    cols += ["dh_pnl", "dh_ret", "y_price", "y_price_xs", "y_vega", "y_gamma", "F_tp1", "V_tp1", "V_tp1_modeled_flag", "is_penny_price", "is_placeholder_like"]
    cols += ["delta_model", "theta_per_vega", "gamma_per_vega", "atm_iv", "rho_model", "log_mid_t"]
    cols += ["atm_term_slope", "smile_slope", "ivrv_ratio_pred", "rv_pred", "convexity_proxy", "ivchg_proxy", "atm_ivchg_proxy"]
    cols += ["iv_prev", "iv_prev_bucketed", "iv_prev_filled", "d_iv_level", "d_iv_level_filled", "d_iv_atm", "rv_chg", "realskew_chg", "rr25_proxy", "bf25_proxy"]
    cols += ["price_model_t", "mispricing_raw", "mispricing_price", "mispricing_price_xs", "mispricing_price_bps", "mispricing_bps_bucket_z"]
    cols += ["oi_herf", "log_pcr_oi", "d_log_pcr_oi", "log_open_interest", "d_log_open_interest", "log_pcr_oi_opt", "d_log_pcr_oi_opt", "log_open_interest_opt", "d_log_open_interest_opt"]
    cols += ["opt_rel_spread_raw", "opt_rel_spread_final", "opt_rel_spread_chg", "fut_baspread", "fut_baspread_chg", "fut_baspread_z", "baspread_chg"]
    cols += ["dist_to_wall", "ddist_to_wall", "gex_proxy", "dgex_proxy", "ddist_to_wall_date", "dgex_proxy_date", "oi_herf_date", "d_openinterest", "volume_to_oi"]
    cols += ["z_gt", "reddit_pos_z", "reddit_neg_z", "reddit_neu_z"]
    cols += [c for c in df.columns if any(x in c for x in ["funding_rates", "leverage_ratio", "taker_buy", "liquidations", "spot", "der", "addresses_count"])]
    cols += [f"{c}_x_{m}" for c in interact_cols for m in ["mny", "tau"]]
    cols += ["ddist_to_wall_is_na", "dgex_proxy_is_na", "oi_herf_is_na", "vega_ivchg_proxy_is_na", "opt_rel_spread_imputed_flag"]
    cols += ["spot_close", "basis", "d_basis", "abs_delta", "signed_mny", "moneyness_std"]
    cols += [f"d_{g}_{s}" for g in ["delta", "gamma", "vega", "theta", "rho"] for s in ["model"]]
    cols += ["vega_1volpt", "vega_model", "gamma_model", "gamma_scaler", "theta_model", "volume"]
    cols += ["futmom_5", "futmom_10", "futmom_21"]

    cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
    df_out = df.loc[:, cols].copy()

    summary = pd.DataFrame(
        {
            "stage": [
                "raw_options_loaded",
                "after_dte_filter_1_31",
                "post_merge_panel_rows",
                "has_mid_t",
                "has_iv",
                "has_greeks",
                "has_F_tp1",
                "has_V_tp1",
                "has_y_price",
            ],
            "count": [
                n_raw_opt,
                n_after_dte,
                len(df),
                has_mid_t.sum(),
                has_iv.sum(),
                has_greeks.sum(),
                has_F_tp1.sum(),
                has_V_tp1.sum(),
                has_y_price.sum(),
            ],
        }
    )
    summary["share_of_postDTE_%"] = (summary["count"] / max(n_after_dte, 1) * 100.0).round(2)
    summary.to_csv(outdir / "diagnostics_filter_summary.csv", index=False)

    print(f"[DIAG] Raw: {n_raw_opt} | Panel: {len(df)} | y_price: {int(has_y_price.sum())}")

    print("[INFO] Applying modeling trims & pooled winsorisation...")
    mid_floor, rel_cap = _compute_adaptive_micro_thresholds(df_out)
    df_out = df_out[df_out["y_price"]< 25]
    print(f"[MICRO_DOC] mid_floor={mid_floor:.4f}, rel_cap={rel_cap:.4f}")

    before = len(df_out)
    mid_vec = safe_numeric_col(df_out, "mid_t")
    mask_mid = _finite(mid_vec) & (mid_vec >= mid_floor)

    spread_raw = safe_numeric_col(df_out, "opt_rel_spread_raw")
    spread_final = safe_numeric_col(df_out, "opt_rel_spread_final")
    imp_flag = df_out.get("opt_rel_spread_imputed_flag", spread_raw.isna()).fillna(1).astype(int)

    mask_rel_real = _finite(spread_raw) & (spread_raw <= rel_cap)
    mask_rel_imp = (_finite(spread_final) & (spread_final <= rel_cap)) if CONFIG["APPLY_SPREAD_CAP_TO_IMPUTED"] else True
    mask_rel = np.where(imp_flag == 0, mask_rel_real, mask_rel_imp).astype(bool)

    n_mid, n_real, n_imp = (~mask_mid).sum(), ((imp_flag == 0) & (~mask_rel)).sum(), ((imp_flag == 1) & (~mask_rel)).sum()
    df_out = df_out.loc[mask_mid & mask_rel].copy()

    print(f"[TRIM] Mid floor drop: {n_mid} | Spread cap drop (real/imp): {n_real}/{n_imp} | Remaining: {len(df_out)}")
    print(f"[MICRO_DOC] drops: mid={n_mid}, spread_real={n_real}, spread_imp={n_imp}")

    if CONFIG["DROP_PLACEHOLDER_LIKE"] and "is_placeholder_like" in df_out.columns:
        df_out = df_out[df_out["is_placeholder_like"] != 1].copy()
        print(f"[TRIM] Dropped placeholder-like -> {len(df_out)}")

    if CONFIG["REQUIRE_REAL_BBO"] and "opt_rel_spread_imputed_flag" in df_out.columns:
        df_out = df_out[df_out["opt_rel_spread_imputed_flag"] == 0].copy()
        print(f"[TRIM] Required real BBO -> {len(df_out)}")

    if {"F_t", "strike", "is_call", "mid_t"}.issubset(df_out.columns):
        intr_t = _compute_intrinsic_today(df_out)
        ratio = safe_numeric_col(df_out, "mid_t") / np.where(intr_t == 0, np.nan, intr_t)
        bad = (intr_t > 0) & (ratio < CONFIG["INTRINSIC_ALPHA"])
        if bad.sum() > 0:
            df_out = df_out[~bad].copy()
            print(f"[TRIM] Intrinsic consistency (n={bad.sum()}) -> {len(df_out)}")
        print(f"[MICRO_DOC] itm_drops={bad.sum()}")

    req_cols = ["y_price", "iv", "delta_model", "vega_model", "gamma_model", "theta_model"]
    if all(c in df_out.columns for c in req_cols):
        mask_req = pd.DataFrame({c: _finite(df_out[c]) for c in req_cols}).all(axis=1)
        df_out = df_out.loc[mask_req].copy()
        print(f"[KEEP] Finite y_price+IV+Greeks -> {len(df_out)}")

    if not CONFIG.get("ALLOW_MODELED_VTP1", False) and "V_tp1_modeled_flag" in df_out.columns:
        df_out = df_out[df_out["V_tp1_modeled_flag"] == 0].copy()
        print(f"[TRIM] Dropped modeled V_tp1 -> {len(df_out)}")

    if len(df_out) > 0:
        try:
            tmp = df_out.copy()
            tmp["abs_y"] = np.abs(to_num(tmp.get("y_price")))
            cols_diag = [c for c in ["date", "mid_t", "F_t", "y_price", "abs_y"] if c in tmp.columns]
            tmp.sort_values("abs_y", ascending=False)[cols_diag].head(200).to_csv(CONFIG["OUTLIERS_CSV"], index=False)
            print(f"[INFO] Wrote outliers -> {CONFIG['OUTLIERS_CSV']}")
        except Exception:
            pass

    feat_num = [c for c in cols if c in df_out.columns and pd.api.types.is_numeric_dtype(df_out[c])]
    if CONFIG["APPLY_FEATURE_WINSOR"]:
        df_out = _winsorise_pooled(df_out, feat_num, CONFIG["WINSOR_LOWER"], CONFIG["WINSOR_UPPER"])
        print("[WINSOR] Features pooled.")

    if CONFIG["APPLY_TARGET_WINSOR"] and "y_price" in df_out.columns:
        df_out = _winsorise_pooled(df_out, ["y_price"], CONFIG["TARGET_WLO"], CONFIG["TARGET_WHI"])
        print("[WINSOR] Target y_price pooled.")

    try:
        y = to_num(df_out.get("y_price")).dropna()
        if not y.empty:
            pd.DataFrame([_y_distribution_diagnostics(y)]).to_csv(CONFIG["DIST_SUMMARY_CSV"], index=False)
            _save_distribution_plots(y, CONFIG["HIST_MODEL_PNG"], CONFIG["QQ_MODEL_PNG"], CONFIG["PP_MODEL_PNG"])
            print("[DIAG] Saved distribution plots.")
    except Exception:
        pass

    try:
        mny_col = "moneyness_std"
        mny = to_num(df_out.get(mny_col)).dropna()
        if not mny.empty:
            pd.DataFrame([_y_distribution_diagnostics(mny)]).to_csv(CONFIG["MNY_DIST_SUMMARY_CSV"], index=False)
            _save_distribution_plots(mny, CONFIG["MNY_HIST_PNG"], CONFIG["MNY_QQ_PNG"], CONFIG["MNY_PP_PNG"])
            print("[DIAG] Saved moneyness distribution plots.")
    except Exception:
        pass

    if "date" in df_out.columns and feat_num:
        try:
            miss = df_out.groupby("date")[feat_num].apply(lambda g: g.isna().mean().mean()).rename("avg_miss")
            rep = df_out.groupby("date").size().rename("n_rows").to_frame().join(miss).fillna(0)
            rep.to_csv(CONFIG["INTEGRITY_CSV"])
            print("[INFO] Wrote integrity report.")
        except Exception:
            pass

    df_out.to_parquet(outdir / CONFIG["panel_out"], index=False)

    # Optional DTE splits (kept)
    def _save_dte_split(df0: pd.DataFrame, lo: int, hi: int, fname: str) -> None:
        sub = df0[df0["dte"].between(lo, hi)].copy()
        sub.to_parquet(outdir / fname, index=False)
        print(f"[OK] Saved {fname}: {len(sub)} rows (DTE {lo}-{hi})")

    if "dte" in df_out.columns:
        _save_dte_split(df_out, 1, 7, "input17.parquet")
        _save_dte_split(df_out, 8, 31, "input831.parquet")
    else:
        print("[WARN] 'dte' not found in df_out; cannot write split panels.")

    import json
    with open(outdir / "target_exposure_map.json", "w") as f:
        json.dump(
            {
                "y_vega": {"exposure_col": "vega_1volpt", "note": "$PnL ≈ y_vega * vega_1volpt"},
                "y_gamma": {"exposure_col": "gamma_scaler", "note": "$PnL ≈ y_gamma * S_gamma"},
                "y_price": {"exposure_col": "mid_t", "note": "$PnL ≈ y_price * mid_t"},
            },
            f,
            indent=2,
        )
    print(f"[OK] Saved panels to {outdir}")

    for col in ["dh_pnl", "y_price", "y_vega", "y_gamma"]:
        if col in df_out.columns:
            n = df_out[col].isna().sum()
            print(f"  {col:>24}: missing={n:5d} ({n/len(df_out)*100:.2f}%)")




    if "date" in df_out.columns:
        dates = pd.to_datetime(df_out["date"])
        print(f"[MICRO_DOC] Final: {len(df_out)} rows, {dates.nunique()} days ({dates.min().date()} to {dates.max().date()})")

    return df_out

if __name__ == "__main__":
    _ = build_panel()
