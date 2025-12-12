from __future__ import annotations

import json
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.utils import check_random_state

from models import build_equal_weight_ensemble, registry_from_yaml

warnings.filterwarnings("ignore")
LOG = logging.getLogger("optml")

# =============================================================================
# Run configuration
# =============================================================================
INTERACTION_PARENTS = {
    "d_funding_rates_x_mny": ("d_funding_rates", "moneyness_std"),
    "d_funding_rates_x_tau": ("d_funding_rates", "tau"),
    "d_log1p_spot_inflow_total_x_mny": ("d_log1p_spot_inflow_total", "moneyness_std"),
    "d_log1p_spot_inflow_total_x_tau": ("d_log1p_spot_inflow_total", "tau"),
    "d_log_open_interest_x_mny": ("d_log_open_interest", "moneyness_std"),
    "d_log_open_interest_x_tau": ("d_log_open_interest", "tau"),
    "d_log_pcr_oi_x_mny": ("d_log_pcr_oi", "moneyness_std"),
    "d_log_pcr_oi_x_tau": ("d_log_pcr_oi", "tau"),
}


@dataclass
class RunSpec:
    label: str = ""
    base_config_path: str = "config.yaml"
    out_dir: Optional[str] = None
    groups: Optional[List[str]] = None
    contracts_type: Optional[str] = None
    contracts_mny: Optional[str] = None
    atm_band: Optional[Tuple[float, float]] = None
    target: Optional[str] = None
    data_path: Optional[str] = None
    exclude_features: List[str] = field(default_factory=list)
    use_base_yaml_only: bool = False
    grid_sides: Optional[List[str]] = None
    grid_moneyness: Optional[List[str]] = None

SIDES = ["all", "call", "put"]
MNY_BUCKETS = ["all", "otm", "itm", "atm"]
GROUPS = ["I", "B", "M", "C", "T", "INTERACTIONS"]
CONFIG_PATH = "./ew.yaml"

SHORT = 'input/input831.parquet'
ULTRA = 'input/input17.parquet'


# near top in __main__
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--side", choices=["all", "call", "put"], default="all")
args = parser.parse_args()

RUNS = [
    RunSpec(
        label=f"side={args.side}",
        base_config_path=CONFIG_PATH,
        data_path=SHORT,
        groups=GROUPS,
        contracts_type=args.side,
        contracts_mny="all",
    ),
]



# =============================================================================
# Logging
# =============================================================================


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)
    root.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    logging.captureWarnings(True)


def add_file_logger(path: str, level: str = "INFO") -> logging.Handler:
    lvl = getattr(logging, level.upper(), logging.INFO)
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(lvl)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


def remove_handler(h: logging.Handler) -> None:
    try:
        logging.getLogger().removeHandler(h)
        h.close()
    except Exception:
        pass


# =============================================================================
# Utils / I-O
# =============================================================================


def _as_float(x, default=0.0) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return default
    return default


def _as_bool(x, default=False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _as_str_choice(x, allowed: set, default: str) -> str:
    if isinstance(x, str):
        s = x.strip().lower()
        if s in allowed:
            return s
    return default


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet", ".pq", ".parq"}:
        return pd.read_parquet(path)
    if ext in {".feather", ".ft"}:
        return pd.read_feather(path)
    return pd.read_csv(path)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def derive_out_dir(cfg: Dict[str, Any], groups_tag: Optional[str] = None) -> str:
    c = (cfg.get("contracts") or {})
    side = _as_str_choice(c.get("type", "all"), {"all", "call", "put"}, "all")
    mny = _as_str_choice(c.get("moneyness", "all"), {"all", "atm", "otm", "itm"}, "all")
    target = str(cfg.get("data", {}).get("target", "target"))
    root = (cfg.get("paths", {}) or {}).get("out_root", "results")
    base = os.path.join(root, target, f"{side}-{mny}")
    return os.path.join(base, groups_tag) if groups_tag else base


def add_moneyness_std(
    df: pd.DataFrame,
    log_m_col: str = "log_moneyness",
    atm_iv_col: str = "atm_iv",
    tau_col: str = "tau",
    out_col: str = "moneyness_std",
) -> pd.DataFrame:
    """Bali et al.: log(K/S)/(σ_ATM * sqrt(τ))."""
    df = df.copy()
    x = pd.to_numeric(df[log_m_col], errors="coerce")
    sig = pd.to_numeric(df[atm_iv_col], errors="coerce")
    tau = pd.to_numeric(df[tau_col], errors="coerce")
    denom = sig * np.sqrt(np.clip(tau, 1e-12, None))
    denom = denom.replace([0.0, np.inf, -np.inf], np.nan)
    df[out_col] = x / denom
    return df


def ensure_price_per_vega(
    df: pd.DataFrame,
    mid_col: str = "mid_t",
    vega_col: str = "vega_1volpt",
    out_col: str = "price_per_vega",
    eps: float = 1e-10,
) -> pd.DataFrame:
    """
    Ensure df has 'price_per_vega' ≈ option price per 1 vol-point of vega.
    Creates df[out_col] = mid_t / vega_1volpt if missing; otherwise leaves as-is.
    """
    if out_col in df.columns:
        return df
    if (mid_col in df.columns) and (vega_col in df.columns):
        mid = pd.to_numeric(df[mid_col], errors="coerce")
        vega = pd.to_numeric(df[vega_col], errors="coerce")
        df[out_col] = np.where(np.isfinite(vega) & (np.abs(vega) > eps), mid / vega, np.nan)
    return df


# =============================================================================
# Splits & feature selection
# =============================================================================
def time_split(
    df: pd.DataFrame,
    time_col: str,
    tr: float,
    va: float,
    purge_gap: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique calendar dates. Optional embargo (purge_gap) removes
    `purge_gap` dates at the end of train and val blocks, and shifts subsequent
    blocks accordingly (no overlap, no “stale” te_start).
    """
    d = pd.to_datetime(df[time_col], errors="coerce").dt.normalize()
    u = pd.Index(d.dropna().sort_values().unique())
    n = len(u)
    if n < 5:
        raise ValueError(f"Too few unique dates ({n}) for a 3-way split.")

    n_tr = max(1, int(np.floor(n * tr)))
    n_va = max(1, int(np.floor(n * va)))
    if n_tr + n_va >= n:
        n_va = max(1, n - n_tr - 1)

    # contiguous blocks by date
    tr_dates_full = u[:n_tr]
    va_dates_full = u[n_tr : n_tr + n_va]
    te_dates_full = u[n_tr + n_va :]

    gap = max(0, int(purge_gap))
    if gap > 0:
        # remove last gap dates from train, and last gap dates from val
        tr_dates = tr_dates_full[:-gap] if len(tr_dates_full) > gap else tr_dates_full[:0]
        va_dates = va_dates_full[:-gap] if len(va_dates_full) > gap else va_dates_full[:0]

        # shift val/test starts forward by removing the embargoed dates between blocks
        # i.e. val starts at n_tr + gap, test starts at n_tr + n_va + gap
        va_start = min(n, n_tr + gap)
        te_start = min(n, n_tr + n_va + gap)

        va_dates = u[va_start : te_start]
        te_dates = u[te_start:]
    else:
        tr_dates, va_dates, te_dates = tr_dates_full, va_dates_full, te_dates_full

    mask_tr = d.isin(tr_dates)
    mask_va = d.isin(va_dates)
    mask_te = d.isin(te_dates)
    return df.loc[mask_tr].copy(), df.loc[mask_va].copy(), df.loc[mask_te].copy()



def select_features_by_groups(cfg: Dict[str, Any], groups: List[str]) -> List[str]:
    base = list(cfg["data"].get("features", []))
    group_map = cfg["data"].get("feature_groups", {}) or {}
    seen = set(base)
    ordered = list(base)
    for g in groups:
        if g not in group_map:
            raise KeyError(f"Feature group '{g}' not in YAML data.feature_groups")
        for f in group_map[g]:
            if f not in seen:
                ordered.append(f)
                seen.add(f)
    return ordered


# =============================================================================
# Optional: shift date-constant features by 1 day
# =============================================================================


def _date_constant_columns(
    df: pd.DataFrame, date_col: str, cols: List[str], tol: float = 1e-12
) -> List[str]:
    const_cols: List[str] = []
    g = df.groupby(date_col, sort=False)
    for c in cols:
        try:
            s = g[c].transform("std")
            ok = s.dropna()
            if ok.empty:
                continue
            if (ok <= tol).mean() >= 0.95:
                const_cols.append(c)
        except Exception:
            continue
    return const_cols


def shift_features_by_one_day(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: List[str],
    *,
    only_date_constant: bool = True,
    force_all: bool = False,
) -> pd.DataFrame:
    """Replace each date’s feature values with the prior date’s cross-sectional median."""
    df = df.copy()
    if force_all:
        cols_to_shift = list(feature_cols)
    else:
        cols_to_shift = (
            _date_constant_columns(df, date_col, feature_cols)
            if only_date_constant
            else list(feature_cols)
        )
    LOG.info(
        "[shift] date_col=%s | candidates=%d | shifting=%d",
        date_col,
        len(feature_cols),
        len(cols_to_shift),
    )
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    df[date_col] = dates
    u = sorted(dates.unique())
    u_idx = pd.Index(u, name=date_col)
    for c in cols_to_shift:
        per_date = df.groupby(date_col)[c].median().reindex(u_idx)
        per_date_shifted = per_date.shift(1)
        df[c] = dates.map(per_date_shifted)
    return df


# =============================================================================
# Contract filters
# =============================================================================


def apply_contract_filters(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    c = cfg.get("contracts", {}) or {}
    which_type = _as_str_choice(c.get("type", "all"), {"all", "call", "put"}, "all")
    which_mny = _as_str_choice(c.get("moneyness", "all"), {"all", "atm", "otm", "itm"}, "all")
    band = c.get("atm_band", [-1.0, 1.0])
    lo, hi = _as_float(band[0], -1.0), _as_float(band[1], 1.0)

    mcol = cfg["data"].get("moneyness_col", "moneyness")
    tcol = cfg["data"].get("option_type_col", "option_type")

    def norm_type_any(x):
        try:
            if isinstance(x, (int, float, np.number)):
                v = float(x)
                if np.isnan(v):
                    return "put"
                if v > 0:
                    return "call"
                if v < 0:
                    return "put"
                return "put" if v == 0 else "call"
            s = str(x).strip().lower()
            if s in {"c", "call", "calls", "1", "true", "t", "yes", "y"}:
                return "call"
            if s in {"p", "put", "puts", "-1", "0", "false", "f", "no", "n"}:
                return "put"
            return "put"
        except Exception:
            return "put"

    out = df.copy()
    if which_type in {"call", "put"} and tcol in out.columns:
        tt = out[tcol].map(norm_type_any)
        out = out.loc[tt == which_type]

    if which_mny != "all" and mcol in out.columns:
        m = pd.to_numeric(out[mcol], errors="coerce")
        typ = (
            out[tcol].map(norm_type_any)
            if tcol in out.columns
            else pd.Series(["call"] * len(out), index=out.index)
        )

        name = str(mcol).lower()
        is_logratio = ("log" in name) and ("std" not in name)
        is_standardized = ("std" in name) or ("degree" in name)

        if is_logratio:
            lo_t, hi_t = math.log(lo), math.log(hi)
            atm_mask = (m >= lo_t) & (m <= hi_t)
            itm_mask = ((typ == "call") & (m < lo_t)) | ((typ == "put") & (m > hi_t))
        elif is_standardized:
            atm_mask = (m >= lo) & (m <= hi)
            itm_mask = ((typ == "call") & (m < lo)) | ((typ == "put") & (m > hi))
        else:
            atm_mask = (m >= lo) & (m <= hi)
            itm_mask = ((typ == "call") & (m > hi)) | ((typ == "put") & (m < lo))

        otm_mask = ~atm_mask & ~itm_mask

        if which_mny == "atm":
            out = out.loc[atm_mask]
        elif which_mny == "itm":
            out = out.loc[itm_mask]
        elif which_mny == "otm":
            out = out.loc[otm_mask]
    return out


# =============================================================================
# Target normaliser & guardrails
# =============================================================================


class TargetNormalizer:
    """
    Optional per-observation normaliser for the modelling target.
    Default is identity: y_model = y_target. If enabled: y_model = y_target / g.
    """

    def __init__(
        self,
        kind: str = "none",
        *,
        abs_greek: bool = True,
        floor: float = 1e-6,
        column: Optional[str] = None,
        force: bool = False,
    ):
        self.kind = (kind or "none").lower()
        self.abs_greek = abs_greek
        self.floor = float(floor)
        self.column = column
        self.force = bool(force)

    def _vec(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "none":
            return np.ones(len(df), dtype=float)
        if self.kind == "column":
            if not self.column or self.column not in df.columns:
                raise KeyError(f"target_norm.column={self.column} not found.")
            g = df[self.column].to_numpy(dtype=float)
        elif self.kind == "vega":
            g = df["vega_model"].to_numpy(dtype=float)
        elif self.kind == "gamma":
            g = df["gamma_scaled"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Unknown target_norm.kind={self.kind}")
        if self.abs_greek:
            g = np.abs(g)
        return np.maximum(g, self.floor)

    def apply(
        self, y: np.ndarray, df: pd.DataFrame, *, target_col: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (self.kind != "none") and (
            ("gamma" in target_col.lower()) or ("vega" in target_col.lower())
        ) and not self.force:
            LOG.warning(
                "target_norm enabled but target looks pre-scaled (%s). "
                "Set force=true to override. Using no scaling.",
                target_col,
            )
            return y.astype(float), np.ones(len(y), dtype=float)
        g = self._vec(df)
        y_model = y.astype(float) / g
        return y_model, g

    @staticmethod
    def invert(y_model: np.ndarray, g: np.ndarray) -> np.ndarray:
        return y_model * g


class LabelGuardrails:
    """
    Fit on TRAIN ONLY; apply same transform/winsor to VAL/TEST.
    Inverse maps predictions back to the (normalised) label space.
    """

    def __init__(self, cfg: Dict[str, Any]):
        tg = (cfg.get("data", {}).get("target_guardrails") or {})
        self.t_cfg = (tg.get("transform") or {})
        self.w_cfg = (tg.get("winsorize") or {})
        self.enabled_transform = bool(self.t_cfg.get("enabled", False))
        self.enabled_winsor = bool(self.w_cfg.get("enabled", False))
        self.kind = str(self.t_cfg.get("kind", "asinh")).lower()
        self.scale_method = str(
            (self.t_cfg.get("scale") or {}).get("method", "mad")
        ).lower()
        self.scale_fixed = float((self.t_cfg.get("scale") or {}).get("fixed_value", 1.0))
        self.mad_eps = float((self.t_cfg.get("scale") or {}).get("mad_eps", 1e-9))
        self.lower_q = float(self.w_cfg.get("lower_q", 0.001))
        self.upper_q = float(self.w_cfg.get("upper_q", 0.999))
        self._fitted = False
        self._s: float = 1.0
        self._lo: Optional[float] = None
        self._hi: Optional[float] = None
        self._yj: Optional[PowerTransformer] = None

    @staticmethod
    def _mad(x: np.ndarray) -> float:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return 1.4826 * mad

    def _choose_scale(self, y: np.ndarray) -> float:
        if self.scale_method == "fixed":
            return max(self.mad_eps, float(self.scale_fixed))
        if self.scale_method == "median_abs":
            s = float(np.median(np.abs(y)))
            return max(self.mad_eps, s)
        s = float(self._mad(y))
        if not np.isfinite(s) or s <= self.mad_eps:
            s = float(np.percentile(np.abs(y), 75))
        return max(self.mad_eps, s)

    def fit(self, y_train_model_space: np.ndarray) -> None:
        y = y_train_model_space.astype(float)
        if self.enabled_transform:
            if self.kind == "asinh":
                self._s = self._choose_scale(y)
                z = np.arcsinh(y / self._s)
                LOG.info("[guardrails] transform=asinh | scale=%.6g", self._s)
            elif self.kind == "yeo_johnson":
                self._yj = PowerTransformer(method="yeo-johnson", standardize=False)
                z = self._yj.fit_transform(y.reshape(-1, 1)).ravel()
                LOG.info(
                    "[guardrails] transform=yeo_johnson | lambda_=%.4f",
                    float(self._yj.lambdas_[0]),
                )
            else:
                raise ValueError(f"Unknown guardrail transform kind={self.kind}")
        else:
            z = y
            LOG.info("[guardrails] transform=none")

        if self.enabled_winsor:
            lo, hi = np.nanquantile(z, [self.lower_q, self.upper_q])
            self._lo, self._hi = float(lo), float(hi)
            LOG.info(
                "[guardrails] winsor z: lower_q=%.4f (%.6g), upper_q=%.4f (%.6g)",
                self.lower_q,
                self._lo,
                self.upper_q,
                self._hi,
            )
        else:
            self._lo = self._hi = None
            LOG.info("[guardrails] winsor=disabled")

        self._fitted = True

    def _forward(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return y.astype(float)
        if self.enabled_transform:
            if self.kind == "asinh":
                z = np.arcsinh(y / self._s)
            elif self.kind == "yeo_johnson":
                z = self._yj.transform(y.reshape(-1, 1)).ravel()
            else:
                z = y
        else:
            z = y
        if self.enabled_winsor and self._lo is not None and self._hi is not None:
            z = np.clip(z, self._lo, self._hi)
        return z

    def _inverse(self, z: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return z.astype(float)
        if self.enabled_transform:
            if self.kind == "asinh":
                y = np.sinh(z) * self._s
            elif self.kind == "yeo_johnson":
                y = self._yj.inverse_transform(z.reshape(-1, 1)).ravel()
            else:
                y = z
        else:
            y = z
        return y

    def forward(self, y: np.ndarray) -> np.ndarray:
        return self._forward(y)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return self._inverse(z)


# =============================================================================
# Metrics
# =============================================================================

def r2_os(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() == 0:
        return np.nan
    y_true = y_true[good]; y_pred = y_pred[good]
    msfe_model = np.mean((y_true - y_pred) ** 2)
    msfe_naive = np.mean((y_true - 0.0) ** 2)
    if not np.isfinite(msfe_naive) or msfe_naive <= 0:
        return np.nan
    return 1.0 - msfe_model / msfe_naive

def r2_os_xs(y_true: np.ndarray, y_pred: np.ndarray, ids: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "g": ids})
    df = df[np.isfinite(df["y"]) & np.isfinite(df["yhat"])]
    if df.empty:
        return np.nan
    df["y_cs"] = df["y"] - df.groupby("g")["y"].transform("mean")
    df["yhat_cs"] = df["yhat"] - df.groupby("g")["yhat"].transform("mean")
    den = float(np.mean(df["y_cs"] ** 2))
    if not np.isfinite(den) or den <= 0:
        return np.nan
    num = float(np.mean((df["y_cs"] - df["yhat_cs"]) ** 2))
    return 1.0 - num / den



def r2_os_day_weighted(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "g": groups})
    g = df.groupby("g", sort=False)
    e_model = g.apply(lambda x: np.mean((x["y"] - x["yhat"]) ** 2))
    e_naive = g.apply(lambda x: np.mean(x["y"] ** 2))
    return 1.0 - e_model.mean() / e_naive.mean()


def r2_os_winsor_aligned(y_true: np.ndarray, y_pred: np.ndarray, lo: float, hi: float) -> float:
    y_clip = np.clip(y_true, lo, hi)
    msfe_model = np.mean((y_clip - y_pred) ** 2)
    msfe_naive = np.mean((y_clip - 0.0) ** 2)
    return 1.0 - msfe_model / msfe_naive


def winsor_aligned_target(
    y_true_target: np.ndarray, g_vec: np.ndarray, guard: LabelGuardrails
) -> np.ndarray:
    y_model = np.asarray(y_true_target, dtype=float) / np.asarray(g_vec, dtype=float)
    z = guard.forward(y_model)
    y_model_clip = guard.inverse(z)
    return y_model_clip * np.asarray(g_vec, dtype=float)


def clark_west_t(
    y_true: np.ndarray,
    y_pred_alt: np.ndarray,
    y_pred_base: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    hac_lags: int = 5,
) -> float:
    y = np.asarray(y_true, dtype=float)
    ya = np.asarray(y_pred_alt, dtype=float)

    if y_pred_base is not None:
        raise ValueError("clark_west_t currently implemented only for zero benchmark.")

    e0 = y
    e1 = y - ya
    d_row = e0**2 - e1**2

    if groups is None:
        d = d_row
    else:
        gser = pd.Series(groups)
        d = pd.Series(d_row).groupby(gser).mean().values

    try:
        import statsmodels.api as sm
        X = np.ones((len(d), 1))
        res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
        return float(res.tvalues[0])
    except Exception:
        return float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))


def clark_west_t_xs(
    y_true: np.ndarray,
    y_pred_alt: np.ndarray,
    y_pred_base: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    hac_lags: int = 5,
) -> float:
    if groups is None:
        raise ValueError("groups must be provided for CW_t_XS.")

    y = np.asarray(y_true, dtype=float)
    ya = np.asarray(y_pred_alt, dtype=float)

    if y_pred_base is not None:
        raise ValueError("clark_west_t_xs currently implemented only for zero benchmark.")

    df = pd.DataFrame({"y": y, "ya": ya, "g": groups})

    df["y_cs"] = df["y"] - df.groupby("g")["y"].transform("mean")
    df["ya_cs"] = df["ya"] - df.groupby("g")["ya"].transform("mean")

    e0 = df["y_cs"]
    e1 = df["y_cs"] - df["ya_cs"]

    d_row = e0**2 - e1**2
    d_day = d_row.groupby(df["g"]).mean().values

    try:
        import statsmodels.api as sm
        X = np.ones((len(d_day), 1))
        res = sm.OLS(d_day, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
        return float(res.tvalues[0])
    except Exception:
        d = d_day
        return float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))


def diebold_mariano_t(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    groups: Optional[np.ndarray] = None,
    hac_lags: int = 5,
) -> float:
    y = np.asarray(y_true, dtype=float)
    y1 = np.asarray(y_pred1, dtype=float)
    y2 = np.asarray(y_pred2, dtype=float)
    if groups is not None:
        df = pd.DataFrame({"y": y, "y1": y1, "y2": y2, "g": groups})
        g = df.groupby("g", sort=True)[["y", "y1", "y2"]].mean()
        y, y1, y2 = g["y"].to_numpy(), g["y1"].to_numpy(), g["y2"].to_numpy()
    d = (y - y1) ** 2 - (y - y2) ** 2
    try:
        import statsmodels.api as sm
        X = np.ones((len(d), 1))
        res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
        return float(res.tvalues[0])
    except Exception:
        return float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    sx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    sy = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(sx) == 0 or np.std(sy) == 0:
        return np.nan
    return float(np.corrcoef(sx, sy)[0, 1])


# =============================================================================
# Equal-day weights
# =============================================================================
def _equal_day_weights(dates_arr):
    if dates_arr is None:
        return None
    s = pd.to_datetime(pd.Series(dates_arr), errors="coerce").dt.normalize()
    s = s.dropna()
    if s.empty:
        return None
    cnt = s.map(s.value_counts())
    w = (1.0 / cnt.to_numpy(dtype=float))
    m = np.nanmean(w)
    if np.isfinite(m) and m != 0:
        w = w / m
    # map back to original length (NaT -> nan weight)
    out = np.full(shape=(len(pd.Series(dates_arr)),), fill_value=np.nan, dtype=float)
    out[s.index.to_numpy()] = w
    return out


# =============================================================================
# Param sampling (grid/random + dist support)
# =============================================================================


def _iter_param_sets(space: Dict[str, Any], strategy: str, n_iter: int, seed: int):
    values_space: Dict[str, List[Any]] = {}
    dists: Dict[str, Dict[str, Any]] = {}
    for k, v in space.items():
        if isinstance(v, dict) and "values" in v:
            values_space[k] = list(v["values"])
        elif isinstance(v, dict) and "dist" in v:
            dists[k] = v
        else:
            values_space[k] = [v]

    rng = check_random_state(seed)

    def _sample_from_dist(spec: Dict[str, Any]) -> float:
        kind = spec["dist"].lower()
        if kind == "uniform":
            low, high = float(spec["low"]), float(spec["high"])
            return float(low + rng.rand() * (high - low))
        if kind == "loguniform":
            low, high = float(spec["low"]), float(spec["high"])
            if not (low > 0 and high > 0):
                raise ValueError("loguniform requires low>0 and high>0.")
            loglow, loghigh = np.log(low), np.log(high)
            return float(np.exp(loglow + rng.rand() * (loghigh - loglow)))
        raise ValueError(f"Unsupported dist {kind}")

    if not dists and strategy != "grid":
        all_combos = list(ParameterGrid(values_space)) if values_space else [dict()]
        rng.shuffle(all_combos)
        for p in all_combos[: min(n_iter, len(all_combos))]:
            yield p
        return

    if strategy == "grid":
        for grid_params in ParameterGrid(values_space or {}):
            params = dict(grid_params)
            for k, spec in dists.items():
                params[k] = _sample_from_dist(spec)
            yield params
    else:
        for _ in range(n_iter):
            params = {k: rng.choice(v) for k, v in values_space.items()}
            for k, spec in dists.items():
                params[k] = _sample_from_dist(spec)
            yield params


# =============================================================================
# Preprocessing
# =============================================================================


def preprocess_fit(X_tr: np.ndarray, cfg_block: Dict[str, Any]) -> Dict[str, Any]:
    """Fit preprocessing on TRAIN; return fitted transformers for OOS."""
    out = {"keep_sparse_idx": None, "wins": None, "imputer": None, "varth": None, "yj": None, "scaler": None}
    cfg_block = cfg_block or {}

    min_frac = float(cfg_block.get("min_non_null_frac", 0.0))
    if min_frac > 0.0:
        finite_mask = np.isfinite(X_tr)
        denom = np.maximum(1, finite_mask.shape[0])
        frac = finite_mask.sum(axis=0) / denom
        keep = np.where(frac >= min_frac)[0]
    else:
        keep = np.arange(X_tr.shape[1], dtype=int)
    out["keep_sparse_idx"] = keep
    X = X_tr[:, keep] if keep.size else X_tr

    wins = cfg_block.get("winsorize")
    if wins:
        lo = max(0.0, min(_as_float(wins.get("lower_q", 0.0), 0.0), 0.5))
        hi = min(1.0, max(_as_float(wins.get("upper_q", 1.0), 1.0), 0.5))
        qlo = np.nanquantile(X, lo, axis=0)
        qhi = np.nanquantile(X, hi, axis=0)
        qlo = np.where(np.isfinite(qlo), qlo, np.nanmin(X, axis=0))
        qhi = np.where(np.isfinite(qhi), qhi, np.nanmax(X, axis=0))
        out["wins"] = (qlo, qhi)
        X = np.clip(X, qlo, qhi)

    imp_mode = _as_str_choice(cfg_block.get("impute", "none"), {"none", "mean", "median"}, "none")
    if imp_mode != "none":
        imp = SimpleImputer(strategy=imp_mode)
        X = imp.fit_transform(X)
        out["imputer"] = imp

    thr = _as_float(cfg_block.get("variance_threshold", 0.0), 0.0)
    if thr > 0.0:
        vt = VarianceThreshold(threshold=thr)
        X = vt.fit_transform(X)
        out["varth"] = vt

    if _as_bool(cfg_block.get("yeo_johnson", False), False):
        yj = PowerTransformer(method="yeo-johnson", standardize=False)
        X = yj.fit_transform(X)
        out["yj"] = yj

    if _as_bool(cfg_block.get("standardize", False), False):
        sc = StandardScaler(with_mean=True, with_std=True)
        X = sc.fit_transform(X)
        out["scaler"] = sc

    return out


def preprocess_apply(X: np.ndarray, pp: Dict[str, Any]) -> np.ndarray:
    if pp.get("keep_sparse_idx") is not None:
        idx = pp["keep_sparse_idx"]
        X = X[:, idx] if len(idx) else X
    if pp.get("wins") is not None:
        qlo, qhi = pp["wins"]
        X = np.clip(X, qlo, qhi)
    if pp.get("imputer") is not None:
        X = pp["imputer"].transform(X)
    if pp.get("varth") is not None:
        X = pp["varth"].transform(X)
    if pp.get("yj") is not None:
        X = pp["yj"].transform(X)
    if pp.get("scaler") is not None:
        X = pp["scaler"].transform(X)
    return X


def feature_names_after_pp(pp: Dict[str, Any], feat_cols: List[str]) -> List[str]:
    names = list(feat_cols)
    if pp.get("keep_sparse_idx") is not None:
        names = [names[i] for i in pp["keep_sparse_idx"]]
    if pp.get("varth") is not None:
        try:
            support = pp["varth"].get_support(indices=True)
            names = [names[i] for i in support]
        except Exception:
            pass
    return names


# =============================================================================
# CV / training
# =============================================================================


def _filter_finite_y(
    X: Optional[np.ndarray],
    y: np.ndarray,
    ids: Optional[np.ndarray] = None,
    split_name: str = "",
):
    mask = np.isfinite(y)
    removed = int((~mask).sum())
    if removed > 0:
        LOG.warning("Removed %d rows with non-finite y in %s split.", removed, split_name or "?")
    Xf = X[mask] if X is not None else None
    yf = y[mask]
    idf = ids[mask] if ids is not None else None
    return Xf, yf, idf, removed, mask


def _cv_splits_with_dates(
    X: np.ndarray,
    dates: Optional[np.ndarray],
    *,
    n_splits: int,
    gap: int,
    min_train_size: int = 30,
):
    """Date-grouped, forward-chaining CV with a calendar-day embargo."""
    if dates is None:
        raise ValueError("dates must be provided for date-grouped CV.")

    dser = pd.Series(pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).normalize(), index=np.arange(len(dates)))
    u = pd.Index(dser.sort_values().unique())
    n_days = len(u)

    if n_days <= max(2, min_train_size + 1):
        cut = max(min_train_size, n_days - max(1, n_days // 5))
        tr_end = max(0, cut - 1 - max(0, gap))
        tr_dates = u[: tr_end + 1]
        va_dates = u[cut:]
        tr = dser.index[dser.isin(tr_dates)].to_numpy()
        va = dser.index[dser.isin(va_dates)].to_numpy()
        return [(tr, va)] if (tr.size and va.size) else []

    val_blocks = np.array_split(np.arange(n_days), n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for block in val_blocks:
        if len(block) == 0:
            continue
        first_val_pos = int(block[0])
        last_train_pos = first_val_pos - 1 - max(0, gap)
        if last_train_pos + 1 < min_train_size:
            continue
        tr_dates = u[: last_train_pos + 1]
        va_dates = u[block]
        tr_idx = dser.index[dser.isin(tr_dates)].to_numpy()
        va_idx = dser.index[dser.isin(va_dates)].to_numpy()
        if tr_idx.size and va_idx.size:
            assert not np.intersect1d(dser.iloc[tr_idx].unique(), dser.iloc[va_idx].unique()).size
            splits.append((tr_idx, va_idx))

    if not splits:
        val_block = np.arange(int(0.8 * n_days), n_days)
        first_val_pos = int(val_block[0])
        last_train_pos = first_val_pos - 1 - max(0, gap)
        if last_train_pos + 1 >= max(1, min_train_size):
            tr_dates = u[: last_train_pos + 1]
            va_dates = u[val_block]
            tr_idx = dser.index[dser.isin(tr_dates)].to_numpy()
            va_idx = dser.index[dser.isin(va_dates)].to_numpy()
            if tr_idx.size and va_idx.size:
                splits = [(tr_idx, va_idx)]

    return splits


def fit_family(
    registry: Dict[str, Any],
    X_tr,
    z_tr,
    X_va,
    z_va,
    family_names: List[str],
    strategy: str,
    n_iter: int,
    keep_top_k: int,
    seed: int,
    cv_records: List[Dict[str, Any]],
    dates_tr=None,
    dates_va=None,
    cv_cfg: Optional[Dict[str, Any]] = None,
    inverse_to_target_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    y_cv_target: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Model selection per family using time-grouped CV; target-space scoring if available."""
    import inspect
    try:
        from sklearn.pipeline import Pipeline as SkPipeline
    except Exception:
        SkPipeline = tuple()

    def _fit_with_optional_weight(est, X, y, sample_weight):
        if sample_weight is None:
            est.fit(X, y)
            return
        try:
            sig = inspect.signature(est.fit)
            if "sample_weight" in sig.parameters:
                est.fit(X, y, sample_weight=np.asarray(sample_weight))
                return
        except Exception:
            pass
        if isinstance(est, SkPipeline):
            last_name, last_step = est.steps[-1]
            params = {}
            try:
                sig_last = inspect.signature(last_step.fit)
                if "sample_weight" in sig_last.parameters:
                    params[f"{last_name}__sample_weight"] = np.asarray(sample_weight)
                    est.fit(X, y, **params)
                    return
            except Exception:
                pass
            for nm, step in est.steps:
                try:
                    if "sample_weight" in inspect.signature(step.fit).parameters:
                        params[f"{nm}__sample_weight"] = np.asarray(sample_weight)
                except Exception:
                    continue
            if params:
                est.fit(X, y, **params)
                return
        est.fit(X, y)

    kept: Dict[str, Any] = {}
    cv_cfg = cv_cfg or {}
    n_splits = int(cv_cfg.get("n_splits", 5))
    gap = int(cv_cfg.get("gap", 0))
    min_train_days = int(cv_cfg.get("min_train_size", 30))
    use_trva = bool(cv_cfg.get("use_train_plus_val", True))

    if use_trva and (dates_tr is not None and dates_va is not None):
        X_cv = np.vstack([X_tr, X_va])
        z_cv = np.hstack([z_tr, z_va])
        d_cv = np.hstack([np.asarray(dates_tr), np.asarray(dates_va)])
    else:
        X_cv = X_tr
        z_cv = z_tr
        d_cv = np.asarray(dates_tr) if dates_tr is not None else None

    def _score_fold(z_true_val, z_pred_val, va_idx):
        if (inverse_to_target_fn is not None) and (y_cv_target is not None):
            try:
                y_va = y_cv_target[va_idx]
                yhat = inverse_to_target_fn(z_pred_val, va_idx)
                sse = float(np.mean((y_va - yhat) ** 2))
                sst = float(np.mean(y_va ** 2)) if np.any(np.isfinite(y_va)) else np.nan
                return 1.0 - (sse / sst if (sst and np.isfinite(sst)) else np.nan)
            except Exception as e:
                LOG.warning("Target-space scoring failed, z-space fallback: %s", e)
        num = float(np.mean((z_true_val - z_pred_val) ** 2))
        den = float(np.mean(z_true_val ** 2)) if np.any(np.isfinite(z_true_val)) else np.nan
        return 1.0 - (num / den if (den and np.isfinite(den)) else np.nan)

    splits = _cv_splits_with_dates(
        X_cv, d_cv, n_splits=n_splits, gap=gap, min_train_size=min_train_days
    )
    if not splits:
        LOG.warning(
            "No valid CV splits (n_splits=%d, gap=%d, min_train_days=%d). "
            "Fallback to last-split holdout.",
            n_splits,
            gap,
            min_train_days,
        )
        n = X_cv.shape[0]
        cut = max(1, int(0.8 * n))
        splits = [(np.arange(cut), np.arange(cut, n))]

    for name in family_names:
        if name not in registry:
            continue
        spec = registry[name]
        recs = []
        for params in _iter_param_sets(spec.grid_default, strategy=strategy, n_iter=n_iter, seed=seed):
            fold_scores = []
            best_iters = []
            for tr_idx, va_idx in splits:
                Xtr, Xva = X_cv[tr_idx], X_cv[va_idx]
                ztr, zva = z_cv[tr_idx], z_cv[va_idx]
                est = spec.factory()
                if hasattr(est, "set_params"):
                    est.set_params(**params)
                setattr(est, "_fit_params_used", dict(params))
                w_tr = _equal_day_weights(d_cv[tr_idx]) if d_cv is not None else None
                w_va = _equal_day_weights(d_cv[va_idx]) if d_cv is not None else None

                clsname = est.__class__.__name__.lower()
                is_lgbm = clsname.startswith("lgbm")
                if is_lgbm:
                    fit_kwargs = dict(eval_set=[(Xva, zva)], eval_metric="l2")
                    if w_va is not None:
                        fit_kwargs["eval_sample_weight"] = [np.asarray(w_va)]
                    try:
                        import lightgbm as lgb
                        fit_kwargs["callbacks"] = [
                            lgb.early_stopping(stopping_rounds=32, first_metric_only=True, verbose=False),
                            lgb.log_evaluation(period=0),
                        ]
                    except Exception:
                        pass
                    try:
                        est.set_params(verbose=-1)
                    except Exception:
                        pass
                    try:
                        est.set_params(verbosity=-1)
                    except Exception:
                        pass
                    try:
                        est.fit(
                            Xtr,
                            ztr,
                            sample_weight=(np.asarray(w_tr) if w_tr is not None else None),
                            **fit_kwargs,
                        )
                    except TypeError:
                        fit_kwargs.pop("eval_sample_weight", None)
                        try:
                            est.fit(
                                Xtr,
                                ztr,
                                sample_weight=(np.asarray(w_tr) if w_tr is not None else None),
                                **fit_kwargs,
                            )
                        except TypeError:
                            fit_kwargs.pop("callbacks", None)
                            est.fit(
                                Xtr,
                                ztr,
                                sample_weight=(np.asarray(w_tr) if w_tr is not None else None),
                                **fit_kwargs,
                            )
                    best_it = getattr(est, "best_iteration_", None)
                    if best_it is not None and isinstance(best_it, (int, np.integer)) and best_it > 0:
                        best_iters.append(int(best_it))
                        zhat_va = est.predict(Xva, num_iteration=int(best_it))
                    else:
                        zhat_va = est.predict(Xva)
                else:
                    _fit_with_optional_weight(est, Xtr, ztr, w_tr)
                    zhat_va = est.predict(Xva)

                fold_scores.append(_score_fold(zva, zhat_va, va_idx))

            score = float(np.nanmean(fold_scores)) if fold_scores else -np.inf
            chosen_n = int(np.median(best_iters)) if best_iters else None
            recs.append((score, params, chosen_n))
            cv_records.append(
                {
                    "model": name,
                    "params": dict(params),
                    "score_mean": score,
                    "fold_scores": list(map(float, fold_scores)),
                    "n_splits": n_splits,
                    "gap": gap,
                    "min_train_days": min_train_days,
                    "use_trva": use_trva,
                    "best_iteration_median": chosen_n,
                }
            )

        if not recs:
            continue
        recs.sort(key=lambda x: (-(np.inf if np.isnan(x[0]) else x[0]),))
        top = recs[: max(1, keep_top_k)]

        for j, (best_score, best_params, best_n_rounds) in enumerate(top, start=1):
            if use_trva and (dates_tr is not None and dates_va is not None):
                X_fit = np.vstack([X_tr, X_va])
                z_fit = np.hstack([z_tr, z_va])
                d_fit = np.hstack([dates_tr, dates_va])
            else:
                X_fit = X_tr
                z_fit = z_tr
                d_fit = np.asarray(dates_tr) if dates_tr is not None else None

            est = spec.factory()
            if hasattr(est, "set_params"):
                est.set_params(**best_params)
            setattr(est, "_fit_params_used", dict(best_params))
            if est.__class__.__name__.lower().startswith("lgbm") and (best_n_rounds is not None) and (best_n_rounds > 0):
                try:
                    est.set_params(n_estimators=int(best_n_rounds))
                except Exception:
                    pass
            _fit_with_optional_weight(est, X_fit, z_fit, _equal_day_weights(d_fit) if d_fit is not None else None)
            clean = name if keep_top_k == 1 else f"{name}#{j}"
            kept[clean] = est
            LOG.info(
                "[keep] %s score=%.5f params=%s%s",
                clean,
                best_score,
                best_params,
                (f" | n_estimators(refit)={best_n_rounds}" if best_n_rounds else ""),
            )
    return kept


# =============================================================================
# Exports
# =============================================================================


def _safe_jsonable(obj):
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__class__"):
        cls = obj.__class__.__name__
        try:
            if hasattr(obj, "get_params"):
                params = obj.get_params(deep=False)
                lite = {
                    k: v
                    for k, v in params.items()
                    if isinstance(
                        v, (type(None), str, bool, int, float, np.integer, np.floating, np.bool_)
                    )
                }
                return {"__estimator__": cls, "params": _safe_jsonable(lite)}
        except Exception:
            pass
        return f"<{cls}>"
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def export_models_table(models: Dict[str, Any], family: str, out_dir: str) -> None:
    rows = []
    for name, est in models.items():
        is_ens = hasattr(est, "members")
        if is_ens:
            for j, (mname, base) in enumerate(est.members.items(), 1):
                est_name = base.__class__.__name__
                params = getattr(base, "_fit_params_used", {})
                rows.append(
                    {
                        "model": mname,
                        "family": family,
                        "is_ensemble": False,
                        "member_of": name,
                        "member_rank": j,
                        "estimator": est_name,
                        "params_json": json.dumps(_safe_jsonable(params), ensure_ascii=False),
                    }
                )
        else:
            est_name = est.__class__.__name__
            params = getattr(est, "_fit_params_used", {})
            rows.append(
                {
                    "model": name,
                    "family": family,
                    "is_ensemble": False,
                    "member_of": "",
                    "member_rank": "",
                    "estimator": est_name,
                    "params_json": json.dumps(_safe_jsonable(params), ensure_ascii=False),
                }
            )
    if rows:
        df = pd.DataFrame(rows)
        out_name = "models_linear.csv" if family == "linear" else "models_nonlinear.csv"
        df.to_csv(os.path.join(out_dir, out_name), index=False)


def _linear_coefficients_for_estimator(est) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if hasattr(est, "coef_"):
        coef = np.ravel(est.coef_)
        intercept = float(getattr(est, "intercept_", 0.0))
        return coef, intercept
    if hasattr(est, "steps"):
        for _, sub in est.steps:
            if hasattr(sub, "coef_"):
                coef = np.ravel(sub.coef_)
                intercept = float(getattr(sub, "intercept_", 0.0))
                return coef, intercept
    return None, None


def export_linear_coefficients(
    models_linear: Dict[str, Any], feature_names_after_linear_pp: List[str], out_dir: str
) -> None:
    base_dir = ensure_dir(os.path.join(out_dir, "coefficients_linear"))
    feat_names = list(feature_names_after_linear_pp)
    for name, est in models_linear.items():
        if hasattr(est, "members"):
            continue
        coef, intercept = _linear_coefficients_for_estimator(est)
        if coef is None or coef.shape[0] != len(feat_names):
            continue
        rows = [{"feature": f, "coef": float(w), "intercept": float(intercept)} for f, w in zip(feat_names, coef)]
        pd.DataFrame(rows).to_csv(os.path.join(base_dir, f"{name}.csv"), index=False)


def export_diag_all(
    y_true_target: np.ndarray,
    y_true_raw: np.ndarray,
    preds_target: Dict[str, np.ndarray],
    out_dir: str,
    ids: np.ndarray,
):
    df = pd.DataFrame({"group": ids, "y_target": y_true_target, "y_raw": y_true_raw})
    for k, v in preds_target.items():
        df[f"yhat_{k}"] = v
        df[f"resid_{k}"] = df["y_target"] - v
    df.to_csv(os.path.join(out_dir, "diag_all.csv"), index=False)


def export_metrics(
    y_true_target: np.ndarray,
    preds_target: Dict[str, np.ndarray],
    ids: np.ndarray,
    out_dir: str,
    y_true_winsor_aligned: Optional[np.ndarray] = None,
    *,
    filename: str = "metrics.csv",
):
    rows = []
    for k, v in preds_target.items():
        rows.append(
            {
                "model": k,
                "R2_OS": r2_os(y_true_target, v),
                "R2_OS_XS": r2_os_xs(y_true_target, v, ids),
                "CW_t": clark_west_t(y_true_target, v, groups=ids),
                "CW_t_XS": clark_west_t_xs(y_true_target, v, groups=ids),
                "R2_OS_day": r2_os_day_weighted(y_true_target, v, ids),
                "R2_OS_winsor": (
                    r2_os(y_true_winsor_aligned, v) if y_true_winsor_aligned is not None else r2_os(y_true_target, v)
                ),
            }
        )
    pd.DataFrame(rows).sort_values("R2_OS", ascending=False).to_csv(os.path.join(out_dir, filename), index=False)


# --- export_metrics_by_contract_bins, export_portfolios_raw, SHAP helpers ---
# These are unchanged from your pasted script.
# To keep this response usable in one message, they are included verbatim below.

# =============================================================================
# Metrics by contract bins (unchanged)
# =============================================================================

def export_metrics_by_contract_bins(
    y_true_target: np.ndarray,
    preds_target: Dict[str, np.ndarray],
    ids: np.ndarray,
    df_te_small: pd.DataFrame,
    out_dir: str,
    *,
    tcol: str,
    mcol: str,
    band: Tuple[float, float],
    y_true_winsor_aligned: Optional[np.ndarray] = None,
):
    n_obs = int(len(y_true_target))
    if len(ids) != n_obs:
        raise ValueError(f"ids length {len(ids)} != y length {n_obs}")
    if len(df_te_small) != n_obs:
        raise ValueError(f"df_te_small length {len(df_te_small)} != y length {n_obs}")
    for k, v in preds_target.items():
        if len(v) != n_obs:
            raise ValueError(f"preds_target['{k}'] length {len(v)} != y length {n_obs}")

    y = np.asarray(y_true_target, dtype=float)
    id_arr = np.asarray(ids)

    def norm_type_any(x):
        try:
            if isinstance(x, (int, float, np.number)):
                v = float(x)
                if np.isnan(v):
                    return "put"
                if v > 0:
                    return "call"
                if v < 0:
                    return "put"
                return "put" if v == 0 else "call"
            s = str(x).strip().lower()
            if s in {"c", "call", "calls", "1", "true", "t", "yes", "y"}:
                return "call"
            if s in {"p", "put", "puts", "-1", "0", "false", "f", "no", "n"}:
                return "put"
            return "put"
        except Exception:
            return "put"

    types = (
        df_te_small[tcol].map(norm_type_any)
        if tcol in df_te_small.columns
        else pd.Series(["call"] * n_obs, index=df_te_small.index)
    )
    mny = (
        pd.to_numeric(df_te_small[mcol], errors="coerce")
        if mcol in df_te_small.columns
        else pd.Series([np.nan] * n_obs, index=df_te_small.index)
    )

    types_arr = types.to_numpy()
    mny_arr = mny.to_numpy(dtype=float)

    lo, hi = band
    name = str(mcol).lower()
    is_logratio = ("log" in name) and ("std" not in name)
    is_standardized = ("std" in name) or ("degree" in name)

    if is_logratio:
        lo_t, hi_t = np.log(lo), np.log(hi)
        atm_mask = (mny_arr >= lo_t) & (mny_arr <= hi_t)
        itm_call = (types_arr == "call") & (mny_arr < lo_t)
        itm_put = (types_arr == "put") & (mny_arr > hi_t)
    elif is_standardized:
        atm_mask = (mny_arr >= lo) & (mny_arr <= hi)
        itm_call = (types_arr == "call") & (mny_arr < lo)
        itm_put = (types_arr == "put") & (mny_arr > hi)
    else:
        atm_mask = (mny_arr >= lo) & (mny_arr <= hi)
        itm_call = (types_arr == "call") & (mny_arr > hi)
        itm_put = (types_arr == "put") & (mny_arr < lo)

    otm_call = (types_arr == "call") & ~(atm_mask | itm_call)
    otm_put = (types_arr == "put") & ~(atm_mask | itm_put)
    itm_mask_all = itm_call | itm_put
    otm_mask_all = otm_call | otm_put

    valid_y = np.isfinite(y)
    atm_mask &= valid_y
    itm_mask_all &= valid_y
    otm_mask_all &= valid_y

    side_order = ["all", "call", "put"]
    mny_order = ["all", "itm", "atm", "otm"]

    def bin_mask(name_):
        if name_ == "all":
            return np.ones(n_obs, dtype=bool) & valid_y
        if name_ == "itm":
            return itm_mask_all
        if name_ == "atm":
            return atm_mask
        if name_ == "otm":
            return otm_mask_all
        raise ValueError(name_)

    rows = []
    for side in side_order:
        side_mask = (np.ones(n_obs, dtype=bool) if side == "all" else (types_arr == side)) & valid_y

        for bucket in mny_order:
            bmask = bin_mask(bucket)
            mask = side_mask & bmask
            n = int(mask.sum())
            if n == 0:
                for k in preds_target:
                    rows.append(
                        {"side": side, "moneyness": bucket, "model": k, "n": 0, "R2_OS": np.nan, "R2_OS_XS": np.nan, "CW_t": np.nan}
                    )
                continue

            yb = y[mask]
            idb = id_arr[mask]
            for k, v in preds_target.items():
                vb = np.asarray(v, dtype=float)[mask]
                good = np.isfinite(yb) & np.isfinite(vb)
                if good.sum() == 0:
                    R2 = R2x = CW = R2d = R2w = np.nan
                    CWx = np.nan
                else:
                    yg = yb[good]
                    vg = vb[good]
                    idg = idb[good]
                    R2 = r2_os(yg, vg)
                    R2x = r2_os_xs(yg, vg, idg)
                    CW = clark_west_t(yg, vg, groups=idg)
                    CWx = clark_west_t_xs(yg, vg, groups=idg)
                    R2d = r2_os_day_weighted(yg, vg, idg)
                    if y_true_winsor_aligned is not None:
                        yg_w = y_true_winsor_aligned[mask][good]
                        R2w = r2_os(yg_w, vg)
                    else:
                        lo_ = np.nanmin(yg)
                        hi_ = np.nanmax(yg)
                        R2w = r2_os_winsor_aligned(yg, vg, lo_, hi_)

                rows.append(
                    {
                        "side": side,
                        "moneyness": bucket,
                        "model": k,
                        "n": n,
                        "R2_OS": R2,
                        "R2_OS_XS": R2x,
                        "CW_t": CW,
                        "CW_t_XS": CWx,
                        "R2_OS_day": R2d,
                        "R2_OS_winsor": R2w,
                    }
                )

    df_out = pd.DataFrame(rows)
    df_out["side"] = pd.Categorical(df_out["side"], categories=side_order, ordered=True)
    df_out["moneyness"] = pd.Categorical(df_out["moneyness"], categories=mny_order, ordered=True)
    df_out = df_out.sort_values(["side", "moneyness", "R2_OS"], ascending=[True, True, False])

    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(os.path.join(out_dir, "metrics_by_contract.csv"), index=False)
    return df_out


# =============================================================================
# export_portfolios_raw, SHAP tools: unchanged in your file
# =============================================================================
# (Your export_portfolios_raw, try_export_shap, export_shap_n_en are unchanged and should remain here.)

def export_portfolios_raw(
    y,
    preds_scores: Dict[str, Union[np.ndarray, pd.Series]],
    ids,
    out_dir: str,
    n_bins: int = 10,
    opt_rel_spread: Optional[Union[np.ndarray, pd.Series, float]] = None,
    commission_bps: Union[float, np.ndarray, pd.Series] = 0.0,
    delta: Optional[Union[np.ndarray, pd.Series]] = None,
    fut_cost_bps: Union[float, np.ndarray, pd.Series] = 0.0,
    annualize: bool = True,
    regroup_freq: Optional[str] = None,
    y_unit: str = "per_dollar",
    verbose: bool = False,
    *,
    normalize_group_to_day: bool = True,
    min_bin_size_per_side: int = 5,
    write_ic: bool = True,
):
    """
    L/S portfolios from per-dollar y and model scores.
    Robust defaults: per-day grouping, min bin size per side, inferred annualization.
    """
    os.makedirs(out_dir, exist_ok=True)

    y_ser = pd.to_numeric(pd.Series(y), errors="coerce")
    group_dt = pd.to_datetime(pd.Series(ids), errors="coerce")
    if normalize_group_to_day and regroup_freq is None:
        group_dt = group_dt.dt.normalize()

    base = pd.DataFrame({"group": group_dt, "y": y_ser})
    if base["group"].isna().any():
        raise ValueError("Some 'ids' could not be converted to datetime.")
    N = len(base)

    def _to_series(x, name):
        if x is None:
            return pd.Series(np.nan, index=base.index, name=name)
        s = pd.Series(x)
        if s.size == 1 and N > 1:
            s = pd.Series(np.repeat(float(s.iloc[0]), N))
        s = pd.to_numeric(s, errors="coerce")
        s.name = name
        if len(s) != N:
            raise ValueError(f"Length mismatch for '{name}': got {len(s)}, expected {N}")
        return s

    if y_unit.lower() == "per_dollar":
        base["opt_rel_spread"] = (
            _to_series(opt_rel_spread, "opt_rel_spread") if opt_rel_spread is not None else pd.Series(0.0, index=base.index)
        )
        base["commission_bps"] = _to_series(commission_bps, "commission_bps")
        base["delta"] = _to_series(delta, "delta") if delta is not None else pd.Series(0.0, index=base.index)
        base["fut_cost_bps"] = _to_series(fut_cost_bps, "fut_cost_bps")
        commission = (base["commission_bps"] / 1e4).fillna(0.0)
        fut_cost = (base["fut_cost_bps"] / 1e4 * base["delta"].abs()).fillna(0.0)
        spread = base["opt_rel_spread"].fillna(0.0)
        base["y_net"] = base["y"] - spread - commission - fut_cost
    else:
        base["y_net"] = base["y"]
        if verbose:
            print("[INFO] y_unit != 'per_dollar'; costs ignored.")

    def _bin_within_group(series: pd.Series, q: int) -> pd.Series:
        r = series.rank(method="first")
        try:
            return pd.qcut(r, q=q, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=series.index)

    daily_frames = []
    kept_stats = []
    ic_frames = []

    for model_name, score in preds_scores.items():
        df = base.copy()
        df["score"] = pd.to_numeric(pd.Series(score), errors="coerce")
        df = df[np.isfinite(df["y_net"]) & np.isfinite(df["score"]) & df["group"].notna()]
        if df.empty:
            continue

        df["bin"] = df.groupby("group", observed=True)["score"].transform(lambda s: _bin_within_group(s, n_bins))

        tmp = df.dropna(subset=["bin"]).groupby(["group", "bin"], observed=True).size().unstack(fill_value=0)
        if tmp.empty:
            continue
        bmin = tmp.columns.min()
        bmax = tmp.columns.max()
        good = tmp[(tmp.get(bmin, 0) >= min_bin_size_per_side) & (tmp.get(bmax, 0) >= min_bin_size_per_side)].index
        df = df[df["group"].isin(good)]
        if df.empty:
            continue

        per_day = []
        day_ic = []
        for g, d in df.groupby("group", sort=True):
            bmin_g, bmax_g = d["bin"].min(), d["bin"].max()
            top_vals = d.loc[d["bin"] == bmax_g, "y_net"]
            bot_vals = d.loc[d["bin"] == bmin_g, "y_net"]
            top = top_vals.mean()
            bot = bot_vals.mean()
            if np.isfinite(top) and np.isfinite(bot):
                per_day.append({"group": pd.Timestamp(g), "model": model_name, "ls_return": float(top - bot)})
            if write_ic and len(d) >= 5:
                ic = d[["score", "y_net"]].rank().corr(method="pearson").loc["score", "y_net"]
                if np.isfinite(ic):
                    day_ic.append({"group": pd.Timestamp(g), "model": model_name, "ic": float(ic)})

        if not per_day:
            continue

        dm = pd.DataFrame(per_day).sort_values(["model", "group"])

        if regroup_freq:
            dm["group_reg"] = pd.to_datetime(dm["group"]).dt.to_period(regroup_freq).dt.to_timestamp()
            dm = dm.groupby(["model", "group_reg"], as_index=False)["ls_return"].mean().rename(columns={"group_reg": "group"})

        daily_frames.append(dm)

        if write_ic and day_ic:
            ic_frames.append(pd.DataFrame(day_ic))

        if verbose:
            n_groups = dm["group"].nunique()
            kept_stats.append({"model": model_name, "groups_kept": int(n_groups)})

    if not daily_frames:
        pd.DataFrame(columns=["group", "model", "ls_return"]).to_csv(os.path.join(out_dir, "portfolios_daily.csv"), index=False)
        pd.DataFrame(columns=["model", "N", "mean", "vol", "Sharpe_ann", "Hit_ratio"]).to_csv(
            os.path.join(out_dir, "portfolio_metrics.csv"), index=False
        )
        if verbose:
            print("[INFO] No groups survived for any model; wrote empty outputs.")
        return

    daily = pd.concat(daily_frames, ignore_index=True).sort_values(["model", "group"])
    daily.to_csv(os.path.join(out_dir, "portfolios_daily.csv"), index=False)

    def _infer_ann_factor(series_of_timestamps: pd.Series) -> float:
        u = pd.to_datetime(series_of_timestamps.dropna().unique())
        if u.size < 3:
            return 252.0
        u = np.sort(u)
        per_year = pd.Series(u).dt.year.value_counts().sort_index()
        return float(per_year.mean()) if per_year.size else 252.0

    rows = []
    for m, dm in daily.groupby("model"):
        g = pd.to_datetime(dm["group"])
        r = pd.to_numeric(dm["ls_return"], errors="coerce").to_numpy()
        r = r[np.isfinite(r)]
        N = r.size
        mu = float(np.nanmean(r)) if N else np.nan
        sd = float(np.nanstd(r, ddof=1)) if N > 1 else np.nan
        sharpe_period = (mu / sd) if (sd and np.isfinite(sd) and sd > 0) else np.nan
        ann_fac = _infer_ann_factor(g) if annualize else 1.0
        sharpe_ann = float(sharpe_period * np.sqrt(ann_fac)) if (annualize and np.isfinite(sharpe_period)) else (
            float(sharpe_period) if np.isfinite(sharpe_period) else np.nan
        )
        hit = float(np.mean(r > 0)) if N else np.nan
        rows.append({"model": m, "N": int(N), "mean": mu, "vol": sd, "Sharpe_ann": sharpe_ann, "Hit_ratio": hit})

    pd.DataFrame(rows).sort_values("model").to_csv(os.path.join(out_dir, "portfolio_metrics.csv"), index=False)

    if verbose and kept_stats:
        pd.DataFrame(kept_stats).sort_values("model").to_csv(os.path.join(out_dir, "debug_groups_kept.csv"), index=False)

    if write_ic and ic_frames:
        ic_all = pd.concat(ic_frames, ignore_index=True)
        ic_daily = ic_all.groupby("model")["ic"].agg(IC_mean="mean", IC_std=lambda s: s.std(ddof=1), N="count").reset_index()
        ic_all.to_csv(os.path.join(out_dir, "daily_ic.csv"), index=False)
        ic_daily.to_csv(os.path.join(out_dir, "ic_summary.csv"), index=False)


# =============================================================================
# SHAP (optional)
# =============================================================================

def _shap_background(X: np.ndarray, k_bg: int):
    try:
        import shap
    except Exception:
        return X
    X = np.asarray(X, dtype=np.float64, order="C")
    n = X.shape[0]
    if n == 0:
        return X
    k = max(2, min(int(k_bg), n))
    try:
        return shap.kmeans(X, k)
    except Exception:
        idx = np.random.choice(n, size=k, replace=False)
        return X[idx]



def try_export_shap(
    models: Dict[str, Any],
    X: np.ndarray,
    feature_names: List[str],
    out_dir: str,
    model_name: str,
    cfg: Dict[str, Any] = None,
):
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        LOG.info("SHAP not installed; skipping SHAP.")
        return

    if model_name not in models:
        return

    # SHAP config
    sconf = (cfg or {}).get("explain", {}).get("shap", {}) if cfg else {}
    n_rows = int(sconf.get("sample_rows", 1500))
    k_bg = int(sconf.get("background_kmeans", 50))
    check_add = bool(sconf.get("check_additivity", False))
    per_member = bool(sconf.get("per_member", True))
    max_display = int(sconf.get("max_display", 10))  # 9 for thesis-style top features

    # Subsample rows for SHAP
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] > n_rows:
        idx = np.random.permutation(X.shape[0])[:n_rows]
        X_ex = X[idx]
    else:
        X_ex = X

    if X_ex.shape[0] == 0:
        LOG.warning("No rows for SHAP explanation for %s.", model_name)
        return

    # Background for SHAP
    X_bg_obj = _shap_background(X, k_bg)
    X_bg = getattr(X_bg_obj, "data", X_bg_obj)
    X_bg = np.asarray(X_bg, dtype=np.float64, order="C")

    def _is_tree(m):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

        try:
            import lightgbm as lgb

            if isinstance(m, lgb.LGBMRegressor):
                return True
        except Exception:
            pass
        return isinstance(m, (RandomForestRegressor, GradientBoostingRegressor))

    def _is_linear(m):
        return hasattr(m, "coef_") or m.__class__.__name__.lower() in {
            "linearregression",
            "lasso",
            "ridge",
            "elasticnet",
        }

    def _shap_values_for(model, X_ex_local):
        import shap

        if _is_tree(model):
            expl = shap.TreeExplainer(model, feature_perturbation="interventional", data=X_bg)
            vals = expl.shap_values(X_ex_local, check_additivity=check_add)
        elif _is_linear(model):
            expl = shap.LinearExplainer(model, X_bg)
            vals = expl.shap_values(X_ex_local)
        else:
            mask = shap.maskers.Independent(X_bg)
            # Prefer predict() for generic models (e.g., FFN pipelines) to avoid
            # backend-specific explainers picking the wrong interface.
            pred_fn = getattr(model, "predict", None)
            if callable(pred_fn):
                expl = shap.Explainer(pred_fn, mask, algorithm="permutation")
            else:
                expl = shap.Explainer(model, mask)
            vals = expl(X_ex_local).values

        vals = np.asarray(vals)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        if vals.ndim == 3:
            vals = vals.mean(axis=0)
        return vals

    m = models[model_name]

    # Compute SHAP values (possibly averaging over ensemble members)
    if per_member and hasattr(m, "members"):
        vals_all, imps = [], []
        for nm, base in m.members.items():
            try:
                vals = _shap_values_for(base, X_ex)
                vals_all.append(vals)
                imps.append(np.abs(vals).mean(axis=0))
            except Exception as e:
                LOG.warning("SHAP failed for member %s (%s); skipping.", nm, e)
        if not imps:
            return
        shap_vals = np.mean(np.stack(vals_all, axis=0), axis=0)
        imp = np.mean(np.vstack(imps), axis=0)
    else:
        try:
            shap_vals = _shap_values_for(m, X_ex)
            imp = np.abs(shap_vals).mean(axis=0)
        except Exception as e:
            LOG.warning("SHAP failed for %s (%s).", model_name, e)
            return

    # Align feature names
    names = list(feature_names)
    if shap_vals.shape[1] != len(names):
        n_feat = shap_vals.shape[1]
        names = names[:n_feat] + [f"feat_{i}" for i in range(len(names), n_feat)]

    shap_dir = ensure_dir(out_dir)

    # Save mean |SHAP| ranking for all features
    (
        pd.DataFrame({"feature": names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .to_csv(os.path.join(shap_dir, f"shap_{model_name}.csv"), index=False)
    )

    # Save full SHAP matrix (obs × features) for later use
    np.save(os.path.join(shap_dir, f"shap_vals_{model_name}.npy"), shap_vals)

    # Plot top-k violin/beeswarm for the thesis
    try:
        import shap as shaplib

        plt.figure(figsize=(4, 8))
        shaplib.summary_plot(
            shap_vals,
            X_ex,
            feature_names=names,
            plot_type="violin",
            show=False,
            max_display=max_display,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(shap_dir, f"shap_violin_top{max_display}_{model_name}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        LOG.warning("SHAP plotting failed for %s (%s).", model_name, e)
        return

def export_shap_n_en(
    all_models: Dict[str, Any],
    best_by_algo: Dict[str, Tuple[float, str]],
    X_te_non: np.ndarray,
    X_te_ffn: np.ndarray,
    df_te: pd.DataFrame,
    mask_te: np.ndarray,
    feat_names_non: List[str],
    feat_names_ffn: List[str],
    out_dir: str,
    cfg: Dict[str, Any] = None,
) -> None:
    """
    Compute SHAP for the nonlinear ensemble N-En by averaging SHAP values
    over lgbm_gbdt, lgbm_dart, rf and ffn. FFN interaction SHAP is allocated
    back to parent base predictors using INTERACTION_PARENTS.
    """
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        LOG.info("SHAP not installed; skipping N-En SHAP.")
        return

    sconf = (cfg or {}).get("explain", {}).get("shap", {}) if cfg else {}
    n_rows = int(sconf.get("sample_rows", 1500))
    k_bg = int(sconf.get("background_kmeans", 50))
    check_add = bool(sconf.get("check_additivity", False))
    max_display = int(sconf.get("max_display", 9))

    # Basic sanity checks
    n = X_te_non.shape[0]
    if n == 0 or X_te_ffn.shape[0] != n:
        LOG.warning("N-En SHAP: empty test set or mismatched shapes; skipping.")
        return

    # Common subsample of observations for all four models
    idx = np.arange(n)
    if n > n_rows:
        idx = np.random.permutation(n)[:n_rows]

    X_non_ex = np.asarray(X_te_non, dtype=np.float32)[idx]
    X_ffn_ex = np.asarray(X_te_ffn, dtype=np.float32)[idx]

    # Background datasets
    X_bg_non_obj = _shap_background(X_te_non, k_bg)
    X_bg_non = np.asarray(getattr(X_bg_non_obj, "data", X_bg_non_obj), dtype=np.float64, order="C")

    X_bg_ffn_obj = _shap_background(X_te_ffn, k_bg)
    X_bg_ffn = np.asarray(getattr(X_bg_ffn_obj, "data", X_bg_ffn_obj), dtype=np.float64, order="C")

    def _is_tree(m):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

        try:
            import lightgbm as lgb

            if isinstance(m, lgb.LGBMRegressor):
                return True
        except Exception:
            pass
        return isinstance(m, (RandomForestRegressor, GradientBoostingRegressor))

    def _is_linear(m):
        return hasattr(m, "coef_") or m.__class__.__name__.lower() in {
            "linearregression",
            "lasso",
            "ridge",
            "elasticnet",
        }

    def _shap_values_for(model, X_ex_local, X_bg_local):
        if _is_tree(model):
            expl = shap.TreeExplainer(model, feature_perturbation="interventional", data=X_bg_local)
            vals = expl.shap_values(X_ex_local, check_additivity=check_add)
        elif _is_linear(model):
            expl = shap.LinearExplainer(model, X_bg_local)
            vals = expl.shap_values(X_ex_local)
        else:
            mask = shap.maskers.Independent(X_bg_local)
            pred_fn = getattr(model, "predict", None)
            if callable(pred_fn):
                expl = shap.Explainer(pred_fn, mask, algorithm="permutation")
            else:
                expl = shap.Explainer(model, mask)
            vals = expl(X_ex_local).values

        vals = np.asarray(vals)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        if vals.ndim == 3:
            vals = vals.mean(axis=0)
        return vals

    # Pick the concrete model keys for each base algorithm
    base_algos = ["lgbm_gbdt", "lgbm_dart", "rf", "ffn"]
    chosen: Dict[str, str] = {}

    for algo in base_algos:
        key = None
        # Prefer the best-per-algorithm key if available
        if algo in best_by_algo:
            key = best_by_algo[algo][1]
        # Fallback: any model whose name starts with the algo prefix
        if (not key) or (key not in all_models):
            cand = [k for k in all_models.keys() if k.split("#", 1)[0] == algo]
            if cand:
                key = cand[0]
        if not key or key not in all_models:
            LOG.warning("N-En SHAP: no model found for base algo '%s'; skipping it.", algo)
            continue
        chosen[algo] = key

    if len(chosen) < 2:
        LOG.warning("N-En SHAP: fewer than 2 base models found (%d); skipping.", len(chosen))
        return

    # Compute raw SHAP matrices per algo
    shap_raw: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for algo, key in chosen.items():
        model = all_models[key]
        if algo == "ffn":
            shap_vals = _shap_values_for(model, X_ffn_ex, X_bg_ffn)
            feat_list = list(feat_names_ffn)
        else:
            shap_vals = _shap_values_for(model, X_non_ex, X_bg_non)
            feat_list = list(feat_names_non)

        n_feat = shap_vals.shape[1]
        if n_feat != len(feat_list):
            # Keep in sync with SHAP output length
            feat_list = feat_list[:n_feat]
        shap_raw[algo] = (shap_vals, feat_list)

    # Build common base-feature space
    parent_feats = set()
    for parents in INTERACTION_PARENTS.values():
        parent_feats.update(parents)

    non_interaction_ffn = [f for f in feat_names_ffn if f not in INTERACTION_PARENTS]
    base_features = sorted(set(feat_names_non) | set(non_interaction_ffn) | parent_feats)
    if not base_features:
        LOG.warning("N-En SHAP: empty base-feature set; skipping.")
        return

    base_idx = {f: i for i, f in enumerate(base_features)}
    n_ex = X_non_ex.shape[0]
    n_base = len(base_features)

    def _collapse_to_base(shap_vals: np.ndarray, feat_list: List[str], treat_interactions: bool) -> np.ndarray:
        """Map model-specific SHAP to the common base-feature space."""
        phi = np.zeros((n_ex, n_base), dtype=float)
        for j, fname in enumerate(feat_list):
            contrib = shap_vals[:, j]
            if treat_interactions and fname in INTERACTION_PARENTS:
                parents = INTERACTION_PARENTS[fname]
                share = 1.0 / float(len(parents))
                for p in parents:
                    if p in base_idx:
                        phi[:, base_idx[p]] += share * contrib
            else:
                if fname in base_idx:
                    phi[:, base_idx[fname]] += contrib
        return phi

    shap_by_algo: Dict[str, np.ndarray] = {}
    for algo, (vals, flist) in shap_raw.items():
        if algo == "ffn":
            shap_by_algo[algo] = _collapse_to_base(vals, flist, treat_interactions=True)
        else:
            shap_by_algo[algo] = _collapse_to_base(vals, flist, treat_interactions=False)

    # Average over available base algorithms
    mats = list(shap_by_algo.values())
    shap_n_en = np.mean(np.stack(mats, axis=0), axis=0)
    imp = np.abs(shap_n_en).mean(axis=0)

    shap_dir = ensure_dir(out_dir)

    # Save mean |SHAP| ranking
    (
        pd.DataFrame({"feature": base_features, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .to_csv(os.path.join(shap_dir, "shap_N-En.csv"), index=False)
    )

    # Save SHAP matrix
    np.save(os.path.join(shap_dir, "shap_vals_N-En.npy"), shap_n_en)

    # Feature values for plotting: raw test features on the same rows
    df_te_masked = df_te.loc[mask_te].reset_index(drop=True)
    df_ex = df_te_masked.iloc[idx]
    try:
        X_ex_plot = df_ex[base_features].to_numpy(dtype=float)
    except Exception as e:
        LOG.warning(
            "N-En SHAP: could not build feature matrix for plotting (%s); using zeros.",
            e,
        )
        X_ex_plot = np.zeros_like(shap_n_en, dtype=float)

    # Violin / beeswarm plot for top-k features
    try:
        plt.figure(figsize=(4, 8))
        shap.summary_plot(
            shap_n_en,
            X_ex_plot,
            feature_names=base_features,
            plot_type="violin",
            show=False,
            max_display=max_display,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(shap_dir, f"shap_violin_top{max_display}_N-En.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        LOG.warning("N-En SHAP plotting failed (%s).", e)



# =============================================================================
# Helpers for driver
# =============================================================================


def _union_of_all_feature_groups(cfg: Dict[str, Any]) -> List[str]:
    groups_all = list((cfg["data"].get("feature_groups") or {}).keys())
    return (cfg["data"].get("features") or []) if not groups_all else select_features_by_groups(cfg, groups_all)


def build_predictions_frame(
    ids_te: np.ndarray,
    y_te_target: np.ndarray,
    y_te_raw: np.ndarray,
    preds_in_target_units: Dict[str, np.ndarray],
    *,
    df_te: pd.DataFrame,
    mask_te: np.ndarray,
    aux_candidates: List[str],
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "group": ids_te,
            "y_target": y_te_target,
            "y_raw": y_te_raw,
        }
    )
    for k, v in preds_in_target_units.items():
        pred_df[f"yhat_{k}"] = v

    aux_cols = [c for c in aux_candidates if c in df_te.columns]
    if aux_cols:
        aux = df_te.loc[mask_te, aux_cols].reset_index(drop=True)
        assert len(aux) == len(pred_df), "Aux/test row alignment mismatch."
        pred_df = pd.concat([pred_df.reset_index(drop=True), aux], axis=1)
    return pred_df


# =============================================================================
# Driver: one variant (UPDATED: monthly expanding-window evaluation)
# =============================================================================
def run_variant(cfg: Dict[str, Any], variant_label: str, feat_cols: List[str]) -> None:
    data_path = cfg["paths"]["data"]
    base_out_dir = ensure_dir(cfg["paths"]["out_dir"])
    LOG.info("=== Variant: %s ===", variant_label)
    LOG.info("Output dir: %s", base_out_dir)

    # Folder layout you want:
    # results/dh_ret/all-all/IBMCTINTERACTIONS/                <- aggregate outputs live here
    # results/dh_ret/all-all/IBMCTINTERACTIONS/monthly/        <- per-step outputs live here
    monthly_root = ensure_dir(os.path.join(base_out_dir, "monthly"))

    fh_variant = add_file_logger(
        os.path.join(base_out_dir, "run.log"),
        level=cfg.get("logging", {}).get("level", "INFO"),
    )
    summary_root: Dict[str, Any] = {"out_dir": base_out_dir, "monthly_dir": monthly_root}

    try:
        date_col = cfg["data"]["date_col"]
        group_col = cfg["data"].get("group_col", date_col)
        target_col = cfg["data"]["target"]
        raw_return_col = cfg["data"].get("raw_return_col", "dh_ret")
        tcol = cfg["data"].get("option_type_col", "option_type")
        mcol = cfg["data"].get("moneyness_col", "moneyness")
        atm_band = cfg.get("contracts", {}).get("atm_band", [0.98, 1.02])

        split_cfg = (cfg.get("split", {}) or {})
        exp_cfg = (split_cfg.get("expanding_eval", {}) or {})
        exp_enabled = bool(exp_cfg.get("enabled", True))
        exp_freq = str(exp_cfg.get("freq", "M")).upper()
        min_train_months = int(exp_cfg.get("min_train_months", 12))

        purge_gap = int(split_cfg.get("purge_gap", 1))
        tr_frac = float(split_cfg.get("train_frac", 0.70))
        va_frac = float(split_cfg.get("val_frac", 0.15))

        # Target kind (only used for extra columns; metrics use y_target anyway)
        tname = str(target_col).lower()
        if "vega" in tname:
            target_kind = "per_vega"
        elif "gamma" in tname:
            target_kind = "per_gamma"
        else:
            target_kind = "per_dollar"

        # Load and filter
        df_raw = load_data(data_path)
        df_raw = add_moneyness_std(df_raw)
        LOG.info("Loaded data: %d rows, %d cols", *df_raw.shape)

        df = apply_contract_filters(df_raw, cfg)
        LOG.info("After contract filters: %d rows", len(df))

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            n_inf = int(np.isinf(df[num_cols].to_numpy()).sum())
            if n_inf:
                LOG.warning("Replacing %d +/-inf values with NaN before splits.", n_inf)
                df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

        summary_root.update(rows_loaded=int(len(df_raw)), rows_after_filters=int(len(df)))

        shift_cfg = (cfg.get("data", {}).get("shift_features", {}) or {})
        if bool(shift_cfg.get("enabled", False)):
            base_feats_for_shift = select_features_by_groups(cfg, groups=[])
            df = shift_features_by_one_day(
                df,
                date_col=date_col,
                feature_cols=base_feats_for_shift,
                only_date_constant=bool(shift_cfg.get("only_date_constant", True)),
                force_all=bool(shift_cfg.get("force_all", False)),
            )

        # Feature routing by family
        inter_cols = set((cfg.get("data", {}).get("feature_groups") or {}).get("INTERACTIONS", []))
        feat_cols_lin = list(feat_cols)
        feat_cols_non = [c for c in feat_cols if c not in inter_cols]
        if not feat_cols_non:
            raise ValueError("After dropping INTERACTIONS, no features left for tree models.")

        needed_feats = sorted(set(feat_cols_lin) | set(feat_cols_non))
        missing = [c for c in (needed_feats + [target_col, raw_return_col, date_col, group_col]) if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in data: {missing}")

        # Sort by time
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        df = df.sort_values(date_col).reset_index(drop=True)

        # Monthly periods in the filtered panel
        if exp_freq != "M":
            raise ValueError(f"Only monthly freq is implemented (freq='M'). Got: {exp_freq}")

        period = df[date_col].dt.to_period("M")
        months = np.array(sorted(period.unique()))
        if len(months) < (min_train_months + 2):
            raise ValueError(
                f"Not enough months for expanding evaluation: months={len(months)}, "
                f"need at least {min_train_months + 2}."
            )

        LOG.info(
            "[expanding] enabled=%s | freq=%s | min_train_months=%d | months_total=%d",
            exp_enabled, exp_freq, min_train_months, len(months)
        )

        # ---------------------------------------------------------------------
        # Step runner: trains on (train, val), tests on exactly one month
        # Writes EVERYTHING into: base_out_dir/monthly/<YYYY-MM>/
        # Returns prediction frame (for that month) + step summary + N-En shap inputs for final-step SHAP
        # ---------------------------------------------------------------------
        def _run_one_step(
            *,
            step_out_dir: str,
            df_tr: pd.DataFrame,
            df_va: pd.DataFrame,
            df_te: pd.DataFrame,
            step_label: str,
            step_train_end: str,
            step_test_month: str,
        ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
            ensure_dir(step_out_dir)
            fh_step = add_file_logger(
                os.path.join(step_out_dir, "run.log"),
                level=cfg.get("logging", {}).get("level", "INFO"),
            )

            try:
                LOG.info("=== Expanding step: %s ===", step_label)
                LOG.info("Step out_dir: %s", step_out_dir)
                LOG.info("train_end=%s | test_month=%s", step_train_end, step_test_month)
                LOG.info("Split sizes: train=%d | val=%d | test=%d", len(df_tr), len(df_va), len(df_te))

                # Design matrices
                X_tr_lin_raw = df_tr[feat_cols_lin].to_numpy(dtype=float)
                X_va_lin_raw = df_va[feat_cols_lin].to_numpy(dtype=float)
                X_te_lin_raw = df_te[feat_cols_lin].to_numpy(dtype=float)

                X_tr_non_raw = df_tr[feat_cols_non].to_numpy(dtype=float)
                X_va_non_raw = df_va[feat_cols_non].to_numpy(dtype=float)
                X_te_non_raw = df_te[feat_cols_non].to_numpy(dtype=float)

                y_tr_target = df_tr[target_col].to_numpy(dtype=float)
                y_va_target = df_va[target_col].to_numpy(dtype=float)
                y_te_target_all = df_te[target_col].to_numpy(dtype=float)

                y_te_raw_all = df_te[raw_return_col].to_numpy(dtype=float)
                ids_te_all = df_te[group_col].to_numpy()
                d_tr = pd.to_datetime(df_tr[date_col].to_numpy())
                d_va = pd.to_datetime(df_va[date_col].to_numpy())

                # Target normalisation + guardrails
                tn_cfg = (cfg.get("data", {}).get("target_norm") or {"kind": "none"})
                normaliser = TargetNormalizer(
                    kind=tn_cfg.get("kind", "none"),
                    abs_greek=_as_bool(tn_cfg.get("abs", True), True),
                    floor=_as_float(tn_cfg.get("floor", 1e-6), 1e-6),
                    column=tn_cfg.get("column", None),
                    force=_as_bool(tn_cfg.get("force", False), False),
                )
                y_tr_model, _ = normaliser.apply(y_tr_target, df_tr, target_col=target_col)
                y_va_model, _ = normaliser.apply(y_va_target, df_va, target_col=target_col)
                y_te_model_all, g_te_all = normaliser.apply(y_te_target_all, df_te, target_col=target_col)

                guard = LabelGuardrails(cfg)
                guard.fit(y_tr_model)
                z_tr = guard.forward(y_tr_model)
                z_va = guard.forward(y_va_model)
                z_te_all = guard.forward(y_te_model_all)

                # Preprocessing
                pp_cfg = cfg.get("preprocess", {}) or {}

                pp_linear = preprocess_fit(X_tr_lin_raw, pp_cfg.get("linear_nn", {}))
                pp_ffn = preprocess_fit(X_tr_lin_raw, {**pp_cfg.get("linear_nn", {}), "yeo_johnson": False, "standardize": False})
                pp_trees = preprocess_fit(X_tr_non_raw, pp_cfg.get("trees", {}))

                X_tr_lin = preprocess_apply(X_tr_lin_raw, pp_linear)
                X_va_lin = preprocess_apply(X_va_lin_raw, pp_linear)
                X_te_lin = preprocess_apply(X_te_lin_raw, pp_linear)

                X_tr_ffn = preprocess_apply(X_tr_lin_raw, pp_ffn)
                X_va_ffn = preprocess_apply(X_va_lin_raw, pp_ffn)
                X_te_ffn = preprocess_apply(X_te_lin_raw, pp_ffn)

                X_tr_non = preprocess_apply(X_tr_non_raw, pp_trees)
                X_va_non = preprocess_apply(X_va_non_raw, pp_trees)
                X_te_non = preprocess_apply(X_te_non_raw, pp_trees)

                feat_names_lin = feature_names_after_pp(pp_linear, feat_cols_lin)
                feat_names_ffn = feature_names_after_pp(pp_ffn, feat_cols_lin)
                feat_names_non = feature_names_after_pp(pp_trees, feat_cols_non)

                # Filter non-finite y; align
                X_tr_lin, z_tr, d_tr, _, mask_tr = _filter_finite_y(X_tr_lin, z_tr, d_tr, split_name="train")
                X_va_lin, z_va, d_va, _, mask_va = _filter_finite_y(X_va_lin, z_va, d_va, split_name="val")
                X_te_lin, z_te, ids_te, _, mask_te = _filter_finite_y(X_te_lin, z_te_all, ids_te_all, split_name="test")

                X_tr_ffn = X_tr_ffn[mask_tr]
                X_va_ffn = X_va_ffn[mask_va]
                X_te_ffn = X_te_ffn[mask_te]

                X_tr_non = X_tr_non[mask_tr]
                X_va_non = X_va_non[mask_va]
                X_te_non = X_te_non[mask_te]

                y_te_target = y_te_target_all[mask_te]
                y_te_raw = y_te_raw_all[mask_te]
                g_te = g_te_all[mask_te]
                y_te_target_winsor = winsor_aligned_target(y_te_target, g_te, guard)

                # Registry and model lists
                registry = registry_from_yaml(cfg)
                enable = cfg["models"]["enable"]
                linear_list = [m for m in enable.get("linear", []) if m in registry]
                nonlinear_list = [m for m in enable.get("nonlinear", []) if m in registry]
                trees_list = [m for m in nonlinear_list if registry[m].kind == "tree"]
                nn_list = [m for m in nonlinear_list if registry[m].kind == "nn"]

                # Target-space scoring support for CV
                _, g_tr_all = normaliser.apply(y_tr_target, df_tr, target_col=target_col)
                _, g_va_all = normaliser.apply(y_va_target, df_va, target_col=target_col)
                g_tr_f = g_tr_all[mask_tr]
                g_va_f = g_va_all[mask_va]
                y_tr_from_z = normaliser.invert(guard.inverse(z_tr), g_tr_f)
                y_va_from_z = normaliser.invert(guard.inverse(z_va), g_va_f)

                use_trva_for_cv = bool((cfg.get("tuning", {}) or {}).get("cv", {}).get("use_train_plus_val", True))
                if use_trva_for_cv:
                    y_cv_target = np.hstack([y_tr_from_z, y_va_from_z])
                    g_cv = np.hstack([g_tr_f, g_va_f])
                else:
                    y_cv_target = y_tr_from_z
                    g_cv = g_tr_f

                def _inverse_to_target(z_pred_val, va_idx):
                    return normaliser.invert(guard.inverse(z_pred_val), g_cv[va_idx])

                # CV / training
                tcfg = cfg.get("tuning", {})
                strategy = tcfg.get("strategy", "random")
                n_iter = int(tcfg.get("n_iter", 1))
                keep_top_k = int(tcfg.get("keep_top_k", 1))
                seed = int(tcfg.get("seed", 0))
                cv_cfg = (tcfg.get("cv") or {})

                cv_records: List[Dict[str, Any]] = []

                LOG.info("Training linear: %s", linear_list)
                models_lin = fit_family(
                    registry, X_tr_lin, z_tr, X_va_lin, z_va,
                    linear_list, strategy, n_iter, keep_top_k, seed,
                    cv_records, d_tr, d_va, cv_cfg,
                    inverse_to_target_fn=_inverse_to_target, y_cv_target=y_cv_target,
                )

                LOG.info("Training FFN: %s", nn_list)
                models_nn = fit_family(
                    registry, X_tr_ffn, z_tr, X_va_ffn, z_va,
                    nn_list, strategy, n_iter, keep_top_k, seed,
                    cv_records, d_tr, d_va, cv_cfg,
                    inverse_to_target_fn=_inverse_to_target, y_cv_target=y_cv_target,
                )

                LOG.info("Training trees: %s", trees_list)
                models_trees = fit_family(
                    registry, X_tr_non, z_tr, X_va_non, z_va,
                    trees_list, strategy, n_iter, keep_top_k, seed,
                    cv_records, d_tr, d_va, cv_cfg,
                    inverse_to_target_fn=_inverse_to_target, y_cv_target=y_cv_target,
                )

                # Optional linear family ensemble
                build_ens = bool(cfg.get("tuning", {}).get("build_family_ensemble", True))
                if build_ens and models_lin:
                    models_lin["L-En"] = build_equal_weight_ensemble(list(models_lin.keys()), models_lin)

                # Test predictions (ALL models)
                preds_in_target_units: Dict[str, np.ndarray] = {}

                for name, m in models_lin.items():
                    z_hat = m.predict(X_te_lin)
                    y_hat = TargetNormalizer.invert(guard.inverse(z_hat), g_te)
                    preds_in_target_units[name] = y_hat

                for name, m in models_nn.items():
                    z_hat = m.predict(X_te_ffn)
                    y_hat = TargetNormalizer.invert(guard.inverse(z_hat), g_te)
                    preds_in_target_units[name] = y_hat

                for name, m in models_trees.items():
                    z_hat = m.predict(X_te_non)
                    y_hat = TargetNormalizer.invert(guard.inverse(z_hat), g_te)
                    preds_in_target_units[name] = y_hat

                # Prediction-space nonlinear ensemble N-En (mean of available base algos)
                nonlin_base_pred = {"rf", "lgbm_gbdt", "lgbm_dart", "ffn"}
                nonlin_member_names_pred = [
                    nm for nm in preds_in_target_units.keys()
                    if nm.split("#", 1)[0] in nonlin_base_pred
                ]
                if nonlin_member_names_pred:
                    nl_stack = np.column_stack([preds_in_target_units[nm] for nm in nonlin_member_names_pred])
                    preds_in_target_units["N-En"] = nl_stack.mean(axis=1)

                # Exports (per step)
                export_models_table(models_lin, "linear", step_out_dir)
                export_models_table({**models_nn, **models_trees}, "nonlinear", step_out_dir)
                export_linear_coefficients(models_lin, feat_names_lin, step_out_dir)
                if cv_records:
                    pd.DataFrame(cv_records).to_csv(os.path.join(step_out_dir, "cv_results.csv"), index=False)

                # Build predictions frame
                aux_candidates = ["vega_1volpt", "mid_t", "opt_rel_spread_final", "abs_delta", tcol, mcol, "tau", "atm_iv"]
                pred_df = build_predictions_frame(
                    ids_te=ids_te,
                    y_te_target=y_te_target,
                    y_te_raw=y_te_raw,
                    preds_in_target_units=preds_in_target_units,
                    df_te=df_te,
                    mask_te=mask_te,
                    aux_candidates=aux_candidates,
                )
                pred_df["step_label"] = step_label
                pred_df["train_end_month"] = step_train_end
                pred_df["test_month"] = step_test_month

                # Optional extra columns (kept consistent with your earlier code)
                eps = 1e-10
                if target_kind == "per_dollar" and "mid_t" in df_te.columns:
                    mid_te = df_te.loc[mask_te, "mid_t"].to_numpy(dtype=float)
                    for nm, yhat in preds_in_target_units.items():
                        pred_df[f"yhat_usd_{nm}"] = yhat * mid_te
                        pred_df[f"yhat_per_dollar_{nm}"] = yhat

                pred_df.to_csv(os.path.join(step_out_dir, "predictions.csv"), index=False)

                # Per-step metrics ON ALL MODELS (like you asked)
                export_metrics(
                    y_te_target,
                    preds_in_target_units,
                    pd.to_datetime(ids_te).to_numpy(),
                    step_out_dir,
                    y_true_winsor_aligned=y_te_target_winsor,
                    filename="metrics.csv",
                )
                export_diag_all(y_te_target, y_te_raw, preds_in_target_units, step_out_dir, pd.to_datetime(ids_te).to_numpy())

                # Per-step SHAP: N-En only (if present)
                if ("N-En" in preds_in_target_units) and bool((cfg.get("explain", {}).get("shap", {}) or {}).get("enabled", True)):
                    # Build best_by_algo for choosing concrete base models inside export_shap_n_en
                    def _base_algo(nm: str) -> str:
                        return nm.split("#", 1)[0]

                    r2_test = {nm: r2_os(y_te_target, yhat) for nm, yhat in preds_in_target_units.items()}
                    best_by_algo: Dict[str, Tuple[float, str]] = {}
                    for nm, score in r2_test.items():
                        base = _base_algo(nm)
                        if (base not in best_by_algo) or (score > best_by_algo[base][0]):
                            best_by_algo[base] = (score, nm)

                    export_shap_n_en(
                        all_models={**models_trees, **models_nn},
                        best_by_algo=best_by_algo,
                        X_te_non=X_te_non,
                        X_te_ffn=X_te_ffn,
                        df_te=df_te,
                        mask_te=mask_te,
                        feat_names_non=feat_names_non,
                        feat_names_ffn=feat_names_ffn,
                        out_dir=ensure_dir(os.path.join(step_out_dir, "SHAP")),
                        cfg=cfg,
                    )
                else:
                    best_by_algo = {}

                step_summary = {
                    "step_label": step_label,
                    "train_end_month": step_train_end,
                    "test_month": step_test_month,
                    "n_train": int(len(df_tr)),
                    "n_val": int(len(df_va)),
                    "n_test": int(len(df_te)),
                    "n_test_used": int(len(pred_df)),
                    "models_scored": sorted(list(preds_in_target_units.keys())),
                }
                with open(os.path.join(step_out_dir, "run_summary.json"), "w") as f:
                    json.dump(step_summary, f, indent=2, ensure_ascii=False, default=_safe_jsonable)

                # Return extras needed for FINAL-step SHAP at root (optional)
                shap_payload = {
                    "all_models": {**models_trees, **models_nn},
                    "best_by_algo": best_by_algo,
                    "X_te_non": X_te_non,
                    "X_te_ffn": X_te_ffn,
                    "df_te": df_te,
                    "mask_te": mask_te,
                    "feat_names_non": feat_names_non,
                    "feat_names_ffn": feat_names_ffn,
                }
                return pred_df, step_summary, shap_payload

            finally:
                remove_handler(fh_step)

        # ---------------------------------------------------------------------
        # Expanding loop
        # ---------------------------------------------------------------------
        if not exp_enabled:
            raise ValueError("expanding_eval.enabled is false.")

        all_preds: List[pd.DataFrame] = []
        step_summaries: List[Dict[str, Any]] = []
        last_shap_payload: Optional[Dict[str, Any]] = None

        for i in range(min_train_months - 1, len(months) - 1):
            train_end = months[i]
            test_m = months[i + 1]

            train_mask = period <= train_end
            test_mask = period == test_m

            df_train_full = df.loc[train_mask].copy()
            df_test_month = df.loc[test_mask].copy()

            if df_test_month.empty:
                LOG.info("[expanding] skip test_month=%s (no observations)", str(test_m))
                continue

            # 70/15 split *inside* the current expanding training window
            df_tr, df_va, _ = time_split(
                df_train_full,
                date_col,
                tr=tr_frac,
                va=va_frac,
                purge_gap=purge_gap,
            )
            if df_va.empty:
                LOG.warning("[expanding] empty val after purge; retry purge_gap=0 for step %s", str(test_m))
                df_tr, df_va, _ = time_split(
                    df_train_full,
                    date_col,
                    tr=tr_frac,
                    va=va_frac,
                    purge_gap=0,
                )

            step_label = f"train_to_{str(train_end)}__test_{str(test_m)}"
            step_out_dir = ensure_dir(os.path.join(monthly_root, str(test_m)))  # <-- monthly/<YYYY-MM>/

            pred_df_step, step_sum, shap_payload = _run_one_step(
                step_out_dir=step_out_dir,
                df_tr=df_tr,
                df_va=df_va,
                df_te=df_test_month,
                step_label=step_label,
                step_train_end=str(train_end),
                step_test_month=str(test_m),
            )

            all_preds.append(pred_df_step)
            step_summaries.append(step_sum)
            last_shap_payload = shap_payload

        if not all_preds:
            raise ValueError("No expanding-window steps produced predictions. Check date coverage after filters.")

        # ---------------------------------------------------------------------
        # AGGREGATE OUTPUTS LIVE IN base_out_dir (NOT in aggregate/)
        # ---------------------------------------------------------------------
        pred_all = pd.concat(all_preds, axis=0, ignore_index=True)
        pred_all = pred_all.sort_values(["test_month", "group"], kind="stable").reset_index(drop=True)

        pred_all_path = os.path.join(base_out_dir, "predictions_expanding.csv")
        pred_all.to_csv(pred_all_path, index=False)
        LOG.info("Wrote expanding predictions: %s", pred_all_path)

        steps_path = os.path.join(base_out_dir, "steps_summary.json")
        with open(steps_path, "w") as f:
            json.dump(step_summaries, f, indent=2, ensure_ascii=False, default=_safe_jsonable)
        LOG.info("Wrote steps summary: %s", steps_path)
        # Aggregate metrics ON ALL MODELS
        y_all = pred_all["y_target"].to_numpy(dtype=float)
        ids_all = pd.to_datetime(pred_all["group"]).to_numpy()

        model_cols = [
            c for c in pred_all.columns
            if c.startswith("yhat_")
            and (not c.startswith("yhat_usd_"))
            and (not c.startswith("yhat_per_dollar_"))
        ]
        preds_all_models: Dict[str, np.ndarray] = {}
        for c in model_cols:
            nm = c.replace("yhat_", "", 1)
            preds_all_models[nm] = pred_all[c].to_numpy(dtype=float)

        export_metrics(
            y_all,
            preds_all_models,
            ids_all,
            base_out_dir,
            y_true_winsor_aligned=None,
            filename="metrics.csv",
        )
         # -------------------------
        # Metrics per day (wide) - DAILY (not expanding)
        # -------------------------
        def _r2_os_cs_one_day(y: np.ndarray, yhat: np.ndarray) -> float:
            y = np.asarray(y, dtype=float)
            yhat = np.asarray(yhat, dtype=float)
            good = np.isfinite(y) & np.isfinite(yhat)
            if good.sum() < 5:
                return np.nan
            y = y[good]; yhat = yhat[good]
            y_cs = y - y.mean()
            yhat_cs = yhat - yhat.mean()
            den = float(np.mean(y_cs ** 2))
            if not np.isfinite(den) or den <= 0:
                return np.nan
            num = float(np.mean((y_cs - yhat_cs) ** 2))
            return 1.0 - num / den

        pred_all_tmp = pred_all.copy()
        pred_all_tmp["_day"] = pd.to_datetime(pred_all_tmp["group"], errors="coerce").dt.normalize()
        pred_all_tmp = pred_all_tmp.dropna(subset=["_day"])

        days = np.array(sorted(pred_all_tmp["_day"].unique()))
        models = sorted(list(preds_all_models.keys()))

        r2_day = pd.DataFrame(index=pd.to_datetime(days), columns=models, remember=True, dtype=float) \
            if "remember" in pd.DataFrame.__init__.__code__.co_varnames else \
            pd.DataFrame(index=pd.to_datetime(days), columns=models, dtype=float)

        r2_day_xs = pd.DataFrame(index=pd.to_datetime(days), columns=models, dtype=float)

        for day, d in pred_all_tmp.groupby("_day", sort=True):
            y = d["y_target"].to_numpy(dtype=float)
            for nm in models:
                yhat = d[f"yhat_{nm}"].to_numpy(dtype=float)
                r2_day.loc[pd.Timestamp(day), nm] = r2_os(y, yhat)
                r2_day_xs.loc[pd.Timestamp(day), nm] = _r2_os_cs_one_day(y, yhat)

        r2_day.reset_index().rename(columns={"index": "date"}).to_csv(
            os.path.join(base_out_dir, "metrics_per_day.csv"), index=False
        )
        r2_day_xs.reset_index().rename(columns={"index": "date"}).to_csv(
            os.path.join(base_out_dir, "metrics_per_day_xs.csv"), index=False
        )


        # Aggregate portfolio metrics (tertiles as before) for ALL models
        pf_out = ensure_dir(os.path.join(base_out_dir, "portfolio"))
        export_portfolios_raw(
            y=pred_all["y_target"].to_numpy(),
            preds_scores={nm: pred_all[f"yhat_{nm}"].to_numpy() for nm in preds_all_models.keys()},
            ids=pd.to_datetime(pred_all["group"]).to_numpy(),
            out_dir=pf_out,
            n_bins=int((cfg.get("portfolio", {}) or {}).get("n_bins", 3)),
            normalize_group_to_day=True,
            write_ic=True,
        )

        # Final-step SHAP (N-En only) written into base_out_dir/SHAP_final
        shap_conf = (cfg.get("explain", {}).get("shap", {}) or {})
        if last_shap_payload is not None and bool(shap_conf.get("enabled", True)):
            if "best_by_algo" in last_shap_payload and last_shap_payload["best_by_algo"]:
                final_shap_dir = ensure_dir(os.path.join(base_out_dir, "SHAP"))
                export_shap_n_en(
                    out_dir=final_shap_dir,
                    cfg=cfg,
                    **last_shap_payload,
                )

        summary_root.update(
            {
                "expanding_eval": {
                    "enabled": True,
                    "freq": exp_freq,
                    "min_train_months": min_train_months,
                    "months_total": int(len(months)),
                    "steps_run": int(len(step_summaries)),
                    "predictions_path": pred_all_path,
                    "monthly_dir": monthly_root,
                }
            }
        )
        with open(os.path.join(base_out_dir, "run_summary.json"), "w") as f:
            json.dump(summary_root, f, indent=2, ensure_ascii=False, default=_safe_jsonable)

        LOG.info("Variant complete (expanding monthly). Yay! :)")

    finally:
        remove_handler(fh_variant)



# =============================================================================
# RunSpec -> config merge & driver (unchanged)
# =============================================================================


def _apply_contract_overrides(
    cfg_in: Dict[str, Any], side: Optional[str], mny: Optional[str], band: Optional[Tuple[float, float]]
):
    cfg2 = yaml.safe_load(yaml.safe_dump(cfg_in))
    cfg2.setdefault("contracts", {})
    if side:
        cfg2["contracts"]["type"] = side
    if mny:
        cfg2["contracts"]["moneyness"] = mny
    if band:
        cfg2["contracts"]["atm_band"] = list(band)
    return cfg2


def _features_after_exclusions(cfg: Dict[str, Any], base_feats: List[str]) -> Tuple[List[str], List[str]]:
    target = cfg.get("data", {}).get("target", "")
    excl_map = (cfg.get("data", {}).get("exclusions_by_target") or {})
    excluded = list(excl_map.get(target, []))
    if not excluded:
        return base_feats, []
    excl_set = set(excluded)
    kept = [f for f in base_feats if f not in excl_set]
    missing_in_base = sorted(excl_set - set(base_feats))
    if missing_in_base:
        LOG.info("[features] exclusions_by_target referenced non-present features: %s", ", ".join(missing_in_base))
    LOG.info("[features] auto-excluded for target '%s': %s", target, ", ".join(excluded))
    return kept, excluded


def run_from_spec(spec: RunSpec) -> None:
    cfg_base = load_cfg(spec.base_config_path)
    setup_logging(cfg_base.get("logging", {}).get("level", "INFO"))

    if spec.target:
        cfg_base.setdefault("data", {})["target"] = spec.target
    if spec.data_path:
        cfg_base.setdefault("paths", {})["data"] = spec.data_path

    if spec.use_base_yaml_only:
        base_feats = cfg_base["data"].get("features") or select_features_by_groups(cfg_base, [])
        feats_used, excluded_auto = _features_after_exclusions(cfg_base, base_feats)
        outdir = ensure_dir(spec.out_dir) if spec.out_dir else ensure_dir(derive_out_dir(cfg_base))
        cfg_base.setdefault("paths", {})["out_dir"] = outdir
        cfg_base["_resolved_features"] = feats_used
        cfg_base["_resolved_exclusions"] = excluded_auto
        cfg_base.setdefault("_run_overrides", {})["label"] = spec.label
        with open(os.path.join(outdir, "config_resolved.yaml"), "w") as f:
            yaml.safe_dump(cfg_base, f, sort_keys=False)
        run_variant(cfg_base, "", feats_used)
        return

    if spec.groups is None:
        base_feats = cfg_base["data"].get("features") or select_features_by_groups(cfg_base, [])
        groups_tag = None
    else:
        if len(spec.groups) == 1 and spec.groups[0].upper() == "ALL":
            base_feats = _union_of_all_feature_groups(cfg_base)
            groups_tag = "ALL"
        else:
            base_feats = select_features_by_groups(cfg_base, spec.groups)
            groups_tag = "".join(spec.groups)

    feats_after_auto, excluded_auto = _features_after_exclusions(cfg_base, base_feats)

    if spec.exclude_features:
        before = len(feats_after_auto)
        excl_set = set(spec.exclude_features)
        feats_after_auto = [f for f in feats_after_auto if f not in excl_set]
        LOG.info("Excluded %d features by RunSpec; %d remaining.", before - len(feats_after_auto), len(feats_after_auto))

    cfg2 = _apply_contract_overrides(cfg_base, spec.contracts_type, spec.contracts_mny, spec.atm_band)
    outdir = ensure_dir(os.path.join(spec.out_dir, groups_tag)) if spec.out_dir else ensure_dir(
        derive_out_dir(cfg2, groups_tag=groups_tag)
    )
    cfg2.setdefault("paths", {})["out_dir"] = outdir
    cfg2.setdefault("_run_overrides", {}).update(
        {
            "label": spec.label,
            "groups": spec.groups,
            "side": spec.contracts_type,
            "moneyness": spec.contracts_mny,
            "atm_band": spec.atm_band,
            "out_dir": outdir,
        }
    )
    cfg2["_resolved_features"] = feats_after_auto
    cfg2["_resolved_exclusions"] = excluded_auto
    with open(os.path.join(outdir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg2, f, sort_keys=False)
    run_variant(cfg2, "", feats_after_auto)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if RUNS:
        for i, spec in enumerate(RUNS, 1):
            LOG.info("\n===== RUN %d/%d: %s =====", i, len(RUNS), spec.label or "(no label)")
            run_from_spec(spec)
    else:
        LOG.warning("No RUNS configured. Edit RUNS at the top of the file.")
