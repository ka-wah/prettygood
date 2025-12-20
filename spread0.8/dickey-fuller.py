import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class ADFResult:
    test_stat: float
    p_value: float
    used_lag: int
    nobs: int
    critical_values: dict
    icbest: Optional[float]


def _infer_date_col(df: pd.DataFrame) -> str:
    # 1) Prefer datetime dtype columns
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if len(dt_cols) == 1:
        return dt_cols[0]

    # 2) Common names
    candidates = ["date", "datetime", "timestamp", "time", "dt"]
    for c in candidates:
        if c in df.columns:
            return c

    # 3) Try parsing object columns
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    best_col, best_ok = None, -1
    for c in obj_cols:
        parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
        ok = parsed.notna().sum()
        if ok > best_ok:
            best_col, best_ok = c, ok
    if best_col is not None and best_ok > 0:
        return best_col

    raise ValueError("Could not infer a date column. Pass date_col explicitly.")


def _infer_return_col(df: pd.DataFrame) -> str:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found. Pass return_col explicitly.")

    # Prefer columns that look like delta-hedged returns
    keywords = ["dh", "delta", "hedg", "ret", "return", "pnl"]
    scored = []
    for c in numeric_cols:
        name = c.lower()
        score = sum(k in name for k in keywords)
        scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    best_score, best_col = scored[0]
    if best_score == 0 and len(numeric_cols) > 1:
        # Fall back: try common exact names
        for c in ["dh_ret", "delta_hedged_return", "delta_hedged_ret", "ret", "return"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                return c
    return best_col


def _prepare_daily_series(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    agg: str = "mean",
) -> pd.Series:
    x = df[[date_col, return_col]].copy()

    x[date_col] = pd.to_datetime(x[date_col], errors="coerce", utc=True)
    x = x.dropna(subset=[date_col, return_col])

    # Ensure numeric + remove inf
    x[return_col] = pd.to_numeric(x[return_col], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=[return_col])

    # If multiple rows per day, aggregate to a single daily series
    x["__day__"] = x[date_col].dt.floor("D")
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'.")
    if agg == "mean":
        s = x.groupby("__day__")[return_col].mean()
    else:
        s = x.groupby("__day__")[return_col].median()

    s = s.sort_index()
    s.name = f"{return_col}_daily_{agg}"
    return s


def run_adf(
    series: pd.Series,
    regression: str = "c",
    autolag: str = "AIC",
    maxlag: Optional[int] = None,
) -> ADFResult:
    s = series.dropna().astype(float)
    if s.size < 20:
        raise ValueError(f"Series too short for ADF (n={s.size}).")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = adfuller(s.values, regression=regression, autolag=autolag, maxlag=maxlag)

    test_stat, p_value, used_lag, nobs, crit_vals, icbest = out
    return ADFResult(
        test_stat=float(test_stat),
        p_value=float(p_value),
        used_lag=int(used_lag),
        nobs=int(nobs),
        critical_values=dict(crit_vals),
        icbest=None if icbest is None else float(icbest),
    )


def print_adf(res: ADFResult, name: str, alpha: float = 0.05) -> None:
    print(f"\nADF test: {name}")
    print(f"  test statistic : {res.test_stat: .6f}")
    print(f"  p-value        : {res.p_value: .6g}")
    print(f"  used lags      : {res.used_lag}")
    print(f"  n obs          : {res.nobs}")
    if res.icbest is not None:
        print(f"  icbest         : {res.icbest: .6f}")
    print("  critical values:")
    for k, v in res.critical_values.items():
        print(f"    {k:>4}: {v: .6f}")

    verdict = "reject unit root (stationary)" if res.p_value < alpha else "fail to reject unit root"
    print(f"  decision @ {alpha:.0%}: {verdict}")


def adf_on_delta_hedged_returns(
    parquet_path: str = "dhinput.parquet",
    date_col: Optional[str] = None,
    return_col: Optional[str] = None,
    agg: str = "mean",
    regression: str = "c",
    autolag: str = "AIC",
    maxlag: Optional[int] = None,
) -> Tuple[pd.Series, ADFResult]:
    """
    Loads a (possibly panel) dataset of delta-hedged returns, aggregates to a daily series,
    runs the Augmented Dickey-Fuller test, and prints the statistics.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to read parquet. Install a parquet engine (recommended: pyarrow).\n"
            "In Python: pip install pyarrow"
        ) from e

    if date_col is None:
        date_col = _infer_date_col(df)
    if return_col is None:
        return_col = _infer_return_col(df)

    daily = _prepare_daily_series(df, date_col=date_col, return_col=return_col, agg=agg)
    res = run_adf(daily, regression=regression, autolag=autolag, maxlag=maxlag)

    print(f"Loaded: {parquet_path}")
    print(f"Using date_col='{date_col}', return_col='{return_col}', agg='{agg}'")
    print(f"Date range: {daily.index.min().date()} to {daily.index.max().date()} | n_days={daily.shape[0]}")
    print_adf(res, name=daily.name)

    return daily, res


if __name__ == "__main__":
    # Press play: adjust parquet_path if needed.
    adf_on_delta_hedged_returns(parquet_path="spread0.8/dhinput.parquet", agg="mean", regression="c", autolag="AIC")
