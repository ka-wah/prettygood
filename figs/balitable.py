import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, skew, kurtosis
from pathlib import Path

# ======================
# Press-play configuration
# ======================
DATA_PATH = "inputs/short.parquet"   # <-- change to your full dataset path
OUT_DIR = Path("figs/output/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Training/OOS cut (inclusive end of Feb 2023, in UTC after parsing)
TRAIN_END = "2023-02-28"

# Quantiles to show (Bali uses 10/25/50/75/90).
# If you want "better tails", use (0.05,0.25,0.50,0.75,0.95).
P_LOW, P_Q1, P_Q2, P_Q3, P_HIGH = (0.05, 0.25, 0.50, 0.75, 0.95)

# Variables/labels to match the Bali-style rows (based on your column names)
# Delta-Hedged Return: dh_ret
# Days to Maturity: dte
# Moneyness: exp(log_moneyness)
# Implied volatility: iv
# Absolute Delta: abs_delta
VARS = [
    ("dh_ret", "Delta-Hedged Return"),
    ("dte", "Days to Maturity"),
    ("moneyness", "Moneyness"),
    ("iv", "Implied volatility"),
    ("abs_delta", "Absolute Delta"),
]

# Rounding similar to common summary tables:
# - DTE quantiles shown as 1 decimal in Bali examples (often integers); we do 1.
# - Others: 2 decimals
ROW_DECIMALS = {
    "Delta-Hedged Return": 2,
    "Days to Maturity": 1,
    "Moneyness": 2,
    "Implied volatility": 2,
    "Absolute Delta": 2,
}

# ======================
# Load + basic cleaning
# ======================
if str(DATA_PATH).lower().endswith(".parquet"):
    df = pd.read_parquet(DATA_PATH)
elif str(DATA_PATH).lower().endswith(".csv"):
    df = pd.read_csv(DATA_PATH)
else:
    raise ValueError("Unsupported file type. Use .parquet or .csv")

# Parse date strings like "Tue Dec 07 2021 18:00:00 GMT-0600 (Central Standard Time)"
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df = df.loc[df["date"].notna()].copy()

# Numeric conversions (avoid strings silently breaking stats)
for c in ["dh_ret", "dte", "log_moneyness", "iv", "abs_delta", "is_call"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Derived: moneyness level from log_moneyness
df["moneyness"] = np.exp(df["log_moneyness"]) if "log_moneyness" in df.columns else np.nan

# Split train/oos
train_end_ts = pd.to_datetime(TRAIN_END, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
df_train = df.loc[df["date"] <= train_end_ts].copy()
df_oos = df.loc[df["date"] > train_end_ts].copy()

# --- scale IV for reporting (keep raw df["iv"] unchanged for other work) ---
for _d in (df, df_train, df_oos):
    _d["iv"] = 100.0 * _d["iv"]


# ======================
# Summary-stat helpers
# ======================
COLS = ["Mean", "Sd", "10-Pctl", "Q1", "Q2", "Q3", "90-Pctl", "Skew", "Kurt", "JB"]

def _clean_array(x: pd.Series) -> np.ndarray:
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return a

def _jb_pvalue_percent(a: np.ndarray) -> float:
    if a.size < 3:
        return np.nan
    _, p = jarque_bera(a)
    return float(100.0 * p)

def describe_series(x: pd.Series) -> dict:
    a = _clean_array(x)
    if a.size == 0:
        return {k: np.nan for k in COLS}

    # Kurt is excess kurtosis (normal = 0), which matches standard finance convention
    return {
        "Mean": float(np.mean(a)),
        "Sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "10-Pctl": float(np.quantile(a, P_LOW)),
        "Q1": float(np.quantile(a, P_Q1)),
        "Q2": float(np.quantile(a, P_Q2)),
        "Q3": float(np.quantile(a, P_Q3)),
        "90-Pctl": float(np.quantile(a, P_HIGH)),
        "Skew": float(skew(a, bias=False)) if a.size > 2 else np.nan,
        "Kurt": float(kurtosis(a, fisher=True, bias=False)) if a.size > 3 else np.nan,
        "JB": _jb_pvalue_percent(a),  # p-value in percent; prints ~0.0 when strongly rejected
    }

def panel_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in VARS:
        if col not in d.columns:
            s = pd.Series(dtype=float)
        else:
            s = d[col]
        rows.append({"Variable": label, **describe_series(s)})
    tab = pd.DataFrame(rows).set_index("Variable")
    return tab[COLS]

def format_table(tab: pd.DataFrame) -> pd.DataFrame:
    out = tab.copy()
    for r in out.index:
        dec = ROW_DECIMALS.get(r, 2)
        out.loc[r] = out.loc[r].map(lambda z: "" if pd.isna(z) else f"{float(z):.{dec}f}")
    return out

def build_latex(panels, caption, label) -> str:
    # panels: list of (panel_title, N, formatted_df_strings)
    colspec = "l" + "r"*10
    header = [
        "\\begin{table}[!ht]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\hline",
        " & Mean & Sd & 10-Pctl & Q1 & Q2 & Q3 & 90-Pctl & Skew & Kurt & JB \\\\",
        "\\hline",
    ]
    body = []
    for title, nobs, df_str in panels:
        body.append(f"\\multicolumn{{11}}{{l}}{{\\textit{{{title} (N={nobs:,})}}}} \\\\")
        for var, row in df_str.iterrows():
            body.append(f"{var} & " + " & ".join(row.values.tolist()) + " \\\\")
        body.append("\\hline")
    footer = [
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(header + body + footer)

def write_table(tag: str, panels, caption: str):
    # numeric CSV
    numeric = pd.concat({t: p for (t, _, p) in panels}, names=["Panel", "Variable"])
    csv_path = OUT_DIR / f"sumstats_{tag}.csv"
    numeric.to_csv(csv_path)

    # latex TEX
    panels_fmt = [(t, n, format_table(p)) for (t, n, p) in panels]
    tex = build_latex(panels_fmt, caption=caption, label=f"tab:sumstats_{tag}")
    tex_path = OUT_DIR / f"sumstats_{tag}.tex"
    tex_path.write_text(tex, encoding="utf-8")

    print(f"[OK] {csv_path}")
    print(f"[OK] {tex_path}")

# ======================
# Panels (Bali-style A–E)
# ======================
def make_call_put_panels(d: pd.DataFrame):
    d_all = d
    d_call = d.loc[d["is_call"] == 1].copy()
    d_put  = d.loc[d["is_call"] == 0].copy()
    return d_all, d_call, d_put

# Full-sample A/B/C
d_all, d_call, d_put = make_call_put_panels(df)

panels_full = [
    ("Panel A: All Options", len(d_all), panel_table(d_all)),
    ("Panel B: Call Options", len(d_call), panel_table(d_call)),
    ("Panel C: Put Options", len(d_put), panel_table(d_put)),
]

write_table(
    tag="full",
    panels=panels_full,
    caption="Summary statistics."
)

# Optional: add Train/OOS versions (you asked for these earlier)
d_all_tr, d_call_tr, d_put_tr = make_call_put_panels(df_train)
panels_train = [
    ("Panel A: All Options", len(d_all_tr), panel_table(d_all_tr)),
    ("Panel B: Call Options", len(d_call_tr), panel_table(d_call_tr)),
    ("Panel C: Put Options", len(d_put_tr), panel_table(d_put_tr)),
]
write_table(
    tag="train_through_2023_02",
    panels=panels_train,
    caption="Summary statistics (training sample through February 2023)."
)

d_all_o, d_call_o, d_put_o = make_call_put_panels(df_oos)
panels_oos = [
    ("Panel A: All Options", len(d_all_o), panel_table(d_all_o)),
    ("Panel B: Call Options", len(d_call_o), panel_table(d_call_o)),
    ("Panel C: Put Options", len(d_put_o), panel_table(d_put_o)),
]
write_table(
    tag="oos_post_2023_02",
    panels=panels_oos,
    caption="Summary statistics (out-of-sample period after February 2023)."
)

print(f"Train N = {len(df_train):,} | OOS N = {len(df_oos):,}")

# ======================
# If you want Bali-like D/E subperiods inside a sample:
# define date cutoffs and add panels like:
#
# early = df[(df["date"] >= 'YYYY-01-01') & (df["date"] <= 'YYYY-12-31')]
# late  = df[(df["date"] >= 'YYYY-01-01') & (df["date"] <= 'YYYY-12-31')]
# panels_full += [("Panel D: All Options YYYY-YYYY", len(early), panel_table(early)),
#                 ("Panel E: All Options YYYY-YYYY", len(late),  panel_table(late))]
# and then write_table(...) again.
# ======================
