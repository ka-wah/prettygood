"""
Simple DH return investigation: percentiles and bucket counts for the dh_ret column.

Usage:
    python figs/dh_investigation.py [path_to_csv_with_dh_ret]

If no path is given, defaults to outputs/features_and_dh_returns.parquet from repo root.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    default_path = "outputs/features_and_dh_returns.parquet"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "dh_ret" not in df.columns:
        raise ValueError(f"'dh_ret' column not found in {path}. Available: {list(df.columns)}")

    dh = pd.to_numeric(df["dh_ret"], errors="coerce").dropna()
    if dh.empty:
        raise ValueError("No valid numeric values in 'dh_ret'.")

    qs = [0.01, 0.05, 0.50, 0.90, 0.95, 0.99, 0.995]
    quantiles = dh.quantile(qs)
    print("dh_ret quantiles:")
    print(quantiles.to_frame("value"))

    # Bucket counts between consecutive quantile cutoffs (including tails)
    cuts = [dh.min()] + list(quantiles) + [dh.max()]
    labels = []
    counts = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        labels.append(f"({lo:.4g}, {hi:.4g}]")
        counts.append(((dh > lo) & (dh <= hi)).sum())

    bucket_df = pd.DataFrame({"range": labels, "count": counts})
    print("\nCounts per percentile bucket:")
    print(bucket_df)

    # Observations above a threshold
    thr = 10
    n_above = (dh > thr).sum()
    print(f"\nObservations with dh_ret > {thr}: {n_above} / {len(dh)}")

    # Plot distribution (histogram) and save beside the input file
    plt.figure(figsize=(8, 4))
    plt.hist(dh, bins=50, color="#4c72b0", edgecolor="black", alpha=0.75)
    plt.xlabel("dh_ret")
    plt.ylabel("Frequency")
    plt.title("Distribution of dh_ret")
    plt.tight_layout()

    out_path = path.rsplit(".", 1)[0] + "_dh_ret_hist.png"
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved histogram to {out_path}")
    # plt.show()


if __name__ == "__main__":
    main()
