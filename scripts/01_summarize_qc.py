import pandas as pd

df = pd.read_csv("outputs/qc/qc_metrics.csv")

print("\nQC SCORE COUNTS:")
print(df["qc_score"].value_counts(dropna=False))

cols = ["sensor","product","qc_score","nodata_ratio","positive_fraction","tail_fraction","p01","p50","p99","crs","rel_path"]
for c in cols:
    if c not in df.columns:
        df[c] = None

print("\nTOP SUSPICIOUS (RED first, then high nodata/tails):")
order = {"RED":0, "YELLOW":1, "GREEN":2}
df["_score_order"] = df["qc_score"].map(order).fillna(9)
df2 = df.sort_values(["_score_order","nodata_ratio","tail_fraction"], ascending=[True, False, False])
print(df2[cols].head(20).to_string(index=False))
