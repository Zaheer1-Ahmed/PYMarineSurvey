# scripts/04_value_sanity.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

DATA_ROOT = Path("data_raw/pinta_n03_07/geotiff")
INDEX_CSV = Path("outputs/inputs_index.csv")
OUT_CSV = Path("outputs/value_sanity.csv")

def robust_stats(arr: np.ndarray) -> dict:
    a = arr.astype("float64", copy=False)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": np.nan, "p01": np.nan, "p50": np.nan, "p99": np.nan, "max": np.nan, "mean": np.nan}
    return {
        "min": float(np.min(a)),
        "p01": float(np.quantile(a, 0.01)),
        "p50": float(np.quantile(a, 0.50)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }

def main() -> int:
    df = pd.read_csv(INDEX_CSV)
    rows = []

    for _, r in df.iterrows():
        rel = str(r.get("rel_path", ""))
        if not rel.lower().endswith((".tif", ".tiff")):
            continue

        fp = DATA_ROOT / rel
        if not fp.exists():
            continue

        with rasterio.open(fp) as src:
            # read a decimated sample for speed
            h, w = src.height, src.width
            step_r = max(1, h // 1500)
            step_c = max(1, w // 1500)

            a = src.read(1, masked=False)[::step_r, ::step_c]

            nod = src.nodata
            # treat nodata as invalid (important for your huge negative nodata values)
            if nod is not None:
                a = a.astype("float64", copy=False)
                a[a == float(nod)] = np.nan

            st = robust_stats(a)

            # basic “does it look like depth/intensity/thickness” hints
            hint = []
            if str(r.get("product","")) == "sss_mosaic":
                # SSS typically 0..255
                if st["p01"] < 0 or st["p99"] > 255:
                    hint.append("unexpected_range_for_uint8_intensity")
                else:
                    hint.append("looks_like_uint8_intensity")
            if "bathymetry" in str(r.get("sensor","")):
                # bathymetry often negative depths (below sea level), but “positive_values” file is explicitly positive
                hint.append("file_name_suggests_positive_depths")
            if "horizon" in str(r.get("product","")):
                hint.append("sparse_surface_expected")
            if "thickness" in str(r.get("product","")):
                hint.append("thickness_expected_nonnegative")

            rows.append({
                "sensor": r.get("sensor",""),
                "product": r.get("product",""),
                "rel_path": rel,
                "dtype": src.dtypes[0],
                "nodata": nod,
                **st,
                "hint": "|".join(hint),
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")

    # concise view
    cols = ["sensor","product","rel_path","dtype","nodata","p01","p50","p99","min","max","hint"]
    print("\nSANITY TABLE:")
    print(out[cols].to_string(index=False))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
