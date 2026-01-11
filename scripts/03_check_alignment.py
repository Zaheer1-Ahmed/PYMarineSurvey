# scripts/03_check_alignment.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

DATA_ROOT = Path("data_raw/pinta_n03_07/geotiff")
INDEX_CSV = Path("outputs/inputs_index.csv")
OUT_CSV = Path("outputs/alignment_check.csv")

TARGET_CRS = "EPSG:32632"  # use UTM32N as a common frame for comparing bounds (SSS is already 32632)

def bounds_in_crs(src: rasterio.io.DatasetReader, target_crs: str) -> tuple[float, float, float, float]:
    b = src.bounds
    if src.crs is None:
        return (float("nan"),) * 4
    if str(src.crs).upper() == target_crs.upper():
        return (b.left, b.bottom, b.right, b.top)
    return transform_bounds(src.crs, target_crs, b.left, b.bottom, b.right, b.top, densify_pts=21)

def main() -> int:
    df = pd.read_csv(INDEX_CSV)
    rows = []

    for _, r in df.iterrows():
        rel = str(r.get("rel_path", ""))
        if not rel.lower().endswith((".tif", ".tiff")):
            continue

        fp = DATA_ROOT / rel
        if not fp.exists():
            rows.append({
                "rel_path": rel,
                "status": "missing",
            })
            continue

        with rasterio.open(fp) as src:
            bx = bounds_in_crs(src, TARGET_CRS)
            rows.append({
                "sensor": r.get("sensor", ""),
                "product": r.get("product", ""),
                "rel_path": rel,
                "src_crs": str(src.crs),
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "pixel_x": src.transform.a,
                "pixel_y": src.transform.e,
                "bounds_target_minx": bx[0],
                "bounds_target_miny": bx[1],
                "bounds_target_maxx": bx[2],
                "bounds_target_maxy": bx[3],
                "status": "ok",
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")

    # quick human summary
    print("\nCRS COUNTS:")
    if "src_crs" in out.columns:
        print(out["src_crs"].value_counts(dropna=False).to_string())

    print("\nBOUNDS OVERVIEW (in", TARGET_CRS, "):")
    cols = ["sensor","product","rel_path","bounds_target_minx","bounds_target_miny","bounds_target_maxx","bounds_target_maxy"]
    print(out[cols].to_string(index=False))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
