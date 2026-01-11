from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw" / "pinta_n03_07" / "geotiff"
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def guess_sensor(rel_path: str) -> str:
    p = rel_path.lower()
    if p.startswith("bathymetry/"):
        return "multibeam_bathymetry"
    if p.startswith("sss/"):
        return "side_scan_sonar"
    if p.startswith("picked_horizons_tops/"):
        return "sub_bottom_horizon"
    if p.startswith("sediment_thickness/"):
        return "sub_bottom_thickness"
    return "unknown"

def guess_product(rel_path: str) -> str:
    name = Path(rel_path).name.lower()
    if "pos_num" in name or "bathymetry" in rel_path.lower():
        return "bathymetry_grid"
    if rel_path.lower().startswith("sss/"):
        return "sss_mosaic"
    if rel_path.lower().startswith("picked_horizons_tops/"):
        return "horizon_depth"
    if rel_path.lower().startswith("sediment_thickness/"):
        return "sediment_thickness"
    return "raster"

def read_raster_meta(fp: Path) -> dict:
    with rasterio.Env():
        with rasterio.open(fp) as ds:
            b = ds.bounds
            return {
                "width": ds.width,
                "height": ds.height,
                "count": ds.count,
                "dtype": str(ds.dtypes[0]) if ds.count else "",
                "nodata": ds.nodata,
                "crs": ds.crs.to_string() if ds.crs else None,
                "transform_a": ds.transform.a,
                "transform_e": ds.transform.e,
                "pixel_size_x": abs(ds.transform.a),
                "pixel_size_y": abs(ds.transform.e),
                "bounds_left": b.left,
                "bounds_bottom": b.bottom,
                "bounds_right": b.right,
                "bounds_top": b.top,
            }

def main() -> int:
    rasters = sorted([*DATA.rglob("*.tif"), *DATA.rglob("*.tiff")])
    if not rasters:
        raise SystemExit(f"No rasters found in {DATA}")

    rows = []
    for fp in rasters:
        rel = fp.relative_to(DATA).as_posix()
        sensor = guess_sensor(rel)
        product = guess_product(rel)

        meta = {}
        try:
            meta = read_raster_meta(fp)
            status = "ok"
        except Exception as e:
            status = f"read_error: {type(e).__name__}: {e}"

        rows.append({
            "file": str(fp),
            "rel_path": rel,
            "sensor": sensor,
            "product": product,
            "status": status,
            **meta
        })

    df = pd.DataFrame(rows).sort_values(["sensor", "product", "rel_path"])
    out_csv = OUT / "inputs_index.csv"
    df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(df[["sensor","product","status","rel_path","crs","pixel_size_x","width","height"]].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
