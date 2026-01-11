from __future__ import annotations

from pathlib import Path
import os
import uuid
import numpy as np
import rasterio


IN_PATH = Path(r"data_raw\pinta_n03_07\geotiff\SSS\BSH_N-03-07.tif")
OUT_PATH = Path(r"data_raw\pinta_n03_07\geotiff\SSS\BSH_N-03-07_nodata251.tif")
FORCE_NODATA = 251


def guess_background_value(arr: np.ndarray, max_sample: int = 2_000_000) -> int:
    flat = arr.ravel()
    if flat.size > max_sample:
        idx = np.linspace(0, flat.size - 1, max_sample, dtype=np.int64)
        flat = flat[idx]
    if flat.dtype == np.uint8:
        counts = np.bincount(flat, minlength=256)
        return int(np.argmax(counts))
    vals, cnt = np.unique(flat, return_counts=True)
    return int(vals[int(np.argmax(cnt))])


def main() -> int:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH}")

    with rasterio.open(IN_PATH) as src:
        arr = src.read(1)
        profile = src.profile.copy()

    nodata = FORCE_NODATA if FORCE_NODATA is not None else guess_background_value(arr)

    # Always write as GTiff
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["nodata"] = nodata
    profile.update(
        compress="DEFLATE",
        predictor=1 if arr.dtype == np.uint8 else 2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_SAFER",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp first, then atomic replace
    tmp_path = OUT_PATH.with_suffix(OUT_PATH.suffix + f".tmp_{uuid.uuid4().hex}")

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(arr, 1)

    # If OUT exists and is read-only, remove read-only flag
    if OUT_PATH.exists():
        try:
            os.chmod(OUT_PATH, 0o666)
        except Exception:
            pass

    os.replace(tmp_path, OUT_PATH)

    with rasterio.open(OUT_PATH) as s:
        print(f"Wrote: {OUT_PATH}")
        print("driver:", s.driver, "nodata:", s.nodata, "dtype:", s.dtypes[0], "crs:", s.crs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
