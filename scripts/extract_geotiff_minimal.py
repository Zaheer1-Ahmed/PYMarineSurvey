from __future__ import annotations
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data_raw" / "pinta_n03_07" / "N-03-07_GeoTIFF.zip"
OUT_DIR  = ROOT / "data_raw" / "pinta_n03_07" / "geotiff"

KEEP_PREFIXES = [
    "bathymetry/calculated_positive_values/",
    "picked_horizons_tops/",
    "sediment_thickness/",
    "SSS/",
]

SKIP_PREFIXES = [
    "bathymetry/original_caris/",
]

SKIP_SUFFIXES = [
    ".ovr",
]

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for info in z.infolist():
            name = info.filename
            if name.endswith("/"):
                continue

            name_l = name.lower()

            if any(name_l.startswith(p.lower()) for p in SKIP_PREFIXES):
                skipped += 1
                continue

            if any(name_l.endswith(s) for s in SKIP_SUFFIXES):
                skipped += 1
                continue

            if not any(name_l.startswith(p.lower()) for p in KEEP_PREFIXES):
                skipped += 1
                continue

            out_path = OUT_DIR / name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with z.open(info, "r") as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            kept += 1

    print(f"Extracted files: {kept}")
    print(f"Skipped files:   {skipped}")
    print(f"Output folder:   {OUT_DIR}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
