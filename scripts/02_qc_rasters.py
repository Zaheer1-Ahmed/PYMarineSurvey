# scripts/02_qc_rasters.py
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUTS_CSV = ROOT / "outputs" / "inputs_index.csv"

OUT_QC_DIR = ROOT / "outputs" / "qc"
OUT_REPORT_DIR = ROOT / "outputs" / "reports"
OUT_QUICKLOOK_DIR = ROOT / "outputs" / "quicklooks"

OUT_QC_CSV = OUT_QC_DIR / "qc_metrics.csv"
OUT_REPORT_MD = OUT_REPORT_DIR / "qc_report.md"

# Keep QC lightweight on huge rasters by downsampling.
MAX_DIM = 1000  # quicklook sample size


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slug(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        elif ch in (" ", "/", "\\"):
            keep.append("_")
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _read_downsampled(src: rasterio.io.DatasetReader) -> np.ndarray:
    # Compute output shape preserving aspect ratio
    h, w = src.height, src.width
    scale = max(h / MAX_DIM, w / MAX_DIM, 1.0)
    oh = max(1, int(round(h / scale)))
    ow = max(1, int(round(w / scale)))

    arr = src.read(
        1,
        out_shape=(oh, ow),
        resampling=Resampling.nearest,
        masked=False,
    )

    # Convert to float for unified stats (uint8 stays OK)
    if arr.dtype.kind in ("i", "u"):
        return arr.astype(np.float32)
    return arr.astype(np.float32)


def _mask_valid(arr: np.ndarray, nodata) -> np.ndarray:
    m = np.ones(arr.shape, dtype=bool)

    # NaN handling
    m &= ~np.isnan(arr)

    # nodata handling
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        # For float32 nodata sentinels like -1e35, exact compare is fine in these files
        m &= (arr != float(nodata))

    return arr[m]


def _percentiles(x: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if x.size == 0:
        return None, None, None
    p01, p50, p99 = np.percentile(x, [1, 50, 99])
    return float(p01), float(p50), float(p99)


def _top_value_fraction(arr: np.ndarray, nodata) -> tuple[float | None, float | None]:
    # Find most frequent value in the downsampled array
    if arr.size == 0:
        return None, None
    # Use finite values only
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return None, None
    vals, cnt = np.unique(a.astype(np.float32), return_counts=True)
    i = int(np.argmax(cnt))
    topv = float(vals[i])
    topf = float(cnt[i] / cnt.sum())
    return topv, topf


def _score_row(product: str, nodata, nodata_ratio: float, p01, p50, p99, topv, topf) -> tuple[str, str]:
    # General checks
    notes = []

    # If percentiles missing, it's bad (no valid pixels in sample)
    if p50 is None:
        return "RED", "no_valid_pixels_in_sample"

    # Constant / near-constant data check
    if p01 is not None and p99 is not None and abs(p99 - p01) < 1e-6:
        return "RED", "near_constant_values"

    # Product-specific rules
    if product == "sss_mosaic":
        # Critical: if nodata is NOT set and a high value dominates, it's almost certainly nodata missing.
        if nodata is None and topv is not None and topf is not None and topv >= 250 and topf >= 0.20:
            return "RED", "nodata_missing_probably_250+_fill_value"

        # Otherwise, mosaics can legitimately have high nodata.
        if nodata_ratio > 0.98:
            return "RED", "extreme_nodata_ratio"
        if nodata_ratio > 0.80:
            return "YELLOW", "high_nodata_ratio_expected_for_mosaic"
        return "GREEN", "ok"

    if product in ("horizon_depth", "sediment_thickness"):
        # These layers can be sparse; nodata around 0.4–0.9 can be normal.
        if nodata_ratio > 0.995:
            return "RED", "almost_all_nodata"
        if nodata_ratio > 0.98:
            return "YELLOW", "very_high_nodata_sparse_layer"
        return "GREEN", "ok_sparse_layer"

    if product == "bathymetry_grid":
        # Bathymetry also can be patchy, but still flag extreme emptiness.
        if nodata_ratio > 0.98:
            return "RED", "almost_all_nodata"
        if nodata_ratio > 0.90:
            return "YELLOW", "very_high_nodata"
        return "GREEN", "ok"

    # Fallback
    if nodata_ratio > 0.98:
        return "RED", "almost_all_nodata"
    if nodata_ratio > 0.90:
        return "YELLOW", "very_high_nodata"
    return "GREEN", "ok"


def _write_quicklook(arr: np.ndarray, valid: np.ndarray, out_png: Path, title: str) -> None:
    # Normalize using robust percentiles on valid values
    if valid.size == 0:
        img = np.zeros(arr.shape, dtype=np.float32)
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
        img = (arr - vmin) / (vmax - vmin)
        img = np.clip(img, 0, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(img, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _write_hist(valid: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    if valid.size == 0:
        plt.text(0.5, 0.5, "No valid pixels in sample", ha="center", va="center")
        plt.axis("off")
    else:
        # Limit extreme tails for readability
        lo = float(np.percentile(valid, 1))
        hi = float(np.percentile(valid, 99))
        v = valid[(valid >= lo) & (valid <= hi)]
        plt.hist(v, bins=60)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> int:
    if not INPUTS_CSV.exists():
        raise FileNotFoundError(f"Missing {INPUTS_CSV}. Run scripts/00_index_inputs.py first.")

    _safe_mkdir(OUT_QC_DIR)
    _safe_mkdir(OUT_REPORT_DIR)
    _safe_mkdir(OUT_QUICKLOOK_DIR)

    df = pd.read_csv(INPUTS_CSV)

    # Prefer fixed SSS if both exist (keep only *_nodata*.tif for sss_mosaic when present)
    if "product" in df.columns and "rel_path" in df.columns:
        sss = df["product"].astype(str).eq("sss_mosaic")
        if sss.any():
            sss_df = df[sss].copy()
            has_fixed = sss_df["rel_path"].astype(str).str.contains("nodata", case=False, na=False)
            if has_fixed.any():
                df = pd.concat([df[~sss], sss_df[has_fixed]], ignore_index=True)

    rows = []
    for _, r in df.iterrows():
        sensor = str(r.get("sensor", "unknown"))
        product = str(r.get("product", "unknown"))
        rel_path = str(r.get("rel_path", "")).replace("\\", "/")

        abs_path = ROOT / "data_raw" / "pinta_n03_07" / "geotiff" / rel_path
        if not abs_path.exists():
            rows.append(
                dict(
                    sensor=sensor,
                    product=product,
                    rel_path=rel_path,
                    status="missing",
                    qc_score="RED",
                    qc_notes="file_missing",
                )
            )
            continue

        try:
            with rasterio.open(abs_path) as src:
                arr = _read_downsampled(src)
                nodata = src.nodata
                valid = _mask_valid(arr, nodata)

                nodata_ratio = float(1.0 - (valid.size / arr.size)) if arr.size else 1.0
                p01, p50, p99 = _percentiles(valid)
                topv, topf = _top_value_fraction(arr, nodata)

                qc_score, qc_notes = _score_row(
                    product=product,
                    nodata=nodata,
                    nodata_ratio=nodata_ratio,
                    p01=p01,
                    p50=p50,
                    p99=p99,
                    topv=topv,
                    topf=topf,
                )

                # quicklooks
                base = _slug(f"{sensor}__{product}__{Path(rel_path).stem}")
                ql_png = OUT_QUICKLOOK_DIR / f"{base}.png"
                hist_png = OUT_QUICKLOOK_DIR / f"{base}__hist.png"

                _write_quicklook(arr, valid, ql_png, title=f"{sensor} | {product}\n{rel_path}")
                _write_hist(valid, hist_png, title=f"Histogram (sampled, clipped) | {sensor} | {product}")

                rows.append(
                    dict(
                        sensor=sensor,
                        product=product,
                        status="ok",
                        rel_path=rel_path,
                        abs_path=str(abs_path),
                        driver=str(src.driver),
                        crs=str(src.crs) if src.crs else None,
                        width=int(src.width),
                        height=int(src.height),
                        pixel_size_x=float(src.transform.a) if src.transform else None,
                        pixel_size_y=float(abs(src.transform.e)) if src.transform else None,
                        dtype=str(src.dtypes[0]) if src.dtypes else None,
                        nodata=nodata,
                        nodata_ratio=nodata_ratio,
                        p01=p01,
                        p50=p50,
                        p99=p99,
                        top_value=topv,
                        top_fraction=topf,
                        qc_score=qc_score,
                        qc_notes=qc_notes,
                        quicklook_png=str(ql_png),
                        hist_png=str(hist_png),
                    )
                )

        except Exception as e:
            rows.append(
                dict(
                    sensor=sensor,
                    product=product,
                    rel_path=rel_path,
                    status="error",
                    qc_score="RED",
                    qc_notes=f"open_failed: {type(e).__name__}: {e}",
                )
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_QC_CSV, index=False)
    print(f"Wrote: {OUT_QC_CSV}")

    # Report
    score_counts = out["qc_score"].value_counts(dropna=False).to_dict()
    top = out.sort_values(
        by=["qc_score", "nodata_ratio"],
        ascending=[True, False],
        key=lambda s: s.map({"RED": 0, "YELLOW": 1, "GREEN": 2}).fillna(9) if s.name == "qc_score" else s,
    ).head(15)

    lines = []
    lines.append("# PyMarineSurvey QC Report\n")
    lines.append("## Score counts\n")
    for k in ["GREEN", "YELLOW", "RED"]:
        lines.append(f"- {k}: {score_counts.get(k, 0)}")
    lines.append("\n## Top flagged / highest nodata\n")
    lines.append(top[["sensor", "product", "qc_score", "nodata_ratio", "nodata", "p01", "p50", "p99", "qc_notes", "rel_path"]].to_markdown(index=False))
    lines.append("\n")

    OUT_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_REPORT_MD}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
