# scripts/01_quicklook_qc.py
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import rasterio
from rasterio.enums import Resampling

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class QCConfig:
    max_dim_px: int
    nodata_green_max: float
    nodata_yellow_max: float
    positive_fraction_yellow_max: float
    positive_fraction_red_max: float
    tail_fraction_yellow_max: float
    tail_fraction_red_max: float
    max_sample: int
    seed: int


def load_config(path: Path) -> QCConfig:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    return QCConfig(
        max_dim_px=int(cfg["quicklook"]["max_dim_px"]),
        nodata_green_max=float(cfg["qc_thresholds"]["nodata_green_max"]),
        nodata_yellow_max=float(cfg["qc_thresholds"]["nodata_yellow_max"]),
        positive_fraction_yellow_max=float(cfg["qc_thresholds"]["positive_fraction_yellow_max"]),
        positive_fraction_red_max=float(cfg["qc_thresholds"]["positive_fraction_red_max"]),
        tail_fraction_yellow_max=float(cfg["qc_thresholds"]["tail_fraction_yellow_max"]),
        tail_fraction_red_max=float(cfg["qc_thresholds"]["tail_fraction_red_max"]),
        max_sample=int(cfg["sampling"]["max_sample"]),
        seed=int(cfg["sampling"]["seed"]),
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(s: str) -> str:
    bad = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    out = s
    for b in bad:
        out = out.replace(b, "_")
    return out.replace(" ", "_")


def downsample_read(
    src: rasterio.io.DatasetReader,
    band: int = 1,
    max_dim_px: int = 1800,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, float, float]:
    """
    Returns (arr, scale_x, scale_y) where scale_x = original_px / new_px.
    """
    w = src.width
    h = src.height
    scale = max(w / max_dim_px, h / max_dim_px, 1.0)
    out_w = int(math.ceil(w / scale))
    out_h = int(math.ceil(h / scale))

    arr = src.read(
        band,
        out_shape=(out_h, out_w),
        resampling=resampling,
        masked=True,
    )

    scale_x = w / out_w
    scale_y = h / out_h
    return arr, scale_x, scale_y


def sample_pixels(masked_arr: np.ma.MaskedArray, max_sample: int, rng: np.random.Generator) -> np.ndarray:
    data = masked_arr.compressed()
    if data.size == 0:
        return data
    if data.size <= max_sample:
        return data
    idx = rng.choice(data.size, size=max_sample, replace=False)
    return data[idx]


def robust_tail_fraction(values: np.ndarray, p01: float, p99: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean((values < p01) | (values > p99)))


def qc_score(cfg: QCConfig, nodata_ratio: float, positive_fraction: float, tail_fraction: float) -> Tuple[str, List[str]]:
    """
    Simple Green/Yellow/Red based on:
      - nodata ratio
      - positive fraction (useful for bathymetry-like layers, still informative for others)
      - tail fraction beyond robust bounds
    """
    notes: List[str] = []

    # nodata gates everything
    if nodata_ratio <= cfg.nodata_green_max:
        score = "GREEN"
    elif nodata_ratio <= cfg.nodata_yellow_max:
        score = "YELLOW"
        notes.append(f"nodata_ratio={nodata_ratio:.3f} above green threshold")
    else:
        score = "RED"
        notes.append(f"nodata_ratio={nodata_ratio:.3f} above yellow threshold")

    # positive fraction (soft)
    if not math.isnan(positive_fraction):
        if positive_fraction > cfg.positive_fraction_red_max:
            score = "RED"
            notes.append(f"positive_fraction={positive_fraction:.3f} high (possible sign/vertical datum issue)")
        elif positive_fraction > cfg.positive_fraction_yellow_max and score == "GREEN":
            score = "YELLOW"
            notes.append(f"positive_fraction={positive_fraction:.3f} moderate")

    # robust tail fraction
    if not math.isnan(tail_fraction):
        if tail_fraction > cfg.tail_fraction_red_max:
            score = "RED"
            notes.append(f"tail_fraction={tail_fraction:.3f} high (outliers / artifacts)")
        elif tail_fraction > cfg.tail_fraction_yellow_max and score == "GREEN":
            score = "YELLOW"
            notes.append(f"tail_fraction={tail_fraction:.3f} moderate")

    return score, notes


def render_quicklook_png(arr: np.ma.MaskedArray, out_png: Path, title: str) -> Dict[str, float]:
    """
    Render raster quicklook with robust stretch.
    Returns dict with p2/p98 used for stretch.
    """
    data = arr.compressed()
    if data.size == 0:
        # write an empty placeholder image
        fig = plt.figure(figsize=(10, 6), dpi=140)
        plt.title(title)
        plt.text(0.5, 0.5, "No valid data", ha="center", va="center")
        plt.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return {"p2": float("nan"), "p98": float("nan")}

    p2 = float(np.percentile(data, 2))
    p98 = float(np.percentile(data, 98))
    if not np.isfinite(p2) or not np.isfinite(p98) or p2 == p98:
        p2 = float(np.min(data))
        p98 = float(np.max(data))
        if p2 == p98:
            p98 = p2 + 1e-6

    fig = plt.figure(figsize=(10, 6), dpi=140)
    plt.title(title)

    # show masked as transparent-ish via masked array handling
    im = plt.imshow(arr, vmin=p2, vmax=p98)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return {"p2": p2, "p98": p98}


def render_hist_png(values: np.ndarray, out_png: Path, title: str) -> None:
    fig = plt.figure(figsize=(10, 4), dpi=140)
    plt.title(title)
    if values.size == 0:
        plt.text(0.5, 0.5, "No valid samples", ha="center", va="center")
        plt.axis("off")
    else:
        plt.hist(values, bins=80)
        plt.xlabel("Value")
        plt.ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def compute_slope_roughness(arr: np.ma.MaskedArray, pixel_size: float) -> Tuple[float, float]:
    """
    Very lightweight slope/roughness computed on downsampled array.
    Returns (slope_deg_median, roughness_std_median) over valid pixels.
    """
    if arr.count() == 0:
        return float("nan"), float("nan")

    a = arr.filled(np.nan).astype("float64")
    # gradients in x/y (meters)
    gy, gx = np.gradient(a, pixel_size, pixel_size)
    slope = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))

    # roughness = local std in 3x3 window (approx) using nan-safe approach
    # For speed and dependency-free, do a simple nanstd over 3x3 using padding.
    # This is not perfect but useful for QC signals.
    pad = 1
    ap = np.pad(a, pad_width=pad, mode="edge")
    rough = np.full_like(a, np.nan, dtype="float64")
    for y in range(a.shape[0]):
        y0 = y
        y1 = y + 3
        for x in range(a.shape[1]):
            x0 = x
            x1 = x + 3
            w = ap[y0:y1, x0:x1]
            rough[y, x] = np.nanstd(w)

    slope_med = float(np.nanmedian(slope))
    rough_med = float(np.nanmedian(rough))
    return slope_med, rough_med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="outputs/inputs_index.csv")
    ap.add_argument("--root", type=str, default="data_raw/pinta_n03_07/geotiff")
    ap.add_argument("--config", type=str, default="config/qc.yaml")
    ap.add_argument("--out", type=str, default="outputs")
    args = ap.parse_args()

    index_path = Path(args.index)
    root = Path(args.root)
    out_root = Path(args.out)
    cfg = load_config(Path(args.config))

    quick_dir = out_root / "quicklooks"
    qc_dir = out_root / "qc"
    rep_dir = out_root / "reports"
    ensure_dir(quick_dir)
    ensure_dir(qc_dir)
    ensure_dir(rep_dir)

    df = pd.read_csv(index_path)

    rng = np.random.default_rng(cfg.seed)

    rows: List[Dict[str, Any]] = []

    for i, r in df.iterrows():
        rel_path = str(r["rel_path"])
        sensor = str(r["sensor"])
        product = str(r["product"])
        status = str(r["status"])

        full_path = root / rel_path
        file_id = safe_filename(f"{sensor}__{product}__{Path(rel_path).stem}")

        out_img = quick_dir / f"{file_id}.png"
        out_hist = quick_dir / f"{file_id}__hist.png"

        record: Dict[str, Any] = {
            "sensor": sensor,
            "product": product,
            "status": status,
            "rel_path": rel_path,
            "full_path": str(full_path),
            "crs": r.get("crs", ""),
            "pixel_size_x": r.get("pixel_size_x", np.nan),
            "width": r.get("width", np.nan),
            "height": r.get("height", np.nan),
            "quicklook_png": str(out_img).replace("\\", "/"),
            "hist_png": str(out_hist).replace("\\", "/"),
        }

        if status != "ok" or not full_path.exists():
            record.update(
                {
                    "nodata_ratio": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "p01": np.nan,
                    "p50": np.nan,
                    "p99": np.nan,
                    "positive_fraction": np.nan,
                    "tail_fraction": np.nan,
                    "qc_score": "RED",
                    "qc_notes": "missing file or status not ok",
                }
            )
            rows.append(record)
            continue

        try:
            with rasterio.open(full_path) as src:
                nodata = src.nodata

                # For quicklook: choose resampling based on data type
                res = Resampling.bilinear
                arr_ds, sx, sy = downsample_read(src, band=1, max_dim_px=cfg.max_dim_px, resampling=res)

                # nodata ratio on downsampled (good proxy, fast)
                if isinstance(arr_ds, np.ma.MaskedArray):
                    nodata_ratio = float(1.0 - (arr_ds.count() / arr_ds.size))
                else:
                    nodata_ratio = float(np.mean(arr_ds == nodata)) if nodata is not None else 0.0

                # sample for stats (from downsampled masked array to keep it fast)
                sample = sample_pixels(arr_ds, cfg.max_sample, rng)

                if sample.size == 0:
                    vmin = vmax = p01 = p50 = p99 = np.nan
                    pos_frac = np.nan
                    tail_frac = np.nan
                else:
                    vmin = float(np.min(sample))
                    vmax = float(np.max(sample))
                    p01 = float(np.percentile(sample, 1))
                    p50 = float(np.percentile(sample, 50))
                    p99 = float(np.percentile(sample, 99))
                    pos_frac = float(np.mean(sample > 0))
                    tail_frac = robust_tail_fraction(sample, p01, p99)

                # render outputs
                stretch = render_quicklook_png(
                    arr_ds,
                    out_img,
                    title=f"{sensor} | {product}\n{rel_path}",
                )
                render_hist_png(sample, out_hist, title=f"Histogram (sample) | {sensor} | {product}")

                # optional slope/roughness for bathymetry-like layers
                slope_med = np.nan
                rough_med = np.nan
                if sensor == "multibeam_bathymetry":
                    px = float(r.get("pixel_size_x", np.nan))
                    if np.isfinite(px) and px > 0:
                        slope_med, rough_med = compute_slope_roughness(arr_ds, pixel_size=px)

                score, notes = qc_score(cfg, nodata_ratio, pos_frac, tail_frac)

                record.update(
                    {
                        "nodata_ratio": nodata_ratio,
                        "min": vmin,
                        "max": vmax,
                        "p01": p01,
                        "p50": p50,
                        "p99": p99,
                        "positive_fraction": pos_frac,
                        "tail_fraction": tail_frac,
                        "stretch_p2": stretch.get("p2", np.nan),
                        "stretch_p98": stretch.get("p98", np.nan),
                        "slope_deg_median": slope_med,
                        "roughness_median": rough_med,
                        "qc_score": score,
                        "qc_notes": "; ".join(notes) if notes else "",
                    }
                )

        except Exception as e:
            record.update(
                {
                    "nodata_ratio": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "p01": np.nan,
                    "p50": np.nan,
                    "p99": np.nan,
                    "positive_fraction": np.nan,
                    "tail_fraction": np.nan,
                    "qc_score": "RED",
                    "qc_notes": f"exception: {type(e).__name__}: {e}",
                }
            )

        rows.append(record)

    out_csv = qc_dir / "qc_metrics.csv"
    out_md = rep_dir / "qc_report.md"

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    # Markdown report
    lines: List[str] = []
    lines.append("# PyMarineSurvey QC Report")
    lines.append("")
    lines.append(f"- Index: `{index_path.as_posix()}`")
    lines.append(f"- Data root: `{root.as_posix()}`")
    lines.append(f"- Generated: quicklooks + metrics")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    score_counts = out_df["qc_score"].value_counts(dropna=False).to_dict()
    lines.append(f"- GREEN: {int(score_counts.get('GREEN', 0))}")
    lines.append(f"- YELLOW: {int(score_counts.get('YELLOW', 0))}")
    lines.append(f"- RED: {int(score_counts.get('RED', 0))}")
    lines.append("")
    lines.append("## Details (per layer)")
    lines.append("")

    for _, rr in out_df.iterrows():
        lines.append(f"### {rr['sensor']} | {rr['product']}")
        lines.append(f"- File: `{rr['rel_path']}`")
        lines.append(f"- CRS: `{rr.get('crs','')}`")
        lines.append(
            f"- Size: {rr.get('width','?')} x {rr.get('height','?')} px, pixel_x={rr.get('pixel_size_x', 'na')}"
        )
        lines.append(f"- QC Score: **{rr['qc_score']}**")
        if isinstance(rr.get("qc_notes", ""), str) and rr["qc_notes"]:
            lines.append(f"- Notes: {rr['qc_notes']}")
        lines.append("")
        lines.append(f"![quicklook]({rr['quicklook_png']})")
        lines.append("")
        lines.append(f"![hist]({rr['hist_png']})")
        lines.append("")
        lines.append(
            f"- nodata_ratio={rr.get('nodata_ratio', np.nan)} | p01={rr.get('p01', np.nan)} | p50={rr.get('p50', np.nan)} | p99={rr.get('p99', np.nan)}"
        )
        if rr.get("sensor") == "multibeam_bathymetry":
            lines.append(
                f"- slope_deg_median={rr.get('slope_deg_median', np.nan)} | roughness_median={rr.get('roughness_median', np.nan)}"
            )
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote metrics: {out_csv}")
    print(f"Wrote report:  {out_md}")
    print(f"Quicklooks:    {quick_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
