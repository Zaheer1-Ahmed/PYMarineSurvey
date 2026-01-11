from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QC_CSV = ROOT / "outputs" / "qc" / "qc_metrics.csv"
OUT_MD = ROOT / "outputs" / "reports" / "qc_report.md"
QUICKLOOK_DIR = ROOT / "outputs" / "quicklooks"


def slugify(s: str) -> str:
    """
    Make filenames match the quicklook naming style:
    keep letters/numbers/_/-, convert everything else to underscore.
    """
    s = s.replace("\\", "/")
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def quicklook_paths(sensor: str, product: str, rel_path: str) -> tuple[Path, Path]:
    stem = Path(rel_path).stem
    base = f"{sensor}__{product}__{stem}"
    base = slugify(base)
    ql = QUICKLOOK_DIR / f"{base}.png"
    hist = QUICKLOOK_DIR / f"{base}__hist.png"
    return ql, hist


def fmt(x):
    try:
        if pd.isna(x):
            return "n/a"
    except Exception:
        pass
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main() -> int:
    if not QC_CSV.exists():
        raise SystemExit(f"Missing: {QC_CSV}")

    df = pd.read_csv(QC_CSV)

    # Ensure expected columns exist gracefully
    for col in ["qc_score", "sensor", "product", "rel_path"]:
        if col not in df.columns:
            raise SystemExit(f"qc_metrics.csv missing required column: {col}")

    # Normalize qc_score values
    df["qc_score"] = df["qc_score"].astype(str).str.upper()

    # Counts (always show 3)
    counts = {k: int((df["qc_score"] == k).sum()) for k in ["GREEN", "YELLOW", "RED"]}

    lines: list[str] = []
    lines.append("# PyMarineSurvey QC Report")
    lines.append("")
    lines.append(f"- Metrics: `{QC_CSV.relative_to(ROOT)}`")
    lines.append(f"- Quicklooks: `{QUICKLOOK_DIR.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- GREEN: {counts['GREEN']}")
    lines.append(f"- YELLOW: {counts['YELLOW']}")
    lines.append(f"- RED: {counts['RED']}")
    lines.append("")

    # Order: RED -> YELLOW -> GREEN (most attention first)
    score_order = ["RED", "YELLOW", "GREEN"]

    # Pick a compact set of useful fields to print if they exist
    extra_fields = [
        "driver", "nodata", "nodata_ratio",
        "positive_fraction", "tail_fraction",
        "p01", "p50", "p99",
        "slope_deg_median", "roughness_median",
        "crs", "width", "height", "pixel_size_x",
        "qc_notes",
    ]
    extra_fields = [c for c in extra_fields if c in df.columns]

    lines.append("## Details (grouped by QC score)")
    lines.append("")

    for score in score_order:
        sub = df[df["qc_score"] == score].copy()
        lines.append(f"### {score}")
        lines.append("")
        if sub.empty:
            lines.append("- (none)")
            lines.append("")
            continue

        # Sort for stable reading
        sub = sub.sort_values(["sensor", "product", "rel_path"], kind="stable")

        for _, r in sub.iterrows():
            sensor = str(r["sensor"])
            product = str(r["product"])
            rel_path = str(r["rel_path"])

            lines.append(f"#### {sensor} | {product}")
            lines.append(f"- File: `{rel_path}`")

            # Embed quicklook/hist if files exist
            ql, hist = quicklook_paths(sensor, product, rel_path)
            ql_rel = ql.relative_to(ROOT)
            hist_rel = hist.relative_to(ROOT)

            if ql.exists():
                lines.append("")
                lines.append(f"![quicklook]({ql_rel.as_posix()})")
            if hist.exists():
                lines.append("")
                lines.append(f"![hist]({hist_rel.as_posix()})")

            # Print metrics in one compact bullet
            if extra_fields:
                parts = []
                for c in extra_fields:
                    if c in ["sensor", "product", "rel_path", "qc_score"]:
                        continue
                    parts.append(f"{c}={fmt(r.get(c))}")
                lines.append("")
                lines.append("- " + " | ".join(parts))

            lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_MD}")
    print(f"Summary: GREEN={counts['GREEN']} YELLOW={counts['YELLOW']} RED={counts['RED']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
