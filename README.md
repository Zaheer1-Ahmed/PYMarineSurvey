# PyMarineSurvey

PyMarineSurvey is a lightweight Python pipeline for extracting, indexing, and quality-checking marine survey GeoTIFF layers (e.g., multibeam bathymetry grids, side-scan sonar mosaics, sub-bottom horizons, sediment thickness). It generates a reproducible input index, QC metrics, quicklook images, and a Markdown QC report.

## Data source

This repo does not include raw datasets. The example tile used in development was downloaded from the BSH PINTA portal:
https://pinta.bsh.de/N-3.7?lang=en&tab=daten

Please review and comply with the data provider’s terms, licensing, and attribution requirements.

## What this project does

- Extracts only the relevant GeoTIFF layers from the downloaded archive (minimal extraction)
- Indexes raster inputs into a single CSV file (paths, CRS, resolution, shape)
- Runs QC checks:
  - nodata ratio detection
  - distribution percentiles (p01, p50, p99)
  - quicklook render + histogram plots
  - simple GREEN/YELLOW/RED score and notes (including sparse-layer handling)
- Writes a Markdown report you can open and share

## Project structure

PyMarineSurvey/
data_raw/ # local only, not committed
pinta_n03_07/
N-03-07_GeoTIFF.zip # local only, not committed
geotiff/ # extracted rasters (local only, not committed)
outputs/ # generated artifacts (usually not committed)
inputs_index.csv
qc/qc_metrics.csv
reports/qc_report.md
quicklooks/*.png
scripts/
00_index_inputs.py
01_summarize_qc.py
fix_sss_nodata.py
extract_geotiff_minimal.py

## Setup

### 1) Create and activate environment (Conda)

```powershell
conda create -n pymarinesurvey -y python=3.10
conda activate pymarinesurvey

Run the pipeline

From the repo root:

1) Index inputs
python scripts\00_index_inputs.py

Output:

outputs/inputs_index.csv

2) Fix SSS nodata (optional but recommended if SSS uses a constant background like 251)
python scripts\fix_sss_nodata.py

This creates a cleaned GeoTIFF:

data_raw\pinta_n03_07\geotiff\SSS\BSH_N-03-07_nodata251.tif

3) Summarize QC results
python scripts\01_summarize_qc.py

Outputs:

outputs/qc/qc_metrics.csv

outputs/reports/qc_report.md

outputs/quicklooks/*.png

View results

Open the report:

outputs/reports/qc_report.md

Browse quicklooks:

outputs/quicklooks/

Windows shortcuts:
notepad outputs\reports\qc_report.md
explorer outputs\quicklooks


## Sample outputs (from one demo tile)

- Report: `demo/reports/qc_report.md`
- Quicklooks: `demo/quicklooks/`
