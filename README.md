# Wildfire Survival Manuscript Package

Repository accompanying the manuscript:

**A Survival--Probability Fusion Prototype for Censor-Aware Multi-Horizon Wildfire Threat Forecasting and Decision-Utility Evaluation**

Author: `ZhengYang Ren`  
Contact: `d58350215@gmail.com`  
Repository: `https://github.com/eee-afkk/wids2026-wildfire-survival`

## Overview

This repository is a submission-oriented manuscript package built around the current source-of-truth manuscript version, `manuscript/初稿v9.tex` (Version April 4, 2026). It contains the paper source, the final manuscript-numbered figures and tables, the core analysis scripts used to generate the reported outputs, and a limited set of raw/support files retained only for provenance and supplementary-material traceability.

The project studies censor-aware multi-horizon wildfire threat forecasting using a survival--probability fusion framework that combines an XGBoost AFT survival backbone with horizon-specific direct probability heads, calibration, and monotone fusion. In the current manuscript, the prototype achieves strong discrimination and low probability error on the WiDS Datathon 2026 wildfire task while preserving post-fusion monotonicity across the 12 h, 24 h, and 48 h horizons. The repository is intentionally organized to match manuscript numbering rather than a development-time export order so that journal editors, reviewers, and readers can locate each referenced figure, table, and supporting file directly.

## Quick Navigation

- `manuscript/初稿v9.tex`: current manuscript source of truth
- `manuscript/初稿v9.pdf`: compiled manuscript PDF
- `manuscript/figures/main/`: final main-text figures (`Figure 1` to `Figure 10`)
- `manuscript/figures/supplementary/`: final supplementary figures (`Figure S1` to `Figure S7`)
- `tables/main/`: manuscript-numbered main tables (`Table 1` to `Table 9`)
- `tables/supplementary/`: manuscript-numbered supplementary tables (`Table S0` to `Table S11`)
- `code/run_pipeline.py`: main analysis and manuscript-output pipeline
- `code/supplement_analysis.py`: supplementary analyses and robustness checks
- `code/plot_figure10.py`: standalone Figure 10 rendering utility for the main-text robustness diagnostics
- `manuscript_file_manifest.md`: source-to-final manifest for figures and tables
- `docs/data_README.md`: external data requirements and access notes

## Repository Layout

```text
wids2026-wildfire-survival/
|- README.md
|- manuscript_file_manifest.md
|- .gitignore
|- Definitions/
|- manuscript/
|  |- 初稿v9.tex
|  |- 初稿v9.pdf
|  `- figures/
|     |- main/
|     `- supplementary/
|- tables/
|  |- main/
|  `- supplementary/
|- code/
|  |- run_pipeline.py
|  |- supplement_analysis.py
|  `- plot_figure10.py
|- docs/
|  |- README.md
|  |- data_README.md
|  `- requirements.txt
`- raw_outputs/
   `- csv/
```

## Alignment with the Manuscript

The repository follows the final numbering used in `初稿v9`:

- Main figures: `Figure 1` to `Figure 10`
- Supplementary figures: `Figure S1` to `Figure S7`
- Main tables: `Table 1` to `Table 9`
- Supplementary tables: `Table S0` to `Table S11`

This means the curated filenames in `manuscript/figures/` and `tables/` are intentionally different from development-time export names used during earlier drafting. The full provenance mapping is recorded in `manuscript_file_manifest.md`.

## Code Entry Points

### Main analysis

File: `code/run_pipeline.py`

Primary responsibilities:

- nested 5 x 5 cross-validation
- survival--probability fusion model fitting
- main manuscript figures
- main manuscript tables
- manuscript-aligned support outputs

Typical usage:

```bash
python code/run_pipeline.py --data-dir /path/to/wids_data --output-dir /path/to/output
```

Required external inputs:

- `train.csv`
- `test.csv`

### Supplementary analyses

File: `code/supplement_analysis.py`

Primary responsibilities:

- proxy grouped cross-validation
- temporal blocked cross-validation
- informative censoring sensitivity analysis
- TRIPOD-style data-quality reporting
- NIFC WFIGS context analyses used in supplementary materials

Typical usage:

```bash
python code/supplement_analysis.py --train /path/to/train.csv --nifc /path/to/WFIGS_Interagency_Perimeters.csv --output /path/to/output
```

Required external inputs:

- WiDS `train.csv`
- NIFC WFIGS perimeter CSV downloaded separately

### Figure 10 rendering utility

File: `code/plot_figure10.py`

Primary responsibilities:

- renders the main-text robustness diagnostics figure
- combines grouped-CV discrimination and mean-Brier panels with censoring-sensitivity panels
- exports manuscript-ready `PDF` and submission-ready `TIFF` assets for `Figure 10`

Typical usage:

```bash
python code/plot_figure10.py --input_dir /path/to/supplement_output --output_dir /path/to/output
```

## Reproducibility Snapshot

Repository settings aligned with the current manuscript:

- Python 3.10 environment
- Outer CV folds: 5
- Inner CV folds: 5
- Primary random seed: 42
- Stability-analysis seeds: 42, 123, 314, 2024, 7777
- Main horizons: 12 h, 24 h, 48 h
- Supplement-only horizon: 72 h
- Bootstrap resamples: 1000
- Stacking regularization parameter: `lambda = 1.0`

Key manuscript-reported results reflected in this package:

- Pairwise concordance index proxy: 0.940
- Uno's IPCW C-index: 0.941
- Mean Brier score (IBS proxy): 0.041
- Zero post-fusion monotonicity violations
- 24 h risk-tercile observed event rates: 0.0%, 4.5%, 92.3%
- Proxy grouped-CV robustness check: mean C-index 0.9318 to 0.9283 and mean Brier 0.1352 to 0.1541
- Informative-censoring sensitivity range: 0.027 at 12 h, 0.113 at 24 h, and 0.249 at 48 h

## Included and Excluded Materials

Included:

- final manuscript source and compiled PDF
- final manuscript-numbered figures and tables
- analysis code for the main and supplementary results
- curated support CSV files retained for provenance
- MDPI/Fire LaTeX files required for compilation

Not included:

- WiDS Datathon raw competition data (`train.csv`, `test.csv`, `metaData.csv`)
- downloaded NIFC WFIGS raw perimeter data
- local build artifacts and temporary outputs
- trained model binaries or cached virtual environments

## External Data Access

The WiDS Datathon data are not redistributed here. See `docs/data_README.md` for the required external files and access notes. The supplementary NIFC WFIGS file must also be downloaded separately if you want to rerun `code/supplement_analysis.py`.

## Building the Manuscript

From the repository root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript manuscript/初稿v9.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript manuscript/初稿v9.tex
```

This compiles the manuscript against `Definitions/` and the curated figure paths stored under `manuscript/figures/`.

## Supplementary-Material Packaging Notes

The repository is arranged to match common journal supporting-information expectations:

- final numbered files are separated from raw/support files
- main and supplementary figures are stored in separate folders
- supplementary tables use stable manuscript numbering
- provenance is documented in `manuscript_file_manifest.md`
- restricted external datasets are excluded from redistribution
- build artifacts are not tracked
