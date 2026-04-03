# Wildfire Survival Manuscript Package

Curated repository for the manuscript:

**A Survival--Probability Fusion Prototype for Censor-Aware Multi-Horizon Wildfire Threat Forecasting and Decision-Utility Evaluation**

This repository is organized as a manuscript-support package rather than a raw working directory. File names and folder layout follow the final manuscript numbering in `初稿v9`:

- Main figures: `Figure 1` to `Figure 9`
- Supplementary figures: `Figure S1` to `Figure S7`
- Main tables: `Table 1` to `Table 9`
- Supplementary tables: `Table S0` to `Table S11`

The repository URL referenced in the manuscript is:

`https://github.com/eee-afkk/wids2026-wildfire-survival`

## What Is Included

- Final manuscript source and compiled PDF
- Final manuscript-numbered figures and tables
- Main analysis code and supplementary-analysis code
- A small set of raw/support CSV files needed to trace how final supplementary files were assembled
- MDPI/Fire LaTeX class files required to compile the manuscript
- A manifest mapping final files to their raw sources

## What Is Not Included

- WiDS Datathon raw competition data (`train.csv`, `test.csv`, `metaData.csv`)
- Downloaded NIFC WFIGS raw perimeter data
- Local build artifacts and temporary outputs
- Trained model binaries or cached environments

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
|  |- references.bib
|  `- figures/
|     |- main/
|     `- supplementary/
|- tables/
|  |- main/
|  `- supplementary/
|- code/
|  |- run_pipeline.py
|  `- supplement_analysis.py
|- docs/
|  |- README.md
|  |- data_README.md
|  `- requirements.txt
`- raw_outputs/
   `- csv/
```

## Manuscript Correspondence

The curated repository is aligned to the manuscript as follows:

- `manuscript/figures/main/` contains the final paper figures used in `初稿v9.tex`
- `manuscript/figures/supplementary/` contains the final supplementary figures `S1` to `S7`
- `tables/main/` contains numbered CSV versions of main-text tables
- `tables/supplementary/` contains numbered supplementary tables, including generated files such as `Table_S3_ablation_pvalues.csv` and `Table_S9_informative_censoring_sensitivity.xlsx`
- `manuscript_file_manifest.md` records each final file and its raw/support source

## Code Entry Points

### 1. Main analysis pipeline

File: `code/run_pipeline.py`

Purpose:

- nested 5 x 5 cross-validation
- survival-probability fusion model fitting
- main manuscript figures
- main manuscript tables and supporting raw outputs

Typical usage:

```bash
python code/run_pipeline.py --data-dir /path/to/wids_data --output-dir /path/to/output
```

Expected external inputs:

- `train.csv`
- `test.csv`

### 2. Supplementary analysis pipeline

File: `code/supplement_analysis.py`

Purpose:

- proxy grouped cross-validation
- temporal blocked cross-validation
- informative censoring sensitivity analysis
- TRIPOD-style data-quality reporting
- NIFC WFIGS context analyses used for supplementary materials

Typical usage:

```bash
python code/supplement_analysis.py --train /path/to/train.csv --nifc /path/to/WFIGS_Interagency_Perimeters.csv --output /path/to/output
```

Expected external inputs:

- WiDS `train.csv`
- NIFC WFIGS perimeter CSV downloaded separately

## Reproducibility Notes

Main manuscript settings reflected in the repository:

- Python 3.10 environment
- Outer CV folds: 5
- Inner CV folds: 5
- Primary random seed: 42
- Stability-analysis seeds: 42, 123, 314, 2024, 7777
- Main horizons: 12 h, 24 h, 48 h
- Supplement-only horizon: 72 h
- Bootstrap resamples: 1000
- Stacking regularization parameter: `lambda = 1.0`

Primary manuscript results reported in `初稿v9`:

- Pairwise concordance index proxy: 0.940
- Uno's IPCW C-index: 0.941
- Mean Brier score (IBS proxy): 0.041
- Zero post-fusion monotonicity violations
- 24 h risk-tercile observed event rates: 0.0%, 4.5%, 92.3%

## Data Access

The WiDS Datathon data are not redistributed here. See `docs/data_README.md` for the required external files and how to obtain them. The supplementary NIFC WFIGS file must also be downloaded separately if you want to rerun `code/supplement_analysis.py`.

## Manuscript Build

From the repository root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript manuscript/初稿v9.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=manuscript manuscript/初稿v9.tex
```

This compiles the manuscript against `Definitions/` and the curated figure paths in `manuscript/figures/main/`.

## Supplementary-Material Packaging Notes

The repository is already arranged in a journal-friendly supporting-information format:

- final numbered files are separated from raw/support files
- manuscript-ready figures are separated into main and supplementary folders
- supplementary tables are stored under stable manuscript numbering
- provenance is documented in `manuscript_file_manifest.md`
- competition data are excluded to avoid redistribution issues
- build artifacts are intentionally not tracked

Residual caveats:

- the supplementary DOI placeholder in the manuscript remains a publisher-stage item
- no trained model binaries are archived in this repository
