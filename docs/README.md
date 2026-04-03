# Survival–Probability Fusion Framework for Censor-Aware Multi-Horizon Wildfire Threat Forecasting

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and supplementary materials for the paper:

> **A Survival–Probability Fusion Framework for Censor-Aware Multi-Horizon Wildfire Threat Forecasting**
>
> Submitted to *Fire* (MDPI), 2026.

## Overview

This framework unifies a censor-aware survival backbone (XGBoost AFT) with horizon-targeted direct probability heads, post-hoc calibration, and monotone-constrained fusion to produce coherent threat-time probabilities at 12, 24, and 48 h horizons for wildfire evacuation zone triage.

**Key results** (nested 5-fold CV, 221 wildfire–zone encounters):
- Concordance index: 0.941 (95% CI: 0.928–0.956)
- Mean horizon-wise Brier score: 0.043
- Zero post-fusion monotonicity violations
- 24 h risk-tercile separation: 0.0% / 6.1% / 90.8% observed event rates

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── run_pipeline.py              # Main analysis script (single-file pipeline)
├── data/
│   └── README.md                # Data access instructions (WiDS Datathon 2026)
└── supplementary/
    ├── Table_S0_feature_dictionary.csv
    ├── Table_S1_72h_results.csv
    ├── Table_S2_hyperparameters.csv
    ├── Table_S3_twosided_pvalues.csv
    └── Table_S4_fold_metrics.csv
```

## Data

The dataset is from the [WiDS Datathon 2026](https://www.wids.io) wildfire prediction competition. Due to the competition's data usage terms, raw data files (`train.csv`, `test.csv`, `metaData.csv`) are **not included** in this repository.

**To obtain the data:**
1. Visit https://www.wids.io and register for the WiDS Datathon 2026
2. Download `train.csv`, `test.csv`, and `metaData.csv`
3. Place the files in a `data/` directory (or set the `WIDS_DATA_DIR` environment variable)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set data path and run

```bash
# Option A: set environment variable
export WIDS_DATA_DIR=/path/to/your/data
export WIDS_OUTPUT_DIR=./output
python run_pipeline.py

# Option B: place train.csv and test.csv in the current directory
python run_pipeline.py
```

### 3. Outputs

The pipeline produces:
- `output/paper_main_table.csv` — Core model comparison (Table 4 in paper)
- `output/ablation_results.csv` — Ablation study with paired Δ-metric bootstrap (Table 5)
- `output/calibration_summary.csv` — Horizon-specific calibration metrics (Table 3)
- `output/multi_seed_stability.csv` — 5-seed stability analysis (Table 7)
- `output/practical_variants.csv` — Lean variant comparison
- `output/recalibration_comparison.csv` — Pre- vs. post-recalibration (Table 8)
- `output/figures/` — All paper figures (PDF, SVG, PNG, TIFF at 600 DPI)

## Reproducibility

| Parameter | Value |
|-----------|-------|
| Primary random seed | 42 |
| Stability analysis seeds | 42, 123, 314, 2024, 7777 |
| Outer CV folds | 5 (stratified) |
| Inner CV folds | 5 (cross-fitting) |
| Bootstrap resamples | 1000 |
| Stacking λ | 1.0 |

### Software versions

- Python 3.10
- scikit-learn 1.3
- XGBoost 2.0
- LightGBM 4.1
- lifelines 0.28
- scikit-survival 0.23 (for RSF baseline)

## Pipeline Architecture

```
Module A: Gate Prior (logistic regression → OOF event probability)
Module B: Dual Survival Backbone
  ├── XGBoost AFT ensemble (normal / logistic / extreme value) → CDF probabilities
  └── XGBoost Cox ensemble → rank-normalised risk scores
Module C: Calibrated Horizon Heads (LR, GBC, HGB + beta calibration) @ 12/24/48 h
  ├── IPCW branch (24/48 h)
  └── Simple distance branch (24/48 h)
Module D: Uniform-Shrinkage Simplex Stacking + Monotone Fusion
  → P(T≤12h) ≤ P(T≤24h) ≤ P(T≤48h)
```

## Citation

```bibtex
@article{author2026wildfire,
  title={A Survival--Probability Fusion Framework for Censor-Aware Multi-Horizon Wildfire Threat Forecasting},
  author={First Author and Second Author},
  journal={Fire},
  year={2026},
  publisher={MDPI}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
