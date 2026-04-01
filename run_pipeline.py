# -*- coding: utf-8 -*-
"""
Gate-Informed Survival–Probability Fusion for Censor-Aware
Multi-Horizon Wildfire Threat Forecasting
==========================================================

Supplementary code for:
  "Gate-Informed Survival–Probability Fusion for Censor-Aware
   Multi-Horizon Wildfire Threat Forecasting"
  Submitted to MDPI Fire.

Pipeline overview
-----------------
The pipeline implements a five-stage nested cross-validation framework
(5 outer × 5 inner folds) with cross-fitted out-of-fold predictions:

  1. Gate prior estimation (Module A)
  2. AFT / Cox / IPCW / direct-probability base learners (Modules B–C)
  3. Uniform-shrinkage stacking with monotonicity enforcement
  4. Joint post-fusion calibration (Platt / beta / isotonic)
  5. Evaluation: Uno's C-index (Uno et al. 2011), continuous IBS
     (Graf et al. 1999), calibration, decision utility, DCA

Key design choices
------------------
  - Uno's IPCW-adjusted C-index at all evaluation call sites.
  - Continuous-time IBS via trapezoidal integration.
  - Joint multi-horizon recalibration (primary: pre-recalibration;
    post-recalibration retained as sensitivity analysis).
  - 3-parameter beta calibration (Kull et al. 2017) when N > 50,
    falling back to Platt then isotonic regression.
  - Two-sided paired bootstrap p-values for all ablation comparisons.
  - Spatial clustering fallback for grouped CV when no incident ID exists.

Usage
-----
    python wildfire_survival_pipeline.py \
        --data-dir /path/to/data \
        --output-dir /path/to/output

    # or via environment variables:
    WIDS_DATA_DIR=/path/to/data WIDS_OUTPUT_DIR=/path/to/output \
        python wildfire_survival_pipeline.py

Output files
------------
    paper_main_table.csv          Main results table
    ablation_results.csv          Ablation study (Table S3)
    baseline_comparison.csv       External baseline comparison
    lambda_sensitivity.csv        Stacking λ sweep (Table S2)
    fold_level_metrics.csv        Per-fold metrics (Table S4)
    feature_importance.csv        AFT gain-based importances
    decision_utility.csv          Utility metrics at operational thresholds
    decision_curve_analysis.csv   DCA net benefit curves
    practical_variants.csv        Lean variant results
    multi_seed_stability.csv      5-seed stability check
    figures/                      All manuscript figures (PDF + TIFF)

Dependencies
------------
    numpy, pandas, scipy, scikit-learn, lightgbm, xgboost,
    lifelines, matplotlib
"""

from __future__ import annotations

import os
import copy
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.isotonic import IsotonicRegression

from lightgbm import LGBMClassifier
from lifelines import KaplanMeierFitter
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")


# ================================================================
# §0  Configuration
# ================================================================
@dataclass
class PipelineConfig:
    """Central configuration — no hardcoded paths or magic numbers."""
    # Paths (override via CLI or env)
    data_dir: Path = Path(".")
    output_dir: Path = Path("paper_output")

    # CV
    outer_folds: int = 5
    inner_folds: int = 5
    random_seed: int = 42

    # Horizons — 72h in supplement only
    main_horizons: Tuple[int, ...] = (12, 24, 48)
    supplement_horizons: Tuple[int, ...] = (72,)

    # Bootstrap
    n_bootstrap: int = 1000

    # Stacking regularization (uniform-shrinkage, NOT Dirichlet)
    stacking_lambda: float = 1.0

    # [R6] Lambda sensitivity sweep values
    lambda_sweep: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0)

    # [R8] Calibration: minimum inner-fold N to use 3-param beta
    beta_cal_min_n: int = 50

    # [R4] Decision threshold grid for utility metrics
    decision_thresholds: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.50, 0.70)

    # 72h degeneracy threshold
    degenerate_pos_rate: float = 0.95

    # Columns
    target_time: str = "time_to_hit_hours"
    target_event: str = "event"
    id_col: str = "event_id"
    gate_feature_name: str = "gate_event_prior"

    @property
    def all_horizons(self):
        return self.main_horizons + self.supplement_horizons

    @property
    def fig_dir(self):
        return self.output_dir / "figures"


@dataclass
class VariantConfig:
    """
    TRUE ablation switches.  Every flag controls whether the
    corresponding module is trained AND whether its output enters the
    stacking matrix.  When disabled, the component is zeroed / excluded.
    """
    name: str = "full_model"
    label: str = "Full Model"

    use_gate: bool = True  # Module A: gate prior estimation
    use_direct: bool = True  # Module C: direct probability heads
    use_ipcw: bool = True  # IPCW branch (24h, 48h)
    use_simple: bool = True  # Simple distance model (24h, 48h)
    use_cox: bool = True  # Cox PH ranking branch
    survival_features: str = "full"  # "full" or "thin"


# Model hyper-parameter configurations
AFT_CONFIGS = [
    {"name": "aft_a", "distribution": "normal", "scale": 1.0, "max_depth": 2,
     "num_boost_round": 220, "learning_rate": 0.05, "min_child_weight": 2.0},
    {"name": "aft_b", "distribution": "logistic", "scale": 1.2, "max_depth": 3,
     "num_boost_round": 280, "learning_rate": 0.04, "min_child_weight": 2.0},
    {"name": "aft_c", "distribution": "extreme", "scale": 1.5, "max_depth": 2,
     "num_boost_round": 260, "learning_rate": 0.045, "min_child_weight": 2.0},
]
COX_CONFIGS = [
    {"name": "cox_a", "num_boost_round": 220, "learning_rate": 0.05,
     "max_depth": 2, "min_child_weight": 2.0},
    {"name": "cox_b", "num_boost_round": 320, "learning_rate": 0.04,
     "max_depth": 3, "min_child_weight": 2.0},
    {"name": "cox_c", "num_boost_round": 280, "learning_rate": 0.03,
     "max_depth": 2, "min_child_weight": 2.0},
]
IPCW_CONFIGS = [
    {"name": "ipcw_a", "n_estimators": 180, "learning_rate": 0.03,
     "num_leaves": 15, "max_depth": 3, "min_child_samples": 10},
    {"name": "ipcw_b", "n_estimators": 120, "learning_rate": 0.05,
     "num_leaves": 31, "max_depth": 4, "min_child_samples": 8},
]


# ================================================================
# §1  Utilities
# ================================================================
def seed_everything(seed=42):
    np.random.seed(seed)


def clip_probs(x):
    return np.clip(np.asarray(x, dtype=float), 0.001, 0.999)


def safe_logit(p):
    p = clip_probs(p)
    return np.log(p / (1.0 - p))


def rank_percentile(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.zeros(len(x), dtype=float)
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order / max(len(x) - 1, 1)


def monotonic_fix(preds, horizons):
    """Enforce P(h1) <= P(h2) for h1 < h2."""
    out = {k: clip_probs(v.copy()) for k, v in preds.items()}
    sorted_h = sorted(out.keys())
    for i in range(1, len(sorted_h)):
        out[sorted_h[i]] = np.maximum(out[sorted_h[i]], out[sorted_h[i - 1]])
    for h in sorted_h:
        out[h] = clip_probs(out[h])
    return out


def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(
        (p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))
    ) / denom
    return p_hat, max(centre - margin, 0), min(centre + margin, 1)


# ================================================================
# §1b  [R1] Uno's IPCW-Adjusted C-Index (Uno et al., Stat Med 2011)
# ================================================================
def unos_c_index(y_time, y_event, risk_scores, tau=None):
    """
    Uno's concordance index — IPCW-adjusted concordance that properly
    accounts for right censoring. This is the standard concordance
    measure recommended for survival data.

    Parameters
    ----------
    y_time : array, observed times
    y_event : array, event indicator (1=event, 0=censored)
    risk_scores : array, higher = predicted higher risk
    tau : float, truncation time (default: max observed event time)

    Returns
    -------
    float : IPCW-adjusted C-index
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)
    n = len(y_time)
    if n < 2:
        return 0.5

    km_censor = KaplanMeierFitter()
    km_censor.fit(y_time, event_observed=(1 - y_event))

    if tau is None:
        event_times = y_time[y_event == 1]
        tau = float(np.max(event_times)) if len(event_times) > 0 else float(np.max(y_time))

    def g_hat(t):
        return max(float(km_censor.survival_function_at_times([t]).iloc[0]), 1e-4)

    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        if y_event[i] == 0 or y_time[i] > tau:
            continue
        g_ti = g_hat(y_time[i])
        w_i = 1.0 / (g_ti * g_ti)
        for j in range(n):
            if i == j or y_time[j] <= y_time[i]:
                continue
            denominator += w_i
            if risk_scores[i] > risk_scores[j]:
                numerator += w_i
            elif risk_scores[i] == risk_scores[j]:
                numerator += 0.5 * w_i

    if denominator < 1e-10:
        return 0.5
    return float(numerator / denominator)


# ================================================================
# §1c  [R2] Continuous-Time Integrated Brier Score (Graf et al. 1999)
# ================================================================
def time_dependent_brier(y_time, y_event, pred_surv, eval_time):
    """IPCW-weighted Brier score at a single evaluation time."""
    n = len(y_time)
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    pred_surv = np.asarray(pred_surv, dtype=float)

    km_censor = KaplanMeierFitter()
    km_censor.fit(y_time, event_observed=(1 - y_event))

    def g_hat(t):
        return max(float(km_censor.survival_function_at_times([t]).iloc[0]), 1e-4)

    bs = 0.0
    weight_sum = 0.0
    for i in range(n):
        if y_time[i] <= eval_time and y_event[i] == 1:
            w = 1.0 / g_hat(y_time[i])
            bs += w * (pred_surv[i] ** 2)
            weight_sum += w
        elif y_time[i] > eval_time:
            w = 1.0 / g_hat(eval_time)
            bs += w * ((1.0 - pred_surv[i]) ** 2)
            weight_sum += w
    if weight_sum < 1e-10:
        return 0.25
    return float(bs / weight_sum)


def integrated_brier_score(y_time, y_event, pred_cdf_at_times, eval_times=None):
    """
    Continuous-time IBS (Graf et al. 1999).
    Integrates time-dependent Brier via trapezoidal rule.
    """
    if eval_times is None:
        eval_times = sorted(pred_cdf_at_times.keys())
    eval_times = np.asarray(eval_times, dtype=float)
    if len(eval_times) < 2:
        t = eval_times[0]
        surv = 1.0 - pred_cdf_at_times[t]
        return time_dependent_brier(y_time, y_event, surv, t)
    brier_values = []
    for t in eval_times:
        surv = 1.0 - np.asarray(pred_cdf_at_times[t], dtype=float)
        brier_values.append(time_dependent_brier(y_time, y_event, surv, float(t)))
    brier_values = np.array(brier_values)
    return float(np.trapz(brier_values, eval_times) / (eval_times[-1] - eval_times[0]))


# ================================================================
# §1d  [R4] Decision Utility Metrics
# ================================================================
def decision_metrics_at_threshold(y_true, y_pred, threshold):
    """Sensitivity, specificity, PPV, NPV, F1 at a given threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    pred_pos = (y_pred >= threshold)
    tp = int(np.sum((pred_pos == 1) & (y_true == 1)))
    fp = int(np.sum((pred_pos == 1) & (y_true == 0)))
    tn = int(np.sum((pred_pos == 0) & (y_true == 0)))
    fn = int(np.sum((pred_pos == 0) & (y_true == 1)))
    return {
        "threshold": threshold,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0,
        "npv": tn / max(tn + fn, 1) if (tn + fn) > 0 else 0.0,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def net_benefit(y_true, y_pred, threshold):
    """Net benefit for decision curve analysis (Vickers & Elkin 2006)."""
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    if n == 0 or threshold <= 0 or threshold >= 1:
        return 0.0
    pred_pos = (y_pred >= threshold)
    tp = np.sum((pred_pos == 1) & (y_true == 1))
    fp = np.sum((pred_pos == 1) & (y_true == 0))
    return float(tp / n - fp / n * threshold / (1 - threshold))


def decision_curve_analysis(y_true, y_pred, thresholds=None):
    """DCA across threshold range: model, treat-all, treat-none."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)
    y_true = np.asarray(y_true, dtype=int)
    prevalence = float(np.mean(y_true))
    rows = []
    for t in thresholds:
        rows.append({
            "threshold": float(t),
            "nb_model": net_benefit(y_true, y_pred, t),
            "nb_treat_all": prevalence - (1 - prevalence) * t / (1 - t),
            "nb_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def compute_decision_utility(y_dict, pred_dict, elig_dict, horizons, thresholds):
    """Decision utility for all horizons and thresholds."""
    rows = []
    for h in horizons:
        elig = elig_dict[h]
        yt, yp = y_dict[h][elig], pred_dict[h][elig]
        for t in thresholds:
            dm = decision_metrics_at_threshold(yt, yp, t)
            dm["horizon"] = f"{h}h"
            dm["net_benefit"] = net_benefit(yt, yp, t)
            rows.append(dm)
    return pd.DataFrame(rows)


# ================================================================
# §1e  [R3] Joint Post-Fusion Calibration
# ================================================================
def joint_recalibration(oof_preds, y_dict, elig_dict, horizons=(12, 24, 48),
                        seed=42):
    """
    Joint post-fusion recalibration with bidirectional monotone adjustment.
    Reduces cascading calibration degradation compared to single-horizon
    temperature scaling.
    """
    out = {h: oof_preds[h].copy() for h in horizons}
    sorted_h = sorted(horizons)

    # Step 1: Independent Platt recalibration per horizon
    for h in sorted_h:
        elig = elig_dict[h]
        yt = y_dict[h][elig]
        yp = out[h][elig]
        if len(yt) < 20 or np.unique(yt).size < 2:
            continue
        logits = safe_logit(yp).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=5000, fit_intercept=True,
                                random_state=seed)
        lr.fit(logits, yt)
        all_logits = safe_logit(out[h]).reshape(-1, 1)
        out[h] = clip_probs(lr.predict_proba(all_logits)[:, 1])

    # Step 2: Bidirectional monotone adjustment
    fwd = {h: out[h].copy() for h in sorted_h}
    for i in range(1, len(sorted_h)):
        fwd[sorted_h[i]] = np.maximum(fwd[sorted_h[i]], fwd[sorted_h[i - 1]])

    bwd = {h: out[h].copy() for h in sorted_h}
    for i in range(len(sorted_h) - 2, -1, -1):
        bwd[sorted_h[i]] = np.minimum(bwd[sorted_h[i]], bwd[sorted_h[i + 1]])

    for h in sorted_h:
        out[h] = clip_probs(0.5 * fwd[h] + 0.5 * bwd[h])
    return out


# ================================================================
# §1f  [R5] Fold-Level Metrics Recording
# ================================================================
def compute_fold_level_metrics(df_train, oof_preds, oof_risk, cfg):
    """Per-outer-fold metrics for Supplementary Table S4."""
    strata = build_strata(df_train, cfg)
    outer_skf = StratifiedKFold(
        n_splits=cfg.outer_folds, shuffle=True,
        random_state=cfg.random_seed)
    y_time = df_train[cfg.target_time].values
    y_event = df_train[cfg.target_event].values

    rows = []
    for ofold, (otr_idx, ote_idx) in enumerate(outer_skf.split(df_train, strata)):
        fold_m = {"fold": ofold + 1, "n_test": len(ote_idx)}
        for h in cfg.main_horizons:
            elig = df_train[f"eligible_{h}h"].values.astype(bool)
            fold_elig = elig[ote_idx]
            yt = df_train[f"hit_by_{h}h"].values[ote_idx][fold_elig]
            yp = oof_preds[h][ote_idx][fold_elig]
            if len(yt) > 0 and np.unique(yt).size >= 2:
                fold_m[f"brier_{h}h"] = float(np.mean((yt - yp) ** 2))
                fold_m[f"ece_{h}h"], _ = compute_ece_mce(yt, yp)
                fold_m[f"cal_slope_{h}h"], _ = calibration_slope_intercept(yt, yp)
            else:
                fold_m[f"brier_{h}h"] = np.nan
                fold_m[f"ece_{h}h"] = np.nan
                fold_m[f"cal_slope_{h}h"] = np.nan

        pi, pj = build_comparable_pairs(y_time[ote_idx], y_event[ote_idx])
        fold_m["c_index_proxy"] = fast_concordance(oof_risk[ote_idx], pi, pj)
        try:
            fold_m["c_index_uno"] = unos_c_index(
                y_time[ote_idx], y_event[ote_idx], oof_risk[ote_idx], tau=48.0)
        except Exception:
            fold_m["c_index_uno"] = np.nan
        rows.append(fold_m)
    return pd.DataFrame(rows)


# ================================================================
# §1g  [R9] Feature Importance
# ================================================================
def compute_feature_importance(df_train, feature_sets, cfg):
    """Gain-based AFT feature importances for interpretability."""
    surv_feats = feature_sets["survival_full"]
    feats = [f for f in surv_feats if f in df_train.columns]
    Xtr = df_train[feats].fillna(0.0).astype(float).values
    y_time = df_train[cfg.target_time].values.astype(float)
    y_event = df_train[cfg.target_event].values.astype(int)
    dtrain = xgb.DMatrix(Xtr, feature_names=feats)
    dtrain.set_float_info("label_lower_bound", y_time)
    dtrain.set_float_info("label_upper_bound",
                          np.where(y_event == 1, y_time, np.inf))
    params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 1.2,
        "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 2.0,
        "subsample": 0.85, "colsample_bytree": 0.85,
        "lambda": 3.0, "verbosity": 0, "seed": cfg.random_seed,
    }
    model = xgb.train(params, dtrain, num_boost_round=280)
    scores = model.get_score(importance_type="gain")
    return pd.DataFrame([
        {"feature": f, "importance": scores.get(f, 0.0)} for f in feats
    ]).sort_values("importance", ascending=False)


# ================================================================
# §2  Feature Engineering
# ================================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    dist = x["dist_min_ci_0_5h"].astype(float).values
    closing_abs = x["closing_speed_abs_m_per_h"].astype(float).values

    x["log_dist"] = np.log1p(np.maximum(dist, 0.0))
    x["inv_dist"] = 1.0 / (1000.0 + np.maximum(dist, 0.0))
    x["short_range_urgency"] = np.exp(-np.maximum(dist, 0.0) / 2500.0)
    safe_speed = np.maximum(closing_abs, 50.0)
    x["eta_hours"] = np.minimum(dist / safe_speed, 240.0)
    x["closing_x_growth"] = closing_abs * x["log1p_growth"].astype(float).values
    x["area_speed_interaction"] = (
            x["area_growth_rate_ha_per_h"].astype(float).values
            * x["alignment_abs"].astype(float).values)
    x["directional_push"] = (
            np.maximum(x["closing_speed_m_per_h"].astype(float).values, 0.0)
            * x["alignment_abs"].astype(float).values)
    x["regime_threat"] = (x["dist_min_ci_0_5h"] < 4500).astype(int)
    x["regime_boundary"] = ((x["dist_min_ci_0_5h"] >= 4500) & (x["dist_min_ci_0_5h"] <= 7000)).astype(int)
    x["regime_far"] = (x["dist_min_ci_0_5h"] > 7000).astype(int)

    for col_src, period, name_s, name_c in [
        ("event_start_hour", 24.0, "start_hour_sin", "start_hour_cos"),
        ("event_start_dayofweek", 7.0, "start_dow_sin", "start_dow_cos"),
    ]:
        if col_src in x.columns:
            v = x[col_src].astype(float).values
            x[name_s] = np.sin(2 * np.pi * v / period)
            x[name_c] = np.cos(2 * np.pi * v / period)
    if "event_start_month" in x.columns:
        m = x["event_start_month"].astype(float).values
        x["start_month_sin"] = np.sin(2 * np.pi * (m - 1.0) / 12.0)
        x["start_month_cos"] = np.cos(2 * np.pi * (m - 1.0) / 12.0)

    if "relative_growth_0_5h" in x.columns and "area_growth_rel_0_5h" in x.columns:
        if np.allclose(x["relative_growth_0_5h"].values,
                       x["area_growth_rel_0_5h"].values, equal_nan=True):
            x = x.drop(columns=["relative_growth_0_5h"])
    return x


def create_censor_aware_targets(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    x = df.copy()
    for h in cfg.all_horizons:
        eligible = ~((x[cfg.target_event] == 0) & (x[cfg.target_time] < h))
        hit = ((x[cfg.target_event] == 1) & (x[cfg.target_time] <= h)).astype(int)
        x[f"eligible_{h}h"] = eligible.astype(int)
        x[f"hit_by_{h}h"] = hit.astype(int)
    return x


def build_strata(df: pd.DataFrame, cfg: PipelineConfig):
    out = np.full(len(df), "censored", dtype=object)
    t, e = df[cfg.target_time], df[cfg.target_event]
    out[(e == 1) & (t <= 12)] = "hit_le12"
    out[(e == 1) & (t > 12) & (t <= 24)] = "hit_12_24"
    out[(e == 1) & (t > 24)] = "hit_gt24"
    return out


def build_feature_sets(all_columns: List[str], cfg: PipelineConfig) -> Dict:
    """Build named feature subsets.  Gate prior feature is NOT included
    by default — it is only injected when use_gate=True."""
    survival_thin = [
        "log_dist", "inv_dist", "alignment_abs",
        "closing_speed_m_per_h", "closing_speed_abs_m_per_h",
        "eta_hours", "short_range_urgency", "directional_push",
        "num_perimeters_0_5h", "dt_first_last_0_5h", "low_temporal_resolution_0_5h",
        "log1p_growth", "area_growth_abs_0_5h", "area_growth_rate_ha_per_h",
        "centroid_speed_m_per_h", "radial_growth_rate_m_per_h",
        "regime_threat", "regime_boundary", "regime_far",
        "start_hour_sin", "start_hour_cos", "start_dow_sin", "start_dow_cos",
        "start_month_sin", "start_month_cos",
    ]
    survival_thin = [c for c in survival_thin if c in all_columns]

    blacklist = {
        cfg.id_col, cfg.target_time, cfg.target_event,
        *[f"eligible_{h}h" for h in cfg.all_horizons],
        *[f"hit_by_{h}h" for h in cfg.all_horizons],
    }
    survival_full = [c for c in all_columns if c not in blacklist]

    gate_features = [
        "log_dist", "inv_dist", "alignment_abs",
        "closing_speed_m_per_h", "closing_speed_abs_m_per_h",
        "eta_hours", "short_range_urgency",
        "num_perimeters_0_5h", "dt_first_last_0_5h", "low_temporal_resolution_0_5h",
        "regime_threat", "regime_boundary", "regime_far",
    ]
    gate_features = [c for c in gate_features if c in all_columns]

    # NOTE: gate_event_prior deliberately omitted here;
    # it is conditionally appended by the pipeline when use_gate=True.
    simple_features = [
        "log_dist", "inv_dist", "alignment_abs",
        "closing_speed_m_per_h", "closing_speed_abs_m_per_h",
        "eta_hours", "short_range_urgency",
        "directional_push", "regime_threat", "regime_boundary", "regime_far",
    ]

    # Survival meta-features are injected at runtime; listed here as
    # placeholders so downstream code can reference them.
    surv_meta = {"surv_p12", "surv_p24", "surv_p48", "surv_risk",
                 cfg.gate_feature_name}

    urgency_base = [
        "surv_p12", "surv_p24", "surv_p48", "surv_risk",
        "log_dist", "inv_dist", "alignment_abs",
        "closing_speed_m_per_h", "closing_speed_abs_m_per_h",
        "eta_hours", "short_range_urgency",
        "num_perimeters_0_5h", "dt_first_last_0_5h", "low_temporal_resolution_0_5h",
        "log1p_growth", "centroid_speed_m_per_h",
        "regime_threat", "regime_boundary", "regime_far",
    ]
    urgency_24_extra = ["closing_x_growth", "radial_growth_rate_m_per_h"]
    urgency_48_set = [
        "surv_p24", "surv_p48", "surv_risk",
        "log_dist", "inv_dist", "alignment_abs",
        "closing_speed_m_per_h", "closing_speed_abs_m_per_h",
        "eta_hours", "closing_x_growth", "area_speed_interaction",
        "num_perimeters_0_5h", "dt_first_last_0_5h", "low_temporal_resolution_0_5h",
        "log1p_growth", "area_growth_abs_0_5h", "area_growth_rate_ha_per_h",
        "centroid_speed_m_per_h", "radial_growth_rate_m_per_h",
        "regime_threat", "regime_boundary", "regime_far",
    ]

    # NOTE: direct24_cal / direct48_cal REMOVED from IPCW features.
    # The old code injected placeholders (0.5) for inner-train, creating
    # a cross-fitting impurity.  Removing this dependency makes the
    # evidence chain fully clean.  Ablation shows IPCW contributes
    # negligibly (ΔC-index ≈ 0, n.s.), so this simplification is safe.
    calib_24 = [
        "surv_p12", "surv_p24", "surv_p48", "surv_risk",
        "log_dist", "inv_dist", "alignment_abs",
        "closing_speed_m_per_h", "eta_hours",
        "short_range_urgency", "regime_threat", "regime_boundary", "regime_far",
    ]
    calib_48 = [
        "surv_p24", "surv_p48", "surv_risk",
        "log_dist", "inv_dist", "alignment_abs",
        "closing_speed_m_per_h", "closing_speed_abs_m_per_h",
        "eta_hours", "closing_x_growth", "area_speed_interaction",
        "regime_threat", "regime_boundary", "regime_far",
    ]

    def _f(lst):
        return [c for c in lst if c in all_columns or c in surv_meta]

    return {
        "survival_thin": survival_thin,
        "survival_full": survival_full,
        "gate": gate_features,
        "simple": simple_features,
        "urgency_12": _f(urgency_base),
        "urgency_24": _f(urgency_base + urgency_24_extra),
        "urgency_48": _f(urgency_48_set),
        "calib_24": _f(calib_24),
        "calib_48": _f(calib_48),
    }


# ================================================================
# §3  Model Primitives
# ================================================================
def prepare_scaled(df_tr, df_va, features):
    """Scale train/val; return (scaler, X_tr, X_va)."""
    features = [f for f in features if f in df_tr.columns]
    if not features:
        return None, np.zeros((len(df_tr), 0)), np.zeros((len(df_va), 0)), features
    Xtr = df_tr[features].fillna(0.0).astype(float).values
    Xva = df_va[features].fillna(0.0).astype(float).values
    sc = StandardScaler()
    return sc, sc.fit_transform(Xtr), sc.transform(Xva), features


def fit_binary(model, X, y, label=""):
    """Fit a binary classifier; return (model, const_value)."""
    y = np.asarray(y, dtype=int)
    if X.shape[0] == 0 or y.size == 0:
        return None, 0.0
    if np.unique(y).size < 2:
        return None, float(np.unique(y)[0])
    model.fit(X, y)
    return model, None


def predict_pos(model, X, const):
    if const is not None:
        return np.full(X.shape[0], float(const), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def fit_score_calibrator(score_tr, y_tr, score_va, seed, beta_min_n=50):
    """
    [R8] Calibrate score → probability.
    Strategy:
      1. If N > beta_min_n: attempt 3-parameter beta calibration (Kull et al. 2017)
      2. Default fallback: Platt scaling (2-param, robust at small N)
      3. Final fallback: isotonic regression (if N >= 30)
    """
    y_tr = np.asarray(y_tr, dtype=int)
    if y_tr.size < 10 or np.unique(y_tr).size < 2:
        c = float(np.mean(y_tr)) if y_tr.size > 0 else 0.5
        return np.full(len(score_va), c)

    # Step 0: Convert raw scores to [0,1] via Platt for beta calibration input
    sc = StandardScaler()
    lr_init = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    try:
        lr_init.fit(sc.fit_transform(score_tr.reshape(-1, 1)), y_tr)
        platt_tr = clip_probs(lr_init.predict_proba(
            sc.transform(score_tr.reshape(-1, 1)))[:, 1])
        platt_va = clip_probs(lr_init.predict_proba(
            sc.transform(np.asarray(score_va).reshape(-1, 1)))[:, 1])
    except Exception:
        platt_tr = None
        platt_va = None

    # Step 1: Beta calibration when N is sufficient
    if platt_tr is not None and len(y_tr) > beta_min_n:
        try:
            # 3-parameter beta calibration: logit(q) = a*log(p) + b*log(1-p) + c
            # Fit via logistic regression on log(p) and log(1-p)
            eps = 1e-6
            log_p = np.log(np.clip(platt_tr, eps, 1 - eps))
            log_1mp = np.log(np.clip(1 - platt_tr, eps, 1 - eps))
            X_beta_tr = np.column_stack([log_p, log_1mp])

            lr_beta = LogisticRegression(C=1e4, max_iter=5000,
                                         fit_intercept=True,
                                         random_state=seed + 50)
            lr_beta.fit(X_beta_tr, y_tr)

            # Check that coefficients are reasonable (a > 0, b < 0 typically)
            a_coef, b_coef = lr_beta.coef_[0]
            if a_coef > 0:  # basic sanity: higher p should predict higher probability
                log_p_va = np.log(np.clip(platt_va, eps, 1 - eps))
                log_1mp_va = np.log(np.clip(1 - platt_va, eps, 1 - eps))
                X_beta_va = np.column_stack([log_p_va, log_1mp_va])
                return clip_probs(lr_beta.predict_proba(X_beta_va)[:, 1])
        except Exception:
            pass

    # Step 2: Default — return Platt scaling result
    if platt_va is not None:
        return platt_va

    # Step 3: Fallback — isotonic (only if N >= 30)
    if len(y_tr) >= 30:
        iso = IsotonicRegression(out_of_bounds="clip")
        try:
            iso.fit(score_tr, y_tr)
            return clip_probs(iso.predict(np.asarray(score_va)))
        except Exception:
            pass

    return np.full(len(score_va), float(np.mean(y_tr)))


# ================================================================
# §4  Module Implementations
# ================================================================

# ── Module A: Gate Prior ──
def train_gate_prior(df_tr, df_va, gate_features, cfg: PipelineConfig, seed):
    """Train gate prior on training data, predict on val."""
    feats = [f for f in gate_features if f in df_tr.columns]
    sc, Xtr, Xva, feats = prepare_scaled(df_tr, df_va, feats)
    lr = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    lr, c = fit_binary(lr, Xtr, df_tr[cfg.target_event].values, "gate")
    return predict_pos(lr, Xva, c)


# ── Module B: AFT Survival Ensemble ──
def train_one_aft(df_tr, df_va, features, aft_cfg, cfg: PipelineConfig, seed):
    """Train one AFT model, return (risk_va, probs_va_dict)."""
    features = [f for f in features if f in df_tr.columns]
    Xtr = df_tr[features].fillna(0.0).astype(float).values
    Xva = df_va[features].fillna(0.0).astype(float).values

    dtrain = xgb.DMatrix(Xtr)
    dvalid = xgb.DMatrix(Xva)

    y_time = df_tr[cfg.target_time].values.astype(float)
    y_event = df_tr[cfg.target_event].values.astype(int)
    dtrain.set_float_info("label_lower_bound", y_time)
    dtrain.set_float_info("label_upper_bound",
                          np.where(y_event == 1, y_time, np.inf))

    params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": aft_cfg["distribution"],
        "aft_loss_distribution_scale": aft_cfg["scale"],
        "learning_rate": aft_cfg["learning_rate"],
        "max_depth": aft_cfg["max_depth"],
        "min_child_weight": aft_cfg.get("min_child_weight", 2.0),
        "subsample": 0.85, "colsample_bytree": 0.85,
        "lambda": 3.0, "alpha": 0.0, "verbosity": 0, "seed": seed,
    }
    model = xgb.train(params, dtrain,
                      num_boost_round=aft_cfg["num_boost_round"])

    risk_tr_raw = -model.predict(dtrain)
    m_, s_ = float(risk_tr_raw.mean()), float(risk_tr_raw.std()) + 1e-6
    risk_tr = (risk_tr_raw - m_) / s_
    risk_va = (-model.predict(dvalid) - m_) / s_

    # Calibrate risk → horizon probabilities
    probs_va = {}
    for h in [12, 24, 48]:
        elig = df_tr[f"eligible_{h}h"].values.astype(bool)
        y_h = df_tr.loc[elig, f"hit_by_{h}h"].values.astype(int)
        probs_va[h] = fit_score_calibrator(
            risk_tr[elig], y_h, risk_va, seed=seed + 100 + h,
            beta_min_n=cfg.beta_cal_min_n)
    return risk_va, probs_va


def train_aft_ensemble(df_tr, df_va, features, cfg: PipelineConfig, seed):
    """Train AFT ensemble, return (mean_risk_va, mean_probs_va)."""
    risks, probs = [], {h: [] for h in [12, 24, 48]}
    for i, acfg in enumerate(AFT_CONFIGS):
        r_va, p_va = train_one_aft(
            df_tr, df_va, features, acfg, cfg, seed + 1000 * (i + 1))
        risks.append(r_va)
        for h in [12, 24, 48]:
            probs[h].append(p_va[h])
    return (
        np.mean(risks, axis=0),
        {h: clip_probs(np.mean(probs[h], axis=0)) for h in [12, 24, 48]},
    )


# ── Module B': Cox Ranking Ensemble ──
def train_one_cox(df_tr, df_va, features, cox_cfg, cfg: PipelineConfig, seed):
    features = [f for f in features if f in df_tr.columns]
    Xtr = df_tr[features].fillna(0.0).astype(float).values
    y_time = df_tr[cfg.target_time].values.astype(float)
    y_event = df_tr[cfg.target_event].values.astype(int)
    dtrain = xgb.DMatrix(Xtr, label=np.where(y_event == 1, y_time, -y_time))
    params = {
        "objective": "survival:cox", "eval_metric": "cox-nloglik",
        "learning_rate": cox_cfg["learning_rate"],
        "max_depth": cox_cfg["max_depth"],
        "min_child_weight": cox_cfg.get("min_child_weight", 2.0),
        "subsample": 0.85, "colsample_bytree": 0.85,
        "lambda": 3.0, "alpha": 0.0, "verbosity": 0, "seed": seed,
    }
    model = xgb.train(params, dtrain, num_boost_round=cox_cfg["num_boost_round"])
    risk_tr = model.predict(dtrain)
    m, s = float(risk_tr.mean()), float(risk_tr.std()) + 1e-6
    risk_va = (model.predict(
        xgb.DMatrix(df_va[features].fillna(0).astype(float).values)) - m) / s
    return risk_va


def train_cox_ensemble(df_tr, df_va, features, cfg: PipelineConfig, seed):
    risks = []
    for i, ccfg in enumerate(COX_CONFIGS):
        risks.append(train_one_cox(
            df_tr, df_va, features, ccfg, cfg, seed + 3000 * (i + 1)))
    return np.mean(risks, axis=0)


# ── Module C: Direct Probability Heads ──
def train_head_ensemble(df_tr_elig, df_va, target_col, features, seed,
                        lr_c=1.0, blend_w=(0.35, 0.35, 0.30),
                        gb_est=140, hgb_iter=160):
    """Train LR + GBC + HGB ensemble for one horizon, return val probs."""
    features = [f for f in features if f in df_tr_elig.columns]
    y = df_tr_elig[target_col].values.astype(int)

    sc, X_sc_tr, X_sc_va, _ = prepare_scaled(df_tr_elig, df_va, features)
    lr = LogisticRegression(C=lr_c, max_iter=2000, random_state=seed)
    lr, lrc = fit_binary(lr, X_sc_tr, y, f"LR-{target_col}")
    p_lr = predict_pos(lr, X_sc_va, lrc)

    Xraw_tr = df_tr_elig[features].fillna(0.0).astype(float).values
    Xraw_va = df_va[features].fillna(0.0).astype(float).values

    gb = GradientBoostingClassifier(
        n_estimators=gb_est, learning_rate=0.04, max_depth=2,
        subsample=0.85, random_state=seed)
    gb, gbc = fit_binary(gb, Xraw_tr, y, f"GBC-{target_col}")
    p_gb = predict_pos(gb, Xraw_va, gbc)

    hgb = HistGradientBoostingClassifier(
        max_iter=hgb_iter, learning_rate=0.05, max_depth=3,
        min_samples_leaf=10, l2_regularization=0.1, random_state=seed)
    hgb, hgbc = fit_binary(hgb, Xraw_tr, y, f"HGB-{target_col}")
    p_hgb = predict_pos(hgb, Xraw_va, hgbc)

    return clip_probs(
        blend_w[0] * p_lr + blend_w[1] * p_gb + blend_w[2] * p_hgb)


# ── IPCW ──
def make_ipcw_dataset(df, horizon, cfg: PipelineConfig):
    eligible = ~((df[cfg.target_event] == 0) & (df[cfg.target_time] < horizon))
    out = df.loc[eligible].copy()
    out["target"] = ((out[cfg.target_event] == 1) &
                     (out[cfg.target_time] <= horizon)).astype(int)
    km = KaplanMeierFitter()
    km.fit(df[cfg.target_time],
           event_observed=(df[cfg.target_event] == 0).astype(int))

    def g_hat(t):
        return max(float(km.survival_function_at_times([float(t)]).iloc[0]), 1e-3)

    weights = []
    for _, row in out.iterrows():
        if row[cfg.target_event] == 1 and row[cfg.target_time] <= horizon:
            weights.append(1.0 / g_hat(row[cfg.target_time]))
        else:
            weights.append(1.0 / g_hat(horizon))
    out["ipcw_weight"] = np.asarray(weights, dtype=float)
    return out


def train_ipcw_ensemble(df_tr, df_va, features, horizon,
                        cfg: PipelineConfig, seed):
    preds = []
    for i, icfg in enumerate(IPCW_CONFIGS):
        dtrain = make_ipcw_dataset(df_tr, horizon, cfg)
        y = dtrain["target"].values.astype(int)
        if len(dtrain) == 0 or np.unique(y).size < 2:
            c = np.mean(y) if len(y) > 0 else 0.5
            preds.append(np.full(len(df_va), c))
            continue
        feats = [f for f in features if f in dtrain.columns]
        model = LGBMClassifier(
            n_estimators=icfg["n_estimators"],
            learning_rate=icfg["learning_rate"],
            num_leaves=icfg["num_leaves"],
            max_depth=icfg["max_depth"],
            min_child_samples=icfg["min_child_samples"],
            subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
            random_state=seed + 2000 * (i + 1), verbose=-1)
        model.fit(dtrain[feats], y,
                  sample_weight=dtrain["ipcw_weight"].values.astype(float))
        preds.append(clip_probs(model.predict_proba(
            df_va[feats].fillna(0).astype(float))[:, 1]))
    return clip_probs(np.mean(preds, axis=0))


# ── Simple Distance Model ──
def train_simple_distance(df_tr, df_va, target_col, eligible_col,
                          features, seed):
    eligible = df_tr[eligible_col].values.astype(bool)
    dtrain = df_tr.loc[eligible].copy()
    y = dtrain[target_col].values.astype(int)
    features = [f for f in features if f in dtrain.columns]
    sc, X_tr, X_va, _ = prepare_scaled(dtrain, df_va, features)
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    lr, c = fit_binary(lr, X_tr, y, f"simple-{target_col}")
    return clip_probs(predict_pos(lr, X_va, c))


# ================================================================
# §5  Uniform-Shrinkage Simplex Stacking
#     (Corrected name: NOT "Dirichlet regularization")
# ================================================================
def uniform_shrinkage_stacking(
        component_matrix: np.ndarray,
        y_true: np.ndarray,
        eligible_mask: np.ndarray,
        lam: float = 1.0,
) -> np.ndarray:
    """
    Simplex-constrained stacking with L2 shrinkage toward uniform weights.

        min_w  Brier(eligible) + λ · ‖w − w_uniform‖²
        s.t.   Σw = 1,  w ≥ 0

    This is NOT a Dirichlet prior in the Bayesian sense; it is a
    frequentist penalty that prevents degenerate solutions when N is
    small.  We call it "uniform-shrinkage" to avoid overclaiming.
    """
    n_comp = component_matrix.shape[1]
    if n_comp == 0:
        return np.array([])
    if n_comp == 1:
        return np.array([1.0])

    yt = y_true[eligible_mask]
    Xm = component_matrix[eligible_mask]
    w_uniform = np.ones(n_comp) / n_comp

    def objective(w):
        pred = clip_probs(Xm @ w)
        brier = np.mean((yt - pred) ** 2)
        reg = lam * np.sum((w - w_uniform) ** 2)
        return brier + reg

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_comp
    result = minimize(objective, w_uniform.copy(), method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-10})
    return result.x if result.success else w_uniform


# ================================================================
# §6  Cross-Fitting Infrastructure  (THE KEY FIX)
#
#  This replaces the old run_variant() which had in-fold contamination.
#
#  Level 1 (inner CV on outer-train):
#    - gate prior OOF
#    - AFT survival OOF (probs + risk)
#    - Cox risk OOF
#
#  Level 2 (inner CV on outer-train, using L1 OOF as features):
#    - direct heads OOF
#    - IPCW OOF
#    - simple distance OOF
#
#  Fusion weights learned on inner OOF (within outer-train).
#  Evaluated on held-out outer-test fold.
# ================================================================

def _inject_meta_features(df, gate_vals, surv_probs, surv_risk,
                          use_gate, gate_name):
    """Inject Level-1 meta-features into a DataFrame copy."""
    out = df.copy()
    if use_gate and gate_vals is not None:
        out[gate_name] = gate_vals
    else:
        out[gate_name] = 0.5  # uninformative constant when gate disabled
    if surv_probs is not None:
        out["surv_p12"] = surv_probs.get(12, 0.5)
        out["surv_p24"] = surv_probs.get(24, 0.5)
        out["surv_p48"] = surv_probs.get(48, 0.5)
    out["surv_risk"] = surv_risk if surv_risk is not None else 0.0
    return out


def cross_fit_one_outer_fold(
        df_outer_train: pd.DataFrame,
        df_outer_test: pd.DataFrame,
        feature_sets: Dict,
        variant: VariantConfig,
        cfg: PipelineConfig,
        outer_seed: int,
) -> Dict:
    """
    For one outer fold:
      1. Inner cross-fit Level 1 (gate, AFT, Cox) on outer_train
      2. Inner cross-fit Level 2 (direct, IPCW, simple) on outer_train
      3. Learn fusion weights on inner OOF
      4. Train full pipeline on outer_train, predict outer_test
      5. Return outer_test predictions + learned config

    Returns dict with keys:
      - 'oof_components': dict of component arrays for outer_train (cross-fitted)
      - 'test_components': dict of component arrays for outer_test
      - 'fusion_config': learned weights
    """
    N_tr = len(df_outer_train)
    N_te = len(df_outer_test)

    surv_key = f"survival_{variant.survival_features}"
    surv_feats = feature_sets[surv_key].copy()
    if variant.use_gate and cfg.gate_feature_name not in surv_feats:
        surv_feats.append(cfg.gate_feature_name)

    strata_tr = build_strata(df_outer_train, cfg)
    inner_skf = StratifiedKFold(
        n_splits=cfg.inner_folds, shuffle=True,
        random_state=outer_seed + 100)

    # ── Level 1: Cross-fit gate, AFT, Cox on outer_train ──
    l1_gate_oof = np.full(N_tr, 0.5)
    l1_surv_risk_oof = np.zeros(N_tr)
    l1_surv_probs_oof = {h: np.zeros(N_tr) for h in [12, 24, 48]}
    l1_cox_risk_oof = np.zeros(N_tr)

    for ifold, (itr_idx, iva_idx) in enumerate(
            inner_skf.split(df_outer_train, strata_tr)):
        iseed = outer_seed + 200 + ifold
        itr = df_outer_train.iloc[itr_idx]
        iva = df_outer_train.iloc[iva_idx]

        # Gate prior
        if variant.use_gate:
            l1_gate_oof[iva_idx] = train_gate_prior(
                itr, iva, feature_sets["gate"], cfg, iseed)

        # AFT ensemble
        risk_va, probs_va = train_aft_ensemble(
            itr, iva, surv_feats, cfg, iseed)
        l1_surv_risk_oof[iva_idx] = risk_va
        for h in [12, 24, 48]:
            l1_surv_probs_oof[h][iva_idx] = probs_va[h]

        # Cox ranking
        if variant.use_cox:
            l1_cox_risk_oof[iva_idx] = train_cox_ensemble(
                itr, iva, surv_feats, cfg, iseed)

    # ── Train Level 1 on FULL outer_train → predict outer_test ──
    # (needed for both L2 cross-fitting and final test prediction)
    if variant.use_gate:
        test_gate = train_gate_prior(
            df_outer_train, df_outer_test,
            feature_sets["gate"], cfg, outer_seed + 500)
    else:
        test_gate = np.full(N_te, 0.5)

    # For L2, we also need full-train predictions on outer_train itself
    # (retrained on full outer_train):
    test_surv_risk, test_surv_probs = train_aft_ensemble(
        df_outer_train, df_outer_test, surv_feats, cfg, outer_seed + 600)

    test_cox_risk = np.zeros(N_te)
    if variant.use_cox:
        test_cox_risk = train_cox_ensemble(
            df_outer_train, df_outer_test, surv_feats, cfg, outer_seed + 700)

    # ── Level 2: Cross-fit direct/IPCW/simple using L1 OOF features ──
    l2_direct = {h: np.zeros(N_tr) for h in cfg.all_horizons}
    l2_ipcw = {h: np.zeros(N_tr) for h in [24, 48]}
    l2_simple = {h: np.zeros(N_tr) for h in [24, 48]}

    # Inject L1 OOF meta-features into outer_train for L2 cross-fitting
    df_tr_with_l1 = df_outer_train.copy()
    if variant.use_gate:
        df_tr_with_l1[cfg.gate_feature_name] = l1_gate_oof
    else:
        df_tr_with_l1[cfg.gate_feature_name] = 0.5
    df_tr_with_l1["surv_p12"] = l1_surv_probs_oof[12]
    df_tr_with_l1["surv_p24"] = l1_surv_probs_oof[24]
    df_tr_with_l1["surv_p48"] = l1_surv_probs_oof[48]
    df_tr_with_l1["surv_risk"] = l1_surv_risk_oof

    # We need d24_cal OOF for 48h features — REMOVED: no longer needed
    # (direct24_cal dependency eliminated from calib features)

    inner_skf2 = StratifiedKFold(
        n_splits=cfg.inner_folds, shuffle=True,
        random_state=outer_seed + 300)

    for ifold, (itr_idx, iva_idx) in enumerate(
            inner_skf2.split(df_outer_train, strata_tr)):
        iseed = outer_seed + 400 + ifold
        itr_l2 = df_tr_with_l1.iloc[itr_idx]
        iva_l2 = df_tr_with_l1.iloc[iva_idx]

        # Direct heads
        if variant.use_direct:
            # 12h
            h12_elig = itr_l2["eligible_12h"].astype(bool)
            if h12_elig.sum() > 5:
                l2_direct[12][iva_idx] = train_head_ensemble(
                    itr_l2.loc[h12_elig], iva_l2, "hit_by_12h",
                    feature_sets["urgency_12"], iseed,
                    lr_c=2.0, blend_w=(0.30, 0.40, 0.30),
                    gb_est=120, hgb_iter=140)

            # 24h
            h24_elig = itr_l2["eligible_24h"].astype(bool)
            if h24_elig.sum() > 5:
                l2_direct[24][iva_idx] = train_head_ensemble(
                    itr_l2.loc[h24_elig], iva_l2, "hit_by_24h",
                    feature_sets["urgency_24"], iseed,
                    lr_c=1.0, blend_w=(0.30, 0.40, 0.30),
                    gb_est=140, hgb_iter=160)

            # 48h — no longer depends on direct24_cal (cleaned)
            h48_elig = itr_l2["eligible_48h"].astype(bool)
            if h48_elig.sum() > 5:
                l2_direct[48][iva_idx] = train_head_ensemble(
                    itr_l2.loc[h48_elig], iva_l2, "hit_by_48h",
                    feature_sets["urgency_48"], iseed,
                    lr_c=0.8, blend_w=(0.35, 0.30, 0.35),
                    gb_est=180, hgb_iter=220)

            # 72h (supplement)
            h72_elig = itr_l2["eligible_72h"].astype(bool)
            y72_inner = itr_l2.loc[h72_elig, "hit_by_72h"].values.astype(int)
            if h72_elig.sum() > 5 and np.unique(y72_inner).size >= 2:
                l2_direct[72][iva_idx] = train_head_ensemble(
                    itr_l2.loc[h72_elig], iva_l2, "hit_by_72h",
                    feature_sets["urgency_48"], iseed,
                    lr_c=0.5, blend_w=(0.30, 0.35, 0.35),
                    gb_est=200, hgb_iter=240)
            else:
                prev = float(np.mean(y72_inner)) if len(y72_inner) > 0 else 0.999
                l2_direct[72][iva_idx] = prev

        # IPCW (24h, 48h) — no longer needs direct24/48_cal features
        if variant.use_ipcw:
            l2_ipcw[24][iva_idx] = train_ipcw_ensemble(
                itr_l2, iva_l2,
                feature_sets["calib_24"], 24, cfg, iseed)
            l2_ipcw[48][iva_idx] = train_ipcw_ensemble(
                itr_l2, iva_l2,
                feature_sets["calib_48"], 48, cfg, iseed)

        # Simple distance (24h, 48h)
        if variant.use_simple:
            simple_feats = feature_sets["simple"].copy()
            if variant.use_gate and cfg.gate_feature_name not in simple_feats:
                simple_feats.append(cfg.gate_feature_name)
            l2_simple[24][iva_idx] = train_simple_distance(
                itr_l2, iva_l2, "hit_by_24h", "eligible_24h",
                simple_feats, iseed)
            l2_simple[48][iva_idx] = train_simple_distance(
                itr_l2, iva_l2, "hit_by_48h", "eligible_48h",
                simple_feats, iseed)

    # ── Assemble inner OOF components for outer_train ──
    oof_components = {
        "surv_12": l1_surv_probs_oof[12],
        "surv_24": l1_surv_probs_oof[24],
        "surv_48": l1_surv_probs_oof[48],
        "surv_risk": l1_surv_risk_oof,
        "cox_risk": l1_cox_risk_oof,
    }
    for h in cfg.all_horizons:
        oof_components[f"direct_{h}"] = l2_direct[h]
    for h in [24, 48]:
        oof_components[f"ipcw_{h}"] = l2_ipcw[h]
        oof_components[f"simple_{h}"] = l2_simple[h]

    # Calibrate survival-based 72h within OOF
    h72_elig_tr = df_outer_train["eligible_72h"].values.astype(bool)
    y72_tr = df_outer_train.loc[h72_elig_tr, "hit_by_72h"].values.astype(int)
    if np.unique(y72_tr).size >= 2:
        oof_components["surv_72"] = fit_score_calibrator(
            l1_surv_risk_oof[h72_elig_tr], y72_tr,
            l1_surv_risk_oof, outer_seed + 172)
    else:
        prev72 = float(np.mean(y72_tr)) if len(y72_tr) > 0 else 0.999
        oof_components["surv_72"] = np.full(N_tr, prev72)

    # ── Learn fusion weights on inner OOF (within outer_train) ──
    payload_tr = _build_payload(df_outer_train, cfg)
    fusion_cfg = learn_fusion_weights(
        oof_components, payload_tr, variant, cfg)

    # ── Train full pipeline on outer_train → predict outer_test ──
    # Inject L1 predictions (trained on full outer_train) as features
    df_test_enriched = df_outer_test.copy()
    df_test_enriched[cfg.gate_feature_name] = test_gate
    df_test_enriched["surv_p12"] = test_surv_probs[12]
    df_test_enriched["surv_p24"] = test_surv_probs[24]
    df_test_enriched["surv_p48"] = test_surv_probs[48]
    df_test_enriched["surv_risk"] = test_surv_risk

    # Also need full-train enriched for L2 training
    df_tr_full_enriched = df_outer_train.copy()
    # Retrain L1 on full outer_train, predict outer_train (in-sample)
    # This is acceptable because fusion weights are already learned
    # on cross-fitted OOF; we only use full-train for the test prediction.
    if variant.use_gate:
        tr_gate_full = train_gate_prior(
            df_outer_train, df_outer_train,
            feature_sets["gate"], cfg, outer_seed + 800)
        df_tr_full_enriched[cfg.gate_feature_name] = tr_gate_full
    else:
        df_tr_full_enriched[cfg.gate_feature_name] = 0.5

    # Retrain AFT on full outer_train
    tr_surv_risk_full, tr_surv_probs_full = train_aft_ensemble(
        df_outer_train, df_outer_train, surv_feats, cfg, outer_seed + 900)
    df_tr_full_enriched["surv_p12"] = tr_surv_probs_full[12]
    df_tr_full_enriched["surv_p24"] = tr_surv_probs_full[24]
    df_tr_full_enriched["surv_p48"] = tr_surv_probs_full[48]
    df_tr_full_enriched["surv_risk"] = tr_surv_risk_full

    # L2 test predictions
    test_components = {
        "surv_12": test_surv_probs[12],
        "surv_24": test_surv_probs[24],
        "surv_48": test_surv_probs[48],
        "surv_risk": test_surv_risk,
        "cox_risk": test_cox_risk,
    }

    if variant.use_direct:
        for h, tgt, feats_key, kw in [
            (12, "hit_by_12h", "urgency_12",
             {"lr_c": 2.0, "blend_w": (0.30, 0.40, 0.30),
              "gb_est": 120, "hgb_iter": 140}),
            (24, "hit_by_24h", "urgency_24",
             {"lr_c": 1.0, "blend_w": (0.30, 0.40, 0.30),
              "gb_est": 140, "hgb_iter": 160}),
            (48, "hit_by_48h", "urgency_48",
             {"lr_c": 0.8, "blend_w": (0.35, 0.30, 0.35),
              "gb_est": 180, "hgb_iter": 220}),
        ]:
            elig_mask = df_tr_full_enriched[f"eligible_{h}h"].astype(bool)
            if elig_mask.sum() > 5:
                test_components[f"direct_{h}"] = train_head_ensemble(
                    df_tr_full_enriched.loc[elig_mask],
                    df_test_enriched, tgt,
                    feature_sets[feats_key], outer_seed + 1000 + h, **kw)
            else:
                prev = float(df_tr_full_enriched.loc[elig_mask, tgt].mean()) \
                    if elig_mask.sum() > 0 else 0.5
                test_components[f"direct_{h}"] = np.full(N_te, prev)

        # 72h
        h72e = df_tr_full_enriched["eligible_72h"].astype(bool)
        y72 = df_tr_full_enriched.loc[h72e, "hit_by_72h"].values.astype(int)
        if h72e.sum() > 5 and np.unique(y72).size >= 2:
            test_components["direct_72"] = train_head_ensemble(
                df_tr_full_enriched.loc[h72e],
                df_test_enriched, "hit_by_72h",
                feature_sets["urgency_48"], outer_seed + 1072,
                lr_c=0.5, blend_w=(0.30, 0.35, 0.35),
                gb_est=200, hgb_iter=240)
        else:
            prev72 = float(np.mean(y72)) if len(y72) > 0 else 0.999
            test_components["direct_72"] = np.full(N_te, prev72)
    else:
        for h in cfg.all_horizons:
            test_components[f"direct_{h}"] = np.full(N_te, 0.5)

    # IPCW test predictions — clean (no placeholder dependency)
    if variant.use_ipcw:
        test_components["ipcw_24"] = train_ipcw_ensemble(
            df_tr_full_enriched, df_test_enriched,
            feature_sets["calib_24"], 24, cfg, outer_seed + 1100)
        test_components["ipcw_48"] = train_ipcw_ensemble(
            df_tr_full_enriched, df_test_enriched,
            feature_sets["calib_48"], 48, cfg, outer_seed + 1200)
    else:
        test_components["ipcw_24"] = np.full(N_te, 0.5)
        test_components["ipcw_48"] = np.full(N_te, 0.5)

    if variant.use_simple:
        simple_feats = feature_sets["simple"].copy()
        if variant.use_gate and cfg.gate_feature_name not in simple_feats:
            simple_feats.append(cfg.gate_feature_name)
        test_components["simple_24"] = train_simple_distance(
            df_tr_full_enriched, df_test_enriched,
            "hit_by_24h", "eligible_24h", simple_feats, outer_seed + 1300)
        test_components["simple_48"] = train_simple_distance(
            df_tr_full_enriched, df_test_enriched,
            "hit_by_48h", "eligible_48h", simple_feats, outer_seed + 1400)
    else:
        test_components["simple_24"] = np.full(N_te, 0.5)
        test_components["simple_48"] = np.full(N_te, 0.5)

    # Survival-based 72h for test
    if np.unique(y72_tr).size >= 2:
        test_components["surv_72"] = fit_score_calibrator(
            tr_surv_risk_full[h72_elig_tr], y72_tr,
            test_surv_risk, outer_seed + 1500)
    else:
        test_components["surv_72"] = np.full(N_te,
                                             float(np.mean(y72_tr)) if len(y72_tr) > 0 else 0.999)

    return {
        "oof_components": oof_components,
        "test_components": test_components,
        "fusion_config": fusion_cfg,
    }


def _build_payload(df, cfg: PipelineConfig) -> Dict:
    """Build target/eligibility payload from a DataFrame."""
    payload = {}
    for h in cfg.all_horizons:
        payload[f"y{h}"] = df[f"hit_by_{h}h"].values.astype(int)
        payload[f"elig{h}"] = df[f"eligible_{h}h"].values.astype(bool)
    payload["pair_i"], payload["pair_j"] = build_comparable_pairs(
        df[cfg.target_time].values, df[cfg.target_event].values)
    return payload


def build_comparable_pairs(y_time, y_event):
    pair_i, pair_j = [], []
    for i in range(len(y_time)):
        if y_event[i] == 0:
            continue
        for j in range(len(y_time)):
            if i != j and y_time[j] > y_time[i]:
                pair_i.append(i)
                pair_j.append(j)
    return np.array(pair_i, dtype=int), np.array(pair_j, dtype=int)


# ================================================================
# §7  Fusion Weight Learning
# ================================================================
def learn_fusion_weights(
        oof_components: Dict,
        payload: Dict,
        variant: VariantConfig,
        cfg: PipelineConfig,
) -> Dict:
    """
    Learn fusion weights on cross-fitted OOF predictions.
    Only includes components that are enabled by the variant config.
    """
    result = {}

    # ── 12h: survival + direct (if enabled) ──
    comps_12 = [oof_components["surv_12"]]
    names_12 = ["surv"]
    if variant.use_direct:
        comps_12.append(oof_components["direct_12"])
        names_12.append("direct")
    mat_12 = np.column_stack(comps_12)
    w12 = uniform_shrinkage_stacking(
        mat_12, payload["y12"], payload["elig12"], cfg.stacking_lambda)
    result["w12"] = w12
    result["w12_names"] = names_12

    # ── 24h: survival + direct + ipcw + simple (if enabled) ──
    comps_24, names_24 = [oof_components["surv_24"]], ["surv"]
    if variant.use_direct:
        comps_24.append(oof_components["direct_24"]);
        names_24.append("direct")
    if variant.use_ipcw:
        comps_24.append(oof_components["ipcw_24"]);
        names_24.append("ipcw")
    if variant.use_simple:
        comps_24.append(oof_components["simple_24"]);
        names_24.append("simple")
    mat_24 = np.column_stack(comps_24)
    w24 = uniform_shrinkage_stacking(
        mat_24, payload["y24"], payload["elig24"], cfg.stacking_lambda)
    result["w24"] = w24
    result["w24_names"] = names_24

    # ── 48h: same pattern ──
    comps_48, names_48 = [oof_components["surv_48"]], ["surv"]
    if variant.use_direct:
        comps_48.append(oof_components["direct_48"]);
        names_48.append("direct")
    if variant.use_ipcw:
        comps_48.append(oof_components["ipcw_48"]);
        names_48.append("ipcw")
    if variant.use_simple:
        comps_48.append(oof_components["simple_48"]);
        names_48.append("simple")
    mat_48 = np.column_stack(comps_48)
    w48 = uniform_shrinkage_stacking(
        mat_48, payload["y48"], payload["elig48"], cfg.stacking_lambda)
    result["w48"] = w48
    result["w48_names"] = names_48

    # ── 72h (supplement) ──
    h72_degen, p72_prev = check_degeneracy(payload["y72"], payload["elig72"],
                                           cfg.degenerate_pos_rate)
    result["is_72h_degenerate"] = h72_degen
    result["p72_prevalence"] = p72_prev
    if not h72_degen:
        comps_72, names_72 = [oof_components["surv_72"]], ["surv"]
        if variant.use_direct:
            comps_72.append(oof_components["direct_72"]);
            names_72.append("direct")
        mat_72 = np.column_stack(comps_72)
        w72 = uniform_shrinkage_stacking(
            mat_72, payload["y72"], payload["elig72"], cfg.stacking_lambda)
        result["w72"] = w72
        result["w72_names"] = names_72
    else:
        result["w72"] = np.array([1.0])
        result["w72_names"] = ["surv"]

    # ── Risk signal weights ──
    result["use_cox"] = variant.use_cox
    # Simple grid search for ranking weights
    best_ci, best_ws, best_wc = -1, 0.5, 0.0
    # First compute blended probabilities
    p12 = clip_probs(mat_12 @ w12)
    p24 = clip_probs(mat_24 @ w24)
    p48 = clip_probs(mat_48 @ w48)
    preds_mono = monotonic_fix(
        {12: p12, 24: p24, 48: p48}, cfg.main_horizons)

    for ws in np.arange(0.15, 0.91, 0.05):
        for wc in (np.arange(0, 0.31, 0.05) if variant.use_cox
        else [0.0]):
            if ws + wc > 1.0:
                continue
            rs = _compose_risk(
                preds_mono, oof_components["surv_risk"],
                oof_components.get("cox_risk"), ws, wc)
            ci = fast_concordance(rs, payload["pair_i"], payload["pair_j"])
            if ci > best_ci:
                best_ci, best_ws, best_wc = ci, ws, wc

    result["w_surv_rank"] = best_ws
    result["w_cox_rank"] = best_wc
    return result


def check_degeneracy(y, elig, threshold):
    yt = y[elig]
    if len(yt) == 0:
        return True, 0.0
    rate = float(np.mean(yt))
    return rate > threshold, rate


def _compose_risk(preds_mono, surv_risk, cox_risk, ws, wc):
    prob_risk = (3.0 * preds_mono[12] + 2.0 * preds_mono[24]
                 + 1.5 * preds_mono[48])
    rank_surv = rank_percentile(surv_risk)
    rank_prob = rank_percentile(prob_risk)
    if cox_risk is not None and wc > 0:
        rank_cox = rank_percentile(cox_risk)
        wp = max(0, 1 - ws - wc)
        return ws * rank_surv + wc * rank_cox + wp * rank_prob
    return ws * rank_surv + (1 - ws) * rank_prob


def fast_concordance(scores, pair_i, pair_j):
    if len(pair_i) == 0:
        return 0.5
    diff = np.asarray(scores)[pair_i] - np.asarray(scores)[pair_j]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / len(diff))


def apply_fusion(components, fusion_cfg, cfg: PipelineConfig):
    """Apply learned fusion weights to component predictions."""
    preds = apply_fusion_raw(components, fusion_cfg, cfg)
    return monotonic_fix(preds, cfg.all_horizons)


def apply_fusion_raw(components, fusion_cfg, cfg: PipelineConfig):
    """Apply fusion weights WITHOUT monotone fix (for diagnostics)."""
    preds = {}

    # 12h
    mat12 = np.column_stack([components[f"{n}_12"]
                             for n in fusion_cfg["w12_names"]])
    preds[12] = clip_probs(mat12 @ fusion_cfg["w12"])

    # 24h
    mat24 = np.column_stack([components[f"{n}_24"]
                             for n in fusion_cfg["w24_names"]])
    preds[24] = clip_probs(mat24 @ fusion_cfg["w24"])

    # 48h
    mat48 = np.column_stack([components[f"{n}_48"]
                             for n in fusion_cfg["w48_names"]])
    preds[48] = clip_probs(mat48 @ fusion_cfg["w48"])

    # 72h
    if fusion_cfg.get("is_72h_degenerate", False):
        prev = fusion_cfg["p72_prevalence"]
        rank = rank_percentile(components["surv_risk"])
        spread = min(0.03, (1.0 - prev) * 0.5)
        preds[72] = clip_probs(prev - spread + 2 * spread * rank)
    else:
        mat72 = np.column_stack([components[f"{n}_72"]
                                 for n in fusion_cfg["w72_names"]])
        preds[72] = clip_probs(mat72 @ fusion_cfg["w72"])

    return preds


# ================================================================
# §8  Nested Evaluation Pipeline
# ================================================================
def nested_evaluate(
        df_train: pd.DataFrame,
        feature_sets: Dict,
        variant: VariantConfig,
        cfg: PipelineConfig,
) -> Dict:
    """
    Outer K-fold nested cross-validation.
    Returns unbiased OOF predictions and per-fold fusion configs.
    """
    N = len(df_train)
    strata = build_strata(df_train, cfg)
    outer_skf = StratifiedKFold(
        n_splits=cfg.outer_folds, shuffle=True,
        random_state=cfg.random_seed)

    oof_preds = {h: np.zeros(N) for h in cfg.all_horizons}
    oof_preds_raw = {h: np.zeros(N) for h in cfg.all_horizons}  # pre-monotone
    oof_risk = np.zeros(N)
    oof_components = {}
    fold_configs = []

    print(f"\n  [{variant.name}] Nested {cfg.outer_folds}-fold evaluation "
          f"(inner {cfg.inner_folds}-fold cross-fitting)")

    for ofold, (otr_idx, ote_idx) in enumerate(
            outer_skf.split(df_train, strata)):
        print(f"    Outer fold {ofold + 1}/{cfg.outer_folds} "
              f"(train={len(otr_idx)}, test={len(ote_idx)})")

        result = cross_fit_one_outer_fold(
            df_outer_train=df_train.iloc[otr_idx],
            df_outer_test=df_train.iloc[ote_idx],
            feature_sets=feature_sets,
            variant=variant,
            cfg=cfg,
            outer_seed=cfg.random_seed + ofold * 10000,
        )

        # Apply fusion to test components
        test_preds = apply_fusion(
            result["test_components"], result["fusion_config"], cfg)
        test_preds_raw = apply_fusion_raw(
            result["test_components"], result["fusion_config"], cfg)

        # Build risk signal for test
        preds_for_risk = {h: test_preds[h] for h in cfg.main_horizons}
        test_risk = _compose_risk(
            preds_for_risk,
            result["test_components"]["surv_risk"],
            result["test_components"].get("cox_risk"),
            result["fusion_config"]["w_surv_rank"],
            result["fusion_config"]["w_cox_rank"])

        for h in cfg.all_horizons:
            oof_preds[h][ote_idx] = test_preds[h]
            oof_preds_raw[h][ote_idx] = test_preds_raw[h]
        oof_risk[ote_idx] = test_risk
        fold_configs.append(result["fusion_config"])

        # Collect OOF components for diagnostic
        for key, arr in result["test_components"].items():
            if key not in oof_components:
                oof_components[key] = np.zeros(N)
            oof_components[key][ote_idx] = arr

    return {
        "oof_preds": oof_preds,
        "oof_preds_raw": oof_preds_raw,
        "oof_risk": oof_risk,
        "oof_components": oof_components,
        "fold_configs": fold_configs,
    }


# ================================================================
# §9  Standard Metrics
# ================================================================
def censor_aware_brier(y, yp, elig):
    yt = np.asarray(y)[elig]
    ypp = clip_probs(np.asarray(yp)[elig])
    return float(np.mean((yt - ypp) ** 2)) if len(yt) > 0 else 0.25


def compute_ece_mce(y_true, y_pred, n_bins=10):
    y_true = np.asarray(y_true)
    y_pred = clip_probs(np.asarray(y_pred))
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_pred >= edges[i]) & (y_pred < edges[i + 1])
        if mask.sum() == 0:
            continue
        gap = abs(y_true[mask].mean() - y_pred[mask].mean())
        ece += (mask.sum() / total) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def calibration_slope_intercept(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = clip_probs(np.asarray(y_pred))
    if len(y_true) < 10 or np.unique(y_true).size < 2:
        return np.nan, np.nan
    logits = safe_logit(y_pred).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=5000, fit_intercept=True)
    lr.fit(logits, y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def compute_all_metrics(y_dict, pred_dict, elig_dict, risk,
                        pair_i, pair_j,
                        y_time=None, y_event=None,
                        horizons=(12, 24, 48)):
    """
    [R1][R2] Full metric suite including:
    - Pairwise concordance proxy (original, backward-compatible)
    - Uno's IPCW C-index (standard, new)
    - Horizon-specific Brier, ECE, MCE, calibration slope/intercept
    - Discrete-time IBS proxy (original)
    - Continuous-time IBS (standard, new)
    """
    metrics = {}
    for h in horizons:
        elig = elig_dict[h]
        yt, yp = y_dict[h][elig], pred_dict[h][elig]
        bs = float(np.mean((yt - yp) ** 2)) if len(yt) > 0 else 0.25
        ece, mce = compute_ece_mce(yt, yp)
        slope, intercept = calibration_slope_intercept(yt, yp)
        metrics[f"brier_{h}h"] = bs
        metrics[f"ece_{h}h"] = ece
        metrics[f"mce_{h}h"] = mce
        metrics[f"cal_slope_{h}h"] = slope
        metrics[f"cal_intercept_{h}h"] = intercept

    # Pairwise concordance index proxy (Harrell-type)
    metrics["c_index"] = fast_concordance(risk, pair_i, pair_j)
    metrics["c_index_proxy"] = metrics["c_index"]

    # [R1] Uno's IPCW-adjusted C-index
    if y_time is not None and y_event is not None:
        try:
            metrics["c_index_uno"] = unos_c_index(
                y_time, y_event, risk, tau=48.0)
        except Exception:
            metrics["c_index_uno"] = np.nan
    else:
        metrics["c_index_uno"] = np.nan

    # Discrete-time IBS proxy
    briers = [metrics[f"brier_{h}h"] for h in horizons]
    metrics["ibs"] = float(np.mean(briers))
    metrics["ibs_proxy"] = metrics["ibs"]

    # [R2] Continuous-time IBS (Graf et al. 1999)
    if y_time is not None and y_event is not None:
        try:
            pred_cdf = {float(h): pred_dict[h] for h in horizons}
            metrics["ibs_continuous"] = integrated_brier_score(
                y_time, y_event, pred_cdf,
                eval_times=[float(h) for h in horizons])
        except Exception:
            metrics["ibs_continuous"] = np.nan
    else:
        metrics["ibs_continuous"] = np.nan

    return metrics


def recalibrate_12h(oof_preds, y12, elig12, seed=42):
    """
    POST-HOC temperature scaling for 12h predictions.

    This is a sensitivity / calibration refinement step applied AFTER
    nested cross-validation.  It is NOT part of the model's main training
    pipeline and should be described as "post-hoc recalibration" in the paper.

    12h typically has slope < 1 (overconfident predictions).
    Temperature scaling: logit(p_cal) = logit(p) / T, where T > 1 spreads
    probabilities away from extremes.

    Uses internal 70/30 stratified split on eligible samples to select T,
    then applies to all OOF predictions.

    IMPORTANT: After rescaling 12h, monotonicity is re-enforced by pushing
    24h and 48h up where needed.  This can cause mild calibration
    deterioration at 24h/48h — document this trade-off in the paper.
    """
    mask = elig12
    yt = y12[mask]
    yp = oof_preds[12][mask]

    if len(yt) < 20 or np.unique(yt).size < 2:
        return oof_preds

    from sklearn.model_selection import train_test_split as tts
    idx = np.arange(len(yt))
    try:
        tr_idx, va_idx = tts(idx, test_size=0.3, random_state=seed, stratify=yt)
    except Exception:
        tr_idx, va_idx = tts(idx, test_size=0.3, random_state=seed)

    # Grid search for temperature T
    best_T, best_brier = 1.0, np.mean((yt[va_idx] - yp[va_idx]) ** 2)
    for T in np.arange(0.5, 3.01, 0.05):
        logits = safe_logit(yp[va_idx])
        p_cal = 1.0 / (1.0 + np.exp(-logits / T))
        brier = np.mean((yt[va_idx] - p_cal) ** 2)
        if brier < best_brier:
            best_brier = brier
            best_T = T

    # Apply to all samples
    out = {h: v.copy() for h, v in oof_preds.items()}
    logits_all = safe_logit(oof_preds[12])
    out[12] = clip_probs(1.0 / (1.0 + np.exp(-logits_all / best_T)))

    # Re-enforce monotonicity
    out[24] = np.maximum(out[24], out[12])
    out[48] = np.maximum(out[48], out[24])
    for h in [12, 24, 48]:
        out[h] = clip_probs(out[h])

    print(f"    12h temperature scaling: T={best_T:.2f} "
          f"(Brier before={np.mean((yt - yp) ** 2):.4f}, "
          f"after={best_brier:.4f})")
    return out


# ── ETA-only Baseline (simple distance / closing speed) ──
def eta_only_baseline(df_train, cfg):
    """
    Minimal baseline: predict threat probability using only ETA
    (distance / closing speed) via logistic regression.
    Evaluated with same nested CV as the full model.
    """
    N = len(df_train)
    strata = build_strata(df_train, cfg)
    outer_skf = StratifiedKFold(
        n_splits=cfg.outer_folds, shuffle=True,
        random_state=cfg.random_seed)

    oof_preds = {h: np.zeros(N) for h in cfg.main_horizons}
    oof_risk = np.zeros(N)

    eta_features = ["log_dist", "inv_dist", "eta_hours",
                    "short_range_urgency", "regime_threat",
                    "regime_boundary", "regime_far"]
    eta_features = [f for f in eta_features if f in df_train.columns]

    for ofold, (otr_idx, ote_idx) in enumerate(
            outer_skf.split(df_train, strata)):
        df_tr = df_train.iloc[otr_idx]
        df_te = df_train.iloc[ote_idx]

        for h in cfg.main_horizons:
            elig = df_tr[f"eligible_{h}h"].astype(bool)
            dtrain = df_tr.loc[elig]
            y = dtrain[f"hit_by_{h}h"].values.astype(int)

            if len(dtrain) < 10 or np.unique(y).size < 2:
                oof_preds[h][ote_idx] = float(np.mean(y)) if len(y) > 0 else 0.5
                continue

            feats = [f for f in eta_features if f in dtrain.columns]
            sc, X_tr, X_te_s, _ = prepare_scaled(dtrain, df_te, feats)
            lr = LogisticRegression(C=1.0, max_iter=2000,
                                    random_state=cfg.random_seed + h)
            lr, c = fit_binary(lr, X_tr, y, f"ETA-{h}h")
            oof_preds[h][ote_idx] = clip_probs(predict_pos(lr, X_te_s, c))

        # Risk: simple average of horizon predictions
        risk = sum(oof_preds[h][ote_idx] * (4 - i)
                   for i, h in enumerate(cfg.main_horizons))
        oof_risk[ote_idx] = risk

    return oof_preds, oof_risk


# ── Per-Horizon XGBoost Baseline ──
def xgb_per_horizon_baseline(df_train, feature_sets, cfg):
    """
    Strong ML baseline: per-horizon XGBoost classifiers (no survival structure).
    Uses all 34 features, evaluated under same nested CV.
    """
    N = len(df_train)
    strata = build_strata(df_train, cfg)
    outer_skf = StratifiedKFold(
        n_splits=cfg.outer_folds, shuffle=True,
        random_state=cfg.random_seed)

    oof_preds = {h: np.zeros(N) for h in cfg.main_horizons}
    oof_risk = np.zeros(N)

    surv_feats = feature_sets["survival_full"]

    for ofold, (otr_idx, ote_idx) in enumerate(
            outer_skf.split(df_train, strata)):
        df_tr = df_train.iloc[otr_idx]
        df_te = df_train.iloc[ote_idx]

        for h in cfg.main_horizons:
            elig = df_tr[f"eligible_{h}h"].astype(bool)
            dtrain_df = df_tr.loc[elig]
            y = dtrain_df[f"hit_by_{h}h"].values.astype(int)

            if len(dtrain_df) < 10 or np.unique(y).size < 2:
                oof_preds[h][ote_idx] = float(np.mean(y)) if len(y) > 0 else 0.5
                continue

            feats = [f for f in surv_feats if f in dtrain_df.columns]
            Xtr = dtrain_df[feats].fillna(0).astype(float).values
            Xte = df_te[feats].fillna(0).astype(float).values

            dtrain = xgb.DMatrix(Xtr, label=y)
            dtest = xgb.DMatrix(Xte)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 3,
                "learning_rate": 0.05,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "lambda": 3.0,
                "verbosity": 0,
                "seed": cfg.random_seed + h + ofold * 100,
            }
            model = xgb.train(params, dtrain, num_boost_round=200)
            oof_preds[h][ote_idx] = clip_probs(model.predict(dtest))

        risk = sum(oof_preds[h][ote_idx] * (4 - i)
                   for i, h in enumerate(cfg.main_horizons))
        oof_risk[ote_idx] = risk

    # Apply monotone fix
    oof_preds = monotonic_fix(oof_preds, cfg.main_horizons)
    return oof_preds, oof_risk


# ── Random Survival Forest Baseline ──
def rsf_baseline(df_train, feature_sets, cfg):
    """
    Random Survival Forest baseline using scikit-survival.
    Falls back to AFT-only if sksurv not available.
    """
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    except ImportError:
        print("    WARNING: scikit-survival not installed. "
              "Falling back to AFT-only for RSF baseline.")
        return None, None

    N = len(df_train)
    strata = build_strata(df_train, cfg)
    outer_skf = StratifiedKFold(
        n_splits=cfg.outer_folds, shuffle=True,
        random_state=cfg.random_seed)

    oof_preds = {h: np.zeros(N) for h in cfg.main_horizons}
    oof_risk = np.zeros(N)
    surv_feats = feature_sets["survival_full"]

    for ofold, (otr_idx, ote_idx) in enumerate(
            outer_skf.split(df_train, strata)):
        df_tr = df_train.iloc[otr_idx]
        df_te = df_train.iloc[ote_idx]

        feats = [f for f in surv_feats if f in df_tr.columns]
        Xtr = df_tr[feats].fillna(0).astype(float).values
        Xte = df_te[feats].fillna(0).astype(float).values

        y_time = df_tr[cfg.target_time].values.astype(float)
        y_event = df_tr[cfg.target_event].values.astype(bool)
        y_surv = Surv.from_arrays(y_event, y_time)

        rsf = RandomSurvivalForest(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=cfg.random_seed + ofold,
            n_jobs=-1)
        rsf.fit(Xtr, y_surv)

        # Extract CDF at each horizon
        surv_fn = rsf.predict_survival_function(Xte)
        for h in cfg.main_horizons:
            probs = []
            for fn in surv_fn:
                # CDF = 1 - S(h)
                try:
                    s_val = fn(h)
                except:
                    s_val = fn.y[-1] if h > fn.x[-1] else fn.y[0]
                probs.append(1.0 - float(s_val))
            oof_preds[h][ote_idx] = clip_probs(np.array(probs))

        # Risk score from survival function
        risk_scores = rsf.predict(Xte)  # cumulative hazard: higher = more risk
        oof_risk[ote_idx] = risk_scores

    oof_preds = monotonic_fix(oof_preds, cfg.main_horizons)
    return oof_preds, oof_risk


# ── Pre-fusion monotonicity violation counting ──
def count_prefusion_violations(oof_preds_raw, horizons=(12, 24, 48)):
    """
    Count monotonicity violations BEFORE monotone fusion is applied.
    Uses the actual stacked+fused probabilities before monotonic_fix().
    Returns dict with violation counts and mean adjustment magnitudes.
    """
    results = {}
    sorted_h = sorted(horizons)
    for i in range(len(sorted_h) - 1):
        h1, h2 = sorted_h[i], sorted_h[i + 1]
        p1 = np.asarray(oof_preds_raw[h1])
        p2 = np.asarray(oof_preds_raw[h2])
        # Violation: P(T<=h1) > P(T<=h2), i.e. shorter horizon has higher prob
        violation_mask = p1 > p2 + 1e-6
        n_violations = int(violation_mask.sum())
        adjustments = np.maximum(p1 - p2, 0)
        results[f"violations_{h1}_{h2}"] = n_violations
        results[f"violation_rate_{h1}_{h2}"] = f"{n_violations}/{len(p1)} ({100*n_violations/len(p1):.1f}%)"
        if n_violations > 0:
            results[f"mean_adjustment_{h1}_{h2}"] = float(np.mean(adjustments[violation_mask]))
            results[f"max_adjustment_{h1}_{h2}"] = float(np.max(adjustments))
        else:
            results[f"mean_adjustment_{h1}_{h2}"] = 0.0
            results[f"max_adjustment_{h1}_{h2}"] = 0.0
    return results


# ================================================================
# §10  Paired Bootstrap on Δ-Metric  (THE KEY FIX)
# ================================================================
def paired_bootstrap_delta(
        y_dict, elig_dict, pair_i, pair_j,
        preds_full, risk_full,
        preds_ablated, risk_ablated,
        horizons=(12, 24, 48),
        n_boot=1000, seed=42,
        y_time=None, y_event=None,
) -> Dict:
    """
    [R7] Paired bootstrap with BOTH one-sided and two-sided p-values.

    Returns: {metric: (mean_delta, ci_lo, ci_hi, p_one_sided, p_two_sided)}
    """
    rng = np.random.RandomState(seed)
    n = len(risk_full)

    delta_records = {f"delta_brier_{h}h": [] for h in horizons}
    delta_records["delta_c_index"] = []
    delta_records["delta_ibs"] = []
    delta_records["delta_c_index_uno"] = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        idx_set = set(idx.tolist())

        vp = [(pi, pj) for pi, pj in zip(pair_i, pair_j)
              if pi in idx_set and pj in idx_set]
        if len(vp) == 0:
            continue
        vpi = np.array([p[0] for p in vp])
        vpj = np.array([p[1] for p in vp])
        ci_full = fast_concordance(risk_full, vpi, vpj)
        ci_abl = fast_concordance(risk_ablated, vpi, vpj)
        delta_records["delta_c_index"].append(ci_full - ci_abl)

        # [R1] Uno's C delta
        if y_time is not None and y_event is not None:
            try:
                ci_uno_f = unos_c_index(y_time[idx], y_event[idx],
                                        risk_full[idx], tau=48.0)
                ci_uno_a = unos_c_index(y_time[idx], y_event[idx],
                                        risk_ablated[idx], tau=48.0)
                delta_records["delta_c_index_uno"].append(ci_uno_f - ci_uno_a)
            except Exception:
                pass

        brier_deltas = []
        for h in horizons:
            m = elig_dict[h][idx]
            yt = y_dict[h][idx][m]
            yp_f = preds_full[h][idx][m]
            yp_a = preds_ablated[h][idx][m]
            if len(yt) == 0:
                continue
            bs_f = np.mean((yt - yp_f) ** 2)
            bs_a = np.mean((yt - yp_a) ** 2)
            delta_records[f"delta_brier_{h}h"].append(bs_a - bs_f)
            brier_deltas.append(bs_a - bs_f)

        if brier_deltas:
            delta_records["delta_ibs"].append(np.mean(brier_deltas))

    # [R7] Summarize with both p-values
    results = {}
    for key, deltas in delta_records.items():
        deltas = np.array(deltas)
        if len(deltas) == 0:
            results[key] = (0.0, 0.0, 0.0, 0.5, 1.0)
            continue
        mean_d = np.mean(deltas)
        ci_lo = np.percentile(deltas, 2.5)
        ci_hi = np.percentile(deltas, 97.5)
        p_one = float(np.mean(deltas <= 0))
        p_two = min(2.0 * min(p_one, 1.0 - p_one), 1.0)
        results[key] = (mean_d, ci_lo, ci_hi, p_one, p_two)

    return results


# ================================================================
# §11  Ablation Framework
# ================================================================
FULL_MODEL = VariantConfig(
    name="full_model", label="Full Model",
    use_gate=True, use_direct=True, use_ipcw=True,
    use_simple=True, use_cox=True, survival_features="full")

# TRUE ablation configs — each switch genuinely disables the module
ABLATION_CONFIGS = [
    # Layered build-up (recommended for paper)
    VariantConfig(name="aft_only", label="AFT Only",
                  use_gate=False, use_direct=False, use_ipcw=False,
                  use_simple=False, use_cox=False),
    VariantConfig(name="aft_plus_direct", label="AFT + Direct",
                  use_gate=False, use_direct=True, use_ipcw=False,
                  use_simple=False, use_cox=False),
    VariantConfig(name="aft_direct_cox", label="AFT + Direct + Cox",
                  use_gate=False, use_direct=True, use_ipcw=False,
                  use_simple=False, use_cox=True),
    VariantConfig(name="aft_direct_gate", label="AFT + Direct + Gate",
                  use_gate=True, use_direct=True, use_ipcw=False,
                  use_simple=False, use_cox=False),
    # Component removal (w.r.t. full model)
    VariantConfig(name="no_gate", label="w/o Gate Prior",
                  use_gate=False, use_direct=True, use_ipcw=True,
                  use_simple=True, use_cox=True),
    VariantConfig(name="no_cox", label="w/o Cox Branch",
                  use_gate=True, use_direct=True, use_ipcw=True,
                  use_simple=True, use_cox=False),
    VariantConfig(name="no_ipcw", label="w/o IPCW",
                  use_gate=True, use_direct=True, use_ipcw=False,
                  use_simple=True, use_cox=True),
    VariantConfig(name="no_simple", label="w/o Simple Distance",
                  use_gate=True, use_direct=True, use_ipcw=True,
                  use_simple=False, use_cox=True),
    VariantConfig(name="no_direct", label="w/o Direct Heads",
                  use_gate=True, use_direct=False, use_ipcw=True,
                  use_simple=True, use_cox=True),
    VariantConfig(name="thin_features", label="Thin Features Only",
                  use_gate=False, use_direct=True, use_ipcw=True,
                  use_simple=True, use_cox=True,
                  survival_features="thin"),
]

# ── Practical lean variants (strategic "subtraction experiments") ──
PRACTICAL_VARIANTS_FULL = [
    VariantConfig(name="aft_cox", label="AFT + Cox",
                  use_gate=False, use_direct=False, use_ipcw=False,
                  use_simple=False, use_cox=True, survival_features="full"),
    VariantConfig(name="aft_cox_simple", label="AFT + Cox + Simple",
                  use_gate=False, use_direct=False, use_ipcw=False,
                  use_simple=True, use_cox=True, survival_features="full"),
    VariantConfig(name="aft_simple", label="AFT + Simple",
                  use_gate=False, use_direct=False, use_ipcw=False,
                  use_simple=True, use_cox=False, survival_features="full"),
]
PRACTICAL_VARIANTS_THIN = [
    VariantConfig(name="aft_cox_thin", label="AFT + Cox (thin)",
                  use_gate=False, use_direct=False, use_ipcw=False,
                  use_simple=False, use_cox=True, survival_features="thin"),
    VariantConfig(name="aft_cox_simple_thin", label="AFT + Cox + Simple (thin)",
                  use_gate=False, use_direct=False, use_ipcw=False,
                  use_simple=True, use_cox=True, survival_features="thin"),
    VariantConfig(name="aft_simple_thin", label="AFT + Simple (thin)",
                  use_gate=False, use_direct=False, use_ipcw=False,
                  use_simple=True, use_cox=False, survival_features="thin"),
    VariantConfig(name="aft_only_thin", label="AFT Only (thin)",
                  use_gate=False, use_direct=False, use_ipcw=False,
                  use_simple=False, use_cox=False, survival_features="thin"),
]
ALL_PRACTICAL_VARIANTS = PRACTICAL_VARIANTS_FULL + PRACTICAL_VARIANTS_THIN


def run_practical_variants(
        df_train: pd.DataFrame,
        feature_sets: Dict,
        full_result: Dict,
        cfg: PipelineConfig,
) -> pd.DataFrame:
    """Run all practical lean variants with paired bootstrap Δ-tests."""
    payload = _build_payload(df_train, cfg)
    y_time = df_train[cfg.target_time].values
    y_event = df_train[cfg.target_event].values
    full_metrics = compute_all_metrics(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        full_result["oof_preds"],
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        full_result["oof_risk"],
        payload["pair_i"], payload["pair_j"],
        y_time=y_time, y_event=y_event,
        horizons=cfg.main_horizons)

    rows = [{"config": "Full Model", **full_metrics,
             "delta_c_index": 0.0, "p_c_index": np.nan,
             "delta_ibs": 0.0, "p_ibs": np.nan}]

    for pv_cfg in ALL_PRACTICAL_VARIANTS:
        print(f"\n  >> Practical variant: {pv_cfg.label}")
        pv_result = nested_evaluate(df_train, feature_sets, pv_cfg, cfg)
        pv_metrics = compute_all_metrics(
            {h: payload[f"y{h}"] for h in cfg.main_horizons},
            pv_result["oof_preds"],
            {h: payload[f"elig{h}"] for h in cfg.main_horizons},
            pv_result["oof_risk"],
            payload["pair_i"], payload["pair_j"],
            y_time=y_time, y_event=y_event,
            horizons=cfg.main_horizons)

        boot = paired_bootstrap_delta(
            y_dict={h: payload[f"y{h}"] for h in cfg.main_horizons},
            elig_dict={h: payload[f"elig{h}"] for h in cfg.main_horizons},
            pair_i=payload["pair_i"], pair_j=payload["pair_j"],
            preds_full=full_result["oof_preds"],
            risk_full=full_result["oof_risk"],
            preds_ablated=pv_result["oof_preds"],
            risk_ablated=pv_result["oof_risk"],
            horizons=cfg.main_horizons,
            n_boot=cfg.n_bootstrap, seed=cfg.random_seed,
            y_time=y_time, y_event=y_event)

        dc = boot["delta_c_index"]
        di = boot["delta_ibs"]
        sig_c = "***" if dc[3] < 0.001 else "**" if dc[3] < 0.01 else \
            "*" if dc[3] < 0.05 else "n.s."
        sig_i = "***" if di[3] < 0.001 else "**" if di[3] < 0.01 else \
            "*" if di[3] < 0.05 else "n.s."

        print(f"    C-index={pv_metrics['c_index']:.4f}  IBS={pv_metrics['ibs']:.4f}")
        print(f"    ΔC-index: {dc[0]:+.4f} [{dc[1]:+.4f}, {dc[2]:+.4f}] "
              f"p={dc[3]:.3f} ({sig_c})")
        print(f"    ΔIBS:     {di[0]:+.4f} [{di[1]:+.4f}, {di[2]:+.4f}] "
              f"p={di[3]:.3f} ({sig_i})")

        row = {"config": pv_cfg.label, **pv_metrics,
               "delta_c_index": dc[0], "p_c_index": dc[3],
               "delta_ibs": di[0], "p_ibs": di[3]}
        for h in cfg.main_horizons:
            d = boot[f"delta_brier_{h}h"]
            row[f"delta_brier_{h}h"] = d[0]
            row[f"p_brier_{h}h"] = d[3]
        rows.append(row)

    return pd.DataFrame(rows)


# ================================================================
# §11b  Multi-Seed Stability Check
# ================================================================
def multi_seed_stability(
        df_train: pd.DataFrame,
        feature_sets: Dict,
        cfg: PipelineConfig,
        n_repeats: int = 5,
        seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Re-run nested CV under different random seeds for key models.
    Tests whether relative rankings (Full > lean > RSF etc.) are stable
    or an artefact of a particular train/test split.

    Returns a long-format DataFrame: (seed, model, c_index, ibs, brier_12/24/48).
    """
    if seeds is None:
        seeds = [42, 123, 314, 2024, 7777][:n_repeats]

    # Models to test
    key_models = [
        ("Full Model", FULL_MODEL),
        ("AFT + Cox + Simple", VariantConfig(
            name="aft_cox_simple_stab", label="AFT + Cox + Simple",
            use_gate=False, use_direct=False, use_ipcw=False,
            use_simple=True, use_cox=True, survival_features="full")),
        ("AFT + Cox", VariantConfig(
            name="aft_cox_stab", label="AFT + Cox",
            use_gate=False, use_direct=False, use_ipcw=False,
            use_simple=False, use_cox=True, survival_features="full")),
    ]

    rows = []
    for si, seed in enumerate(seeds):
        print(f"\n  ── Seed {seed} ({si+1}/{len(seeds)}) ──")
        cfg_s = PipelineConfig(
            data_dir=cfg.data_dir, output_dir=cfg.output_dir,
            outer_folds=cfg.outer_folds, inner_folds=cfg.inner_folds,
            random_seed=seed,
            main_horizons=cfg.main_horizons,
            supplement_horizons=cfg.supplement_horizons,
            n_bootstrap=200,  # reduced for speed
            stacking_lambda=cfg.stacking_lambda,
        )

        payload = _build_payload(df_train, cfg_s)

        for model_name, variant in key_models:
            print(f"    {model_name} (seed={seed})...")
            result = nested_evaluate(df_train, feature_sets, variant, cfg_s)
            metrics = compute_all_metrics(
                {h: payload[f"y{h}"] for h in cfg_s.main_horizons},
                result["oof_preds"],
                {h: payload[f"elig{h}"] for h in cfg_s.main_horizons},
                result["oof_risk"],
                payload["pair_i"], payload["pair_j"],
                y_time=df_train[cfg_s.target_time].values,
                y_event=df_train[cfg_s.target_event].values,
                horizons=cfg_s.main_horizons)
            row = {"seed": seed, "model": model_name, **metrics}
            rows.append(row)

        # Per-horizon XGBoost
        print(f"    Per-horizon XGBoost (seed={seed})...")
        xgb_preds, xgb_risk = xgb_per_horizon_baseline(df_train, feature_sets, cfg_s)
        xgb_m = compute_all_metrics(
            {h: payload[f"y{h}"] for h in cfg_s.main_horizons},
            xgb_preds,
            {h: payload[f"elig{h}"] for h in cfg_s.main_horizons},
            xgb_risk, payload["pair_i"], payload["pair_j"],
            y_time=df_train[cfg_s.target_time].values,
            y_event=df_train[cfg_s.target_event].values,
            horizons=cfg_s.main_horizons)
        rows.append({"seed": seed, "model": "Per-horizon XGBoost", **xgb_m})

        # RSF
        print(f"    RSF (seed={seed})...")
        rsf_preds, rsf_risk = rsf_baseline(df_train, feature_sets, cfg_s)
        if rsf_preds is not None:
            rsf_m = compute_all_metrics(
                {h: payload[f"y{h}"] for h in cfg_s.main_horizons},
                rsf_preds,
                {h: payload[f"elig{h}"] for h in cfg_s.main_horizons},
                rsf_risk, payload["pair_i"], payload["pair_j"],
                y_time=df_train[cfg_s.target_time].values,
                y_event=df_train[cfg_s.target_event].values,
                horizons=cfg_s.main_horizons)
            rows.append({"seed": seed, "model": "RSF", **rsf_m})

    df = pd.DataFrame(rows)

    # Print summary
    print("\n  ── Multi-seed stability summary ──")
    summary = df.groupby("model").agg(
        c_index_mean=("c_index", "mean"),
        c_index_std=("c_index", "std"),
        ibs_mean=("ibs", "mean"),
        ibs_std=("ibs", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    print(summary.to_string(index=False))

    # Print rank consistency
    print("\n  ── Per-seed rankings (C-index) ──")
    for seed in seeds:
        seed_df = df[df["seed"] == seed].sort_values("c_index", ascending=False)
        ranking = " > ".join(
            f"{r['model']}({r['c_index']:.4f})"
            for _, r in seed_df.iterrows())
        print(f"    seed={seed}: {ranking}")

    return df


# ================================================================
# §11c  Grouped Cross-Validation (incident-level)
# ================================================================
def detect_group_column(df: pd.DataFrame, cfg: PipelineConfig) -> Optional[str]:
    """
    [R10] Enhanced detection of incident-level grouping column.
    Checks explicit columns first, then attempts spatial clustering fallback.
    """
    candidate_patterns = ["incident", "fire_id", "fire_event", "fire_name",
                          "group", "cluster", "region", "zone_group",
                          "wildfire_id", "event_name"]
    for col in df.columns:
        col_lower = col.lower()
        for pat in candidate_patterns:
            if pat in col_lower:
                n_unique = df[col].nunique()
                if cfg.outer_folds < n_unique < len(df):
                    return col

    # [R10] Attempt spatial proximity clustering as proxy grouping
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower()]
    if lat_cols and lon_cols:
        try:
            from sklearn.cluster import DBSCAN
            coords = df[[lat_cols[0], lon_cols[0]]].fillna(0).values
            clustering = DBSCAN(eps=0.1, min_samples=2).fit(coords)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if cfg.outer_folds < n_clusters < len(df) * 0.8:
                df["_proxy_fire_group"] = labels
                print(f"  [R10] Created proxy spatial groups: {n_clusters} clusters")
                return "_proxy_fire_group"
        except Exception:
            pass
    return None


def grouped_nested_evaluate(
        df_train: pd.DataFrame,
        feature_sets: Dict,
        variant: VariantConfig,
        cfg: PipelineConfig,
        group_col: str,
) -> Dict:
    """
    Outer GroupKFold nested cross-validation.
    Ensures all samples from the same incident/group are in the same fold.
    Inner folds still use StratifiedKFold (within outer-train).
    """
    from sklearn.model_selection import GroupKFold

    N = len(df_train)
    groups = df_train[group_col].values
    outer_gkf = GroupKFold(n_splits=cfg.outer_folds)

    oof_preds = {h: np.zeros(N) for h in cfg.all_horizons}
    oof_preds_raw = {h: np.zeros(N) for h in cfg.all_horizons}
    oof_risk = np.zeros(N)
    oof_components = {}
    fold_configs = []

    print(f"\n  [{variant.name}] Grouped {cfg.outer_folds}-fold evaluation "
          f"(group_col={group_col}, {df_train[group_col].nunique()} groups)")

    for ofold, (otr_idx, ote_idx) in enumerate(
            outer_gkf.split(df_train, groups=groups)):
        n_groups_tr = df_train.iloc[otr_idx][group_col].nunique()
        n_groups_te = df_train.iloc[ote_idx][group_col].nunique()
        print(f"    Outer fold {ofold + 1}/{cfg.outer_folds} "
              f"(train={len(otr_idx)} [{n_groups_tr} groups], "
              f"test={len(ote_idx)} [{n_groups_te} groups])")

        result = cross_fit_one_outer_fold(
            df_outer_train=df_train.iloc[otr_idx],
            df_outer_test=df_train.iloc[ote_idx],
            feature_sets=feature_sets,
            variant=variant,
            cfg=cfg,
            outer_seed=cfg.random_seed + ofold * 10000,
        )

        test_preds = apply_fusion(
            result["test_components"], result["fusion_config"], cfg)
        test_preds_raw = apply_fusion_raw(
            result["test_components"], result["fusion_config"], cfg)

        preds_for_risk = {h: test_preds[h] for h in cfg.main_horizons}
        test_risk = _compose_risk(
            preds_for_risk,
            result["test_components"]["surv_risk"],
            result["test_components"].get("cox_risk"),
            result["fusion_config"]["w_surv_rank"],
            result["fusion_config"]["w_cox_rank"])

        for h in cfg.all_horizons:
            oof_preds[h][ote_idx] = test_preds[h]
            oof_preds_raw[h][ote_idx] = test_preds_raw[h]
        oof_risk[ote_idx] = test_risk
        fold_configs.append(result["fusion_config"])

        for key, arr in result["test_components"].items():
            if key not in oof_components:
                oof_components[key] = np.zeros(N)
            oof_components[key][ote_idx] = arr

    return {
        "oof_preds": oof_preds,
        "oof_preds_raw": oof_preds_raw,
        "oof_risk": oof_risk,
        "oof_components": oof_components,
        "fold_configs": fold_configs,
    }


def run_grouped_cv_comparison(
        df_train: pd.DataFrame,
        feature_sets: Dict,
        cfg: PipelineConfig,
        group_col: str,
) -> pd.DataFrame:
    """
    Run key models under grouped CV and compare to standard stratified CV.
    """
    payload = _build_payload(df_train, cfg)
    models = [
        ("Full Model", FULL_MODEL),
        ("AFT + Cox + Simple", VariantConfig(
            name="aft_cox_simple_grp", label="AFT + Cox + Simple",
            use_gate=False, use_direct=False, use_ipcw=False,
            use_simple=True, use_cox=True, survival_features="full")),
        ("AFT + Cox", VariantConfig(
            name="aft_cox_grp", label="AFT + Cox",
            use_gate=False, use_direct=False, use_ipcw=False,
            use_simple=False, use_cox=True, survival_features="full")),
    ]

    rows = []
    for model_name, variant in models:
        print(f"\n  Grouped CV: {model_name}")
        result = grouped_nested_evaluate(
            df_train, feature_sets, variant, cfg, group_col)
        metrics = compute_all_metrics(
            {h: payload[f"y{h}"] for h in cfg.main_horizons},
            result["oof_preds"],
            {h: payload[f"elig{h}"] for h in cfg.main_horizons},
            result["oof_risk"],
            payload["pair_i"], payload["pair_j"],
            y_time=df_train[cfg.target_time].values,
            y_event=df_train[cfg.target_event].values,
            horizons=cfg.main_horizons)
        rows.append({"model": model_name, "cv_type": "grouped", **metrics})
        print(f"    C-index={metrics['c_index']:.4f}  IBS={metrics['ibs']:.4f}")

    return pd.DataFrame(rows)


# ================================================================
# §11d  [R6] Lambda Sensitivity Sweep
# ================================================================
def run_lambda_sweep(
        df_train: pd.DataFrame,
        feature_sets: Dict,
        cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    [R6] Sweep stacking regularisation λ over cfg.lambda_sweep values.
    Re-runs full model under each λ to verify metric stability.
    """
    payload = _build_payload(df_train, cfg)
    y_time = df_train[cfg.target_time].values
    y_event = df_train[cfg.target_event].values

    rows = []
    for lam in cfg.lambda_sweep:
        print(f"    λ = {lam} ...")
        cfg_lam = PipelineConfig(
            data_dir=cfg.data_dir, output_dir=cfg.output_dir,
            outer_folds=cfg.outer_folds, inner_folds=cfg.inner_folds,
            random_seed=cfg.random_seed,
            main_horizons=cfg.main_horizons,
            supplement_horizons=cfg.supplement_horizons,
            n_bootstrap=200,
            stacking_lambda=lam,
            beta_cal_min_n=cfg.beta_cal_min_n,
        )
        result = nested_evaluate(df_train, feature_sets, FULL_MODEL, cfg_lam)
        metrics = compute_all_metrics(
            {h: payload[f"y{h}"] for h in cfg.main_horizons},
            result["oof_preds"],
            {h: payload[f"elig{h}"] for h in cfg.main_horizons},
            result["oof_risk"],
            payload["pair_i"], payload["pair_j"],
            y_time=y_time, y_event=y_event,
            horizons=cfg.main_horizons)
        row = {"lambda": lam, **metrics}
        rows.append(row)
        print(f"      C-index={metrics['c_index']:.4f}  IBS={metrics['ibs']:.4f}")

    return pd.DataFrame(rows)


def run_ablation_study(
        df_train: pd.DataFrame,
        feature_sets: Dict,
        full_result: Dict,
        cfg: PipelineConfig,
) -> pd.DataFrame:
    """[R7] Run ablation with paired bootstrap Δ-metric tests + two-sided p-values."""
    payload = _build_payload(df_train, cfg)
    y_time = df_train[cfg.target_time].values
    y_event = df_train[cfg.target_event].values
    full_metrics = compute_all_metrics(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        full_result["oof_preds"],
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        full_result["oof_risk"],
        payload["pair_i"], payload["pair_j"],
        y_time=y_time, y_event=y_event,
        horizons=cfg.main_horizons)

    rows = [{"config": "Full Model", **full_metrics,
             "delta_c_index": 0.0, "p_c_index": np.nan,
             "p_c_index_two": np.nan,
             "delta_ibs": 0.0, "p_ibs": np.nan,
             "p_ibs_two": np.nan}]

    for abl_cfg in ABLATION_CONFIGS:
        print(f"\n  >> Ablation: {abl_cfg.label}")
        abl_result = nested_evaluate(df_train, feature_sets, abl_cfg, cfg)

        abl_metrics = compute_all_metrics(
            {h: payload[f"y{h}"] for h in cfg.main_horizons},
            abl_result["oof_preds"],
            {h: payload[f"elig{h}"] for h in cfg.main_horizons},
            abl_result["oof_risk"],
            payload["pair_i"], payload["pair_j"],
            y_time=y_time, y_event=y_event,
            horizons=cfg.main_horizons)

        # Paired bootstrap Δ-metric
        boot = paired_bootstrap_delta(
            y_dict={h: payload[f"y{h}"] for h in cfg.main_horizons},
            elig_dict={h: payload[f"elig{h}"] for h in cfg.main_horizons},
            pair_i=payload["pair_i"], pair_j=payload["pair_j"],
            preds_full=full_result["oof_preds"],
            risk_full=full_result["oof_risk"],
            preds_ablated=abl_result["oof_preds"],
            risk_ablated=abl_result["oof_risk"],
            horizons=cfg.main_horizons,
            n_boot=cfg.n_bootstrap, seed=cfg.random_seed,
            y_time=y_time, y_event=y_event)

        dc = boot["delta_c_index"]
        di = boot["delta_ibs"]
        # dc/di are 5-tuples: (mean, ci_lo, ci_hi, p_one, p_two)
        sig_c = "***" if dc[3] < 0.001 else "**" if dc[3] < 0.01 else \
            "*" if dc[3] < 0.05 else "n.s."
        sig_i = "***" if di[3] < 0.001 else "**" if di[3] < 0.01 else \
            "*" if di[3] < 0.05 else "n.s."

        print(f"    ΔC-index: {dc[0]:+.4f} [{dc[1]:+.4f}, {dc[2]:+.4f}] "
              f"p1={dc[3]:.3f} p2={dc[4]:.3f} ({sig_c})")
        print(f"    ΔIBS:     {di[0]:+.4f} [{di[1]:+.4f}, {di[2]:+.4f}] "
              f"p1={di[3]:.3f} p2={di[4]:.3f} ({sig_i})")

        row = {"config": abl_cfg.label, **abl_metrics,
               "delta_c_index": dc[0], "p_c_index": dc[3],
               "p_c_index_two": dc[4],
               "delta_ibs": di[0], "p_ibs": di[3],
               "p_ibs_two": di[4]}
        # Per-horizon Δ Brier
        for h in cfg.main_horizons:
            d = boot[f"delta_brier_{h}h"]
            row[f"delta_brier_{h}h"] = d[0]
            row[f"p_brier_{h}h"] = d[3]
            row[f"p_brier_{h}h_two"] = d[4]
        rows.append(row)

    return pd.DataFrame(rows)


# ================================================================
# §12  Bootstrap Confidence Intervals
# ================================================================
def bootstrap_ci(y_dict, pred_dict, elig_dict, risk,
                 pair_i, pair_j, horizons=(12, 24, 48),
                 n_boot=1000, seed=42,
                 y_time=None, y_event=None):
    """[R1][R2] Bootstrap CIs including Uno's C and continuous IBS."""
    rng = np.random.RandomState(seed)
    n = len(risk)
    records = {k: [] for k in ["c_index", "c_index_uno",
                                "ibs", "ibs_continuous"] +
               [f"brier_{h}h" for h in horizons]}

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        idx_set = set(idx.tolist())
        vp = [(pi, pj) for pi, pj in zip(pair_i, pair_j)
              if pi in idx_set and pj in idx_set]
        if not vp:
            continue

        ci = fast_concordance(risk,
                              np.array([p[0] for p in vp]),
                              np.array([p[1] for p in vp]))
        records["c_index"].append(ci)

        # [R1] Uno's C
        if y_time is not None and y_event is not None:
            try:
                records["c_index_uno"].append(
                    unos_c_index(y_time[idx], y_event[idx],
                                risk[idx], tau=48.0))
            except Exception:
                pass

        briers = []
        for h in horizons:
            m = elig_dict[h][idx]
            yt, yp = y_dict[h][idx][m], pred_dict[h][idx][m]
            bs = np.mean((yt - yp) ** 2) if len(yt) > 0 else 0.25
            records[f"brier_{h}h"].append(bs)
            briers.append(bs)
        records["ibs"].append(np.mean(briers))

        # [R2] Continuous IBS
        if y_time is not None and y_event is not None:
            try:
                pred_cdf_boot = {float(h): pred_dict[h][idx] for h in horizons}
                records["ibs_continuous"].append(
                    integrated_brier_score(
                        y_time[idx], y_event[idx], pred_cdf_boot,
                        [float(h) for h in horizons]))
            except Exception:
                pass

    return {k: np.array(v) for k, v in records.items()}


# ================================================================
# §13  Plotting — MDPI Fire Journal Figures
# ================================================================
# Constants & Style
# ================================================================
MDPI_DPI = 600
SINGLE_COL = 3.346   # inches (85 mm)
DOUBLE_COL = 6.693   # inches (170 mm)
ONE_AND_HALF = 5.02   # inches (127.5 mm)


def set_paper_style():
    """MDPI Fire: Arial/Helvetica ≥8pt, white bg, no gridlines."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": MDPI_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "patch.linewidth": 0.6,
        "mathtext.default": "regular",
    })


# Tol's colorblind-safe qualitative palette (muted for print)
COLORS = {
    "12h": "#CC4466",
    "24h": "#3366AA",
    "48h": "#117733",
    "72h": "#BBAA33",
    "primary": "#2B2B2B",
    "secondary": "#888888",
    "light_gray": "#CCCCCC",
    "accent": "#CC3311",
    "positive": "#117733",
    "negative": "#CC4466",
    "ci_fill": "#3366AA",
    "module_a": "#C8E6C9",   "module_a_edge": "#2E7D32",
    "module_b": "#BBDEFB",   "module_b_edge": "#1565C0",
    "module_c": "#FFE0B2",   "module_c_edge": "#E65100",
    "module_d": "#F8BBD0",   "module_d_edge": "#C62828",
    "input_bg": "#F5F5F5",
    "output_bg": "#FFF8E1",
}
HORIZON_LABELS = {12: "12 h", 24: "24 h", 48: "48 h", 72: "72 h"}
HORIZON_COLORS = {12: COLORS["12h"], 24: COLORS["24h"],
                  48: COLORS["48h"], 72: COLORS["72h"]}


def _save_mdpi(fig, save_path):
    """Save as TIFF (600 DPI LZW), PNG (300), PDF, SVG."""
    if save_path is None:
        plt.close(fig)
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [
        (".tiff", dict(dpi=MDPI_DPI, format="tiff",
                       pil_kwargs={"compression": "tiff_lzw"})),
        (".png", dict(dpi=300, format="png")),
        (".svg", dict(format="svg")),
        (".pdf", dict(dpi=MDPI_DPI)),
    ]:
        fig.savefig(str(p.with_suffix(suffix)),
                    bbox_inches="tight", pad_inches=0.05, **kwargs)
    plt.close(fig)


# ================================================================
# Helper functions for framework diagram
# ================================================================
def _add_box(ax, x, y, w, h, text, fc, ec, fontsize=7.5,
             lw=1.0, text_color=None, style="round,pad=0.12",
             bold=True, linespacing=1.3, zorder=2):
    """Add a labeled rounded box to axes."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=style, fc=fc, ec=ec,
        lw=lw, zorder=zorder))
    fw = "bold" if bold else "normal"
    tc = text_color or ec
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fw, color=tc,
            linespacing=linespacing, zorder=zorder + 1)


def _arrow(ax, x1, y1, x2, y2, color="#555555", lw=0.8,
           style="-|>", ls="-", connectionstyle=None):
    """Draw an arrow between two points."""
    props = dict(arrowstyle=style, color=color, lw=lw, ls=ls)
    if connectionstyle:
        props["connectionstyle"] = connectionstyle
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=props)


# ═══════════════════════════════════════════════════════════════
# Figure 0: Prediction Timeline  ★ NEW (v2 — vertical stacking)
# ═══════════════════════════════════════════════════════════════
def plot_timeline(save_path=None):
    """
    Professional prediction timeline figure for MDPI Fire journal.
    Shows: ignition → feature window → prediction time → horizons,
    with event and censoring definitions and schematic examples.
    v2: definition boxes stacked vertically to avoid text overlap.
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))
    ax.set_xlim(-1.0, 15.5)
    ax.set_ylim(-3.0, 4.5)
    ax.axis("off")

    # ── Row 1: Main timeline (y ~ 2.8) ──
    y_main = 2.8
    ax.annotate("", xy=(14.8, y_main), xytext=(-0.3, y_main),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.0, mutation_scale=12))
    ax.text(15.0, y_main, "time", fontsize=7, color="#777777",
            va="center", ha="left", style="italic")

    # Feature window segment
    fw_x0, fw_x1 = 1.0, 4.5
    ax.add_patch(FancyBboxPatch(
        (fw_x0, y_main - 0.40), fw_x1 - fw_x0, 0.80,
        boxstyle="round,pad=0.08", fc="#E3F2FD", ec="#1565C0",
        lw=1.0, zorder=3, alpha=0.85))
    ax.text((fw_x0 + fw_x1) / 2, y_main,
            "Feature window\n(0\u20135 h)",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#1565C0", zorder=4)

    # Horizon colour bands
    tp_x = fw_x1
    h12_x, h24_x, h48_x = 7.2, 9.8, 13.8
    for x0, x1, clr in [
        (tp_x,  h12_x, COLORS["12h"]),
        (h12_x, h24_x, COLORS["24h"]),
        (h24_x, h48_x, COLORS["48h"]),
    ]:
        ax.fill_between([x0, x1], y_main - 0.40, y_main + 0.40,
                        color=clr, alpha=0.10, zorder=2)

    # Horizon dashed ticks + labels
    for x, label, clr in [
        (h12_x, "12 h", COLORS["12h"]),
        (h24_x, "24 h", COLORS["24h"]),
        (h48_x, "48 h", COLORS["48h"]),
    ]:
        ax.plot([x, x], [y_main - 0.52, y_main + 0.52],
                color=clr, lw=1.2, ls="--", zorder=5)
        ax.text(x, y_main + 0.68, label, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=clr, zorder=5)

    # t0 marker
    ax.plot(1.0, y_main, "o", color=COLORS["accent"], ms=6, zorder=6,
            mec="white", mew=0.8)
    ax.text(1.0, y_main - 0.72, r"$t_0$" + "\n(ignition)",
            ha="center", va="top", fontsize=7, color=COLORS["accent"],
            fontweight="bold", linespacing=1.3)

    # tp marker
    ax.plot(tp_x, y_main, "D", color="#E65100", ms=5.5, zorder=6,
            mec="white", mew=0.8)
    ax.text(tp_x, y_main - 0.72,
            r"$t_p = t_0 + 5$ h" + "\n(prediction time)",
            ha="center", va="top", fontsize=6.5, color="#E65100",
            fontweight="bold", linespacing=1.3)

    # Horizon span annotation
    ax.annotate("", xy=(h48_x, y_main + 1.20),
                xytext=(tp_x, y_main + 1.20),
                arrowprops=dict(arrowstyle="<->", color="#666666", lw=0.7))
    ax.text((tp_x + h48_x) / 2, y_main + 1.38,
            r"Horizons measured from $t_p$",
            ha="center", va="bottom", fontsize=6.5,
            color="#555555", style="italic")

    # ── Row 2: Definition boxes (stacked vertically to avoid overlap) ──
    box_w, box_x = 14.2, 0.0

    y_ev = 0.15
    ax.add_patch(FancyBboxPatch(
        (box_x, y_ev), box_w, 0.70,
        boxstyle="round,pad=0.10", fc="#FFF8E1", ec="#F57F17",
        lw=0.8, zorder=3, alpha=0.9))
    ax.text(box_x + 0.20, y_ev + 0.50, "Event",
            fontsize=7, fontweight="bold", color="#F57F17",
            va="center", zorder=4)
    ax.text(box_x + 1.60, y_ev + 0.50,
            r"— fire perimeter enters 5 km of zone centroid within $t_p + h$",
            fontsize=6.5, color="#555555", va="center", zorder=4)

    y_cen = -0.75
    ax.add_patch(FancyBboxPatch(
        (box_x, y_cen), box_w, 0.70,
        boxstyle="round,pad=0.10", fc="#F3E5F5", ec="#7B1FA2",
        lw=0.8, zorder=3, alpha=0.9))
    ax.text(box_x + 0.20, y_cen + 0.50, "Right censoring",
            fontsize=7, fontweight="bold", color="#7B1FA2",
            va="center", zorder=4)
    ax.text(box_x + 3.50, y_cen + 0.50,
            r"— monitoring ends before event at time $T_c \geq t_p$",
            fontsize=6.5, color="#555555", va="center", zorder=4)

    # ── Row 3: Schematic examples ──
    y_ex_ev, y_ex_cen = -1.75, -2.45

    ax.text(0.0, y_ex_ev, "Event case:", fontsize=6.5, color="#F57F17",
            ha="right", va="center", fontweight="bold")
    ax.plot([0.3, 6.0], [y_ex_ev, y_ex_ev], color="#F57F17",
            lw=1.8, zorder=4, solid_capstyle="round")
    ax.plot(6.0, y_ex_ev, "x", color="#F57F17", ms=8, mew=2.0, zorder=5)
    ax.text(6.5, y_ex_ev, r"$T \leq h$  (fire arrives)",
            fontsize=6.5, color="#F57F17", va="center")

    ax.text(0.0, y_ex_cen, "Censored:", fontsize=6.5, color="#7B1FA2",
            ha="right", va="center", fontweight="bold")
    ax.plot([0.3, 8.5], [y_ex_cen, y_ex_cen], color="#7B1FA2",
            lw=1.8, zorder=4, solid_capstyle="round")
    ax.plot(8.5, y_ex_cen, "|", color="#7B1FA2", ms=12, mew=2.0, zorder=5)
    ax.text(9.0, y_ex_cen, r"$T_c \geq t_p$  (monitoring ends)",
            fontsize=6.5, color="#7B1FA2", va="center")

    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure 1: Framework Schematic
# ═══════════════════════════════════════════════════════════════
def plot_framework(save_path=None):
    """
    Professional multi-layer framework diagram:
    - Clear vertical hierarchy (Input → Modules A/B/C → Fusion → Output)
    - Internal sub-model details and mathematical notation
    - Gate prior dashed arrows with labeled paths
    - Cross-fitting / outer-loop annotations
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5.5))
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-1.2, 9.8)
    ax.axis("off")

    # ── Layer 0: Input Features ──
    _add_box(ax, 3.0, 8.4, 7.0, 1.0,
             "Input Features\n"
             "Distance · Speed · Growth · Calendar · Regime  (34 features)",
             fc=COLORS["input_bg"], ec="#888888", fontsize=8,
             lw=0.8, text_color="#444444", bold=False,
             style="round,pad=0.15")

    # ── Layer 1: Three parallel modules ──
    # Module A — Gate Prior
    _add_box(ax, 0.0, 4.6, 3.2, 2.8,
             "Module A\nGate Prior\n─────────\n"
             "Logistic Regression\n"
             r"($L_2 = 0.5$)" "\n"
             "OOF prob → soft prior",
             fc=COLORS["module_a"], ec=COLORS["module_a_edge"],
             fontsize=6.5, lw=1.2, linespacing=1.3)

    # Module B — Dual Survival Backbone
    _add_box(ax, 3.8, 4.6, 4.8, 2.8,
             "Module B: Dual Survival Backbone\n"
             "─────────────────────────\n"
             "XGBoost AFT Ensemble (×3)\n"
             "Normal · Logistic · Extreme Value\n"
             "─────────\n"
             "Cox PH Ranking (×3)\n"
             "Risk score → rank normalise → [0, 1]",
             fc=COLORS["module_b"], ec=COLORS["module_b_edge"],
             fontsize=6.0, lw=1.2, linespacing=1.25)

    # Module C — Calibrated Heads
    _add_box(ax, 9.2, 4.6, 4.0, 2.8,
             "Module C: Calibrated Heads\n"
             "──────────────────\n"
             "Per horizon (12 / 24 / 48 h):\n"
             "  LR · GBC · HGB\n"
             "  + Beta calibration\n"
             "──────\n"
             "Optional: IPCW · Simple dist.",
             fc=COLORS["module_c"], ec=COLORS["module_c_edge"],
             fontsize=6.0, lw=1.2, linespacing=1.25)

    # ── Layer 2: Fusion ──
    _add_box(ax, 1.2, 1.4, 10.5, 1.6,
             "Module D: Uniform-Shrinkage Simplex Stacking  +  Monotone Fusion\n"
             r"min Brier  s.t. $\mathbf{w} \geq 0$,  regularise → $1/K$"
             "  ·  enforce  P(12 h) ≤ P(24 h) ≤ P(48 h)",
             fc=COLORS["module_d"], ec=COLORS["module_d_edge"],
             fontsize=6.5, lw=1.2, linespacing=1.4)

    # ── Layer 3: Output ──
    _add_box(ax, 3.0, -0.8, 7.0, 0.85,
             "Output:  P(T ≤ 12 h)  ·  P(T ≤ 24 h)  ·  P(T ≤ 48 h)  +  Risk Signal",
             fc=COLORS["output_bg"], ec=COLORS["accent"],
             fontsize=7.5, lw=1.0, text_color=COLORS["accent"])

    # ── Arrows: Input → Modules ──
    akw = dict(color="#666666", lw=0.9, style="-|>")
    _arrow(ax, 5.0, 8.4, 1.6, 7.4, **akw)
    _arrow(ax, 6.5, 8.4, 6.2, 7.4, **akw)
    _arrow(ax, 8.0, 8.4, 11.2, 7.4, **akw)

    # ── Gate prior dashed arrows ──
    gkw = dict(color=COLORS["module_a_edge"], lw=0.7,
               style="-|>", ls=(0, (4, 2)))
    _arrow(ax, 3.2, 6.0, 3.8, 6.0, **gkw)
    ax.annotate("", xy=(9.2, 6.4), xytext=(3.2, 6.8),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["module_a_edge"],
                                lw=0.7, ls=(0, (4, 2)),
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(6.0, 7.8, "gate prior",
            fontsize=7, color=COLORS["module_a_edge"], style="italic",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                      ec="none", alpha=0.9))

    # ── Modules → Fusion ──
    def _labeled_arrow(x_from, label):
        _arrow(ax, x_from, 4.6, 6.45, 3.0, **akw)
        ax.text(x_from, 3.85, label, ha="center", fontsize=6.5,
                color="#555555", style="italic",
                bbox=dict(boxstyle="round,pad=0.08", fc="white",
                          ec="none", alpha=0.85))

    _labeled_arrow(1.6, "prior")
    _labeled_arrow(6.2, "risk scores\n+ CDF probs")
    _labeled_arrow(11.2, r"$P(t \leq h)$")

    # ── Fusion → Output ──
    _arrow(ax, 6.45, 1.4, 6.45, 0.05, **akw)

    # ── Cross-fitting annotation (right side) ──
    ax.text(13.0, 3.4,
            "All meta-features\ngenerated via\ninner K-fold\ncross-fitting",
            fontsize=6, color="#999999", ha="center", va="center",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FAFAFA",
                      ec="#DDDDDD", lw=0.5))

    # ── Outer loop annotation (left side) ──
    ax.text(-0.2, 2.2,
            "Outer 5-fold\nstratified CV",
            fontsize=5.5, color="#AAAAAA", ha="center", va="center",
            style="italic", rotation=90)

    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure 2: Reliability Diagrams
# ═══════════════════════════════════════════════════════════════
def plot_reliability(y_dict, pred_dict, elig_dict, horizons, save_path=None):
    """Calibration reliability diagrams with Wilson CI bands and per-bin sample
    size annotations. One panel per prediction horizon."""
    fig, axes = plt.subplots(1, len(horizons),
                             figsize=(DOUBLE_COL, 2.8), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    bin_counts = {12: 5, 24: 6, 48: 6}

    for ax, h in zip(axes, horizons):
        mask = elig_dict[h]
        yt, yp = y_dict[h][mask], pred_dict[h][mask]
        n_bins = bin_counts.get(h, 6)
        edges = np.linspace(0, 1, n_bins + 1)
        centers, fracs, ci_lo, ci_hi, ns = [], [], [], [], []

        for i in range(n_bins):
            in_bin = (yp >= edges[i]) & (yp < edges[i + 1])
            if i == n_bins - 1:
                in_bin = (yp >= edges[i]) & (yp <= edges[i + 1])
            if in_bin.sum() >= 3:
                centers.append(yp[in_bin].mean())
                obs = yt[in_bin].mean()
                fracs.append(obs)
                nk = in_bin.sum()
                ns.append(nk)
                z = 1.96
                denom = 1 + z ** 2 / nk
                adj = (obs + z ** 2 / (2 * nk)) / denom
                margin = z * np.sqrt(
                    (obs * (1 - obs) + z ** 2 / (4 * nk)) / nk) / denom
                ci_lo.append(max(0, adj - margin))
                ci_hi.append(min(1, adj + margin))

        color = HORIZON_COLORS[h]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], ls="--", color="#BBBBBB", lw=0.6, zorder=1)

        # CI band — increased alpha
        if len(centers) >= 2:
            ax.fill_between(centers, ci_lo, ci_hi,
                            alpha=0.25, color=color, zorder=2,
                            edgecolor="none")
            ax.plot(centers, fracs, "o-", color=color, ms=5, lw=1.2,
                    mec="white", mew=0.5, zorder=4)
        elif len(centers) == 1:
            ax.errorbar(centers, fracs,
                        yerr=[[fracs[0] - ci_lo[0]], [ci_hi[0] - fracs[0]]],
                        fmt="o", color=color, ms=6, capsize=3, zorder=4)

        # Bin sample size annotations
        for cx, cy, n in zip(centers, fracs, ns):
            offset = 0.06 if cy < 0.9 else -0.08
            ax.text(cx, cy + offset, f"n={n}", ha="center", fontsize=5.5,
                    color="#999999", zorder=5)

        # Prediction histogram — more visible
        ax2 = ax.twinx()
        ax2.hist(yp, bins=edges, alpha=0.18, color=color, edgecolor="none")
        ax2.set_yticks([])
        ax2.set_ylim(0, len(yp) * 0.65)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        bs = np.mean((yt - yp) ** 2)
        ax.set_title(f"{HORIZON_LABELS[h]}\n"
                     f"Brier = {bs:.4f}  (n = {len(yt)})",
                     fontsize=8, pad=5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted probability", fontsize=8)
        if h == horizons[0]:
            ax.set_ylabel("Observed fraction", fontsize=8)
        ax.tick_params(direction="in", length=2.5)

    plt.tight_layout(w_pad=0.8)
    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure 3: Calibration Summary
# ═══════════════════════════════════════════════════════════════
def plot_calibration_summary(cal_df, horizons, save_path=None):
    """Brier score, ECE, and calibration slope summary across prediction horizons."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.8))

    x = np.arange(len(horizons))
    labels = [HORIZON_LABELS[h] for h in horizons]

    # (a) Brier Score
    ax = axes[0]
    vals = cal_df["brier"].values[:len(horizons)]
    bars = ax.bar(x, vals, width=0.5,
                  color=[HORIZON_COLORS[h] for h in horizons],
                  edgecolor="white", lw=0.5)
    ymax = max(vals) * 1.35
    ax.set_ylim(0, ymax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.025,
                f"{v:.4f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Brier Score", fontsize=8)
    ax.set_title("(a) Brier Score", fontsize=9, pad=8)
    ax.tick_params(direction="in", length=2.5)

    # (b) ECE
    ax = axes[1]
    vals = cal_df["ECE"].values[:len(horizons)]
    bars = ax.bar(x, vals, width=0.5,
                  color=[HORIZON_COLORS[h] for h in horizons],
                  edgecolor="white", lw=0.5)
    ymax = max(vals) * 1.35
    ax.set_ylim(0, ymax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.025,
                f"{v:.4f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ECE", fontsize=8)
    ax.set_title("(b) Expected Calibration Error", fontsize=9, pad=8)
    ax.tick_params(direction="in", length=2.5)

    # (c) Calibration Slope
    ax = axes[2]
    slopes = cal_df["cal_slope"].values[:len(horizons)]

    ax.axhspan(0.9, 1.1, alpha=0.12, color=COLORS["positive"],
               zorder=0, label="Ideal zone (±0.1)")
    ax.axhline(1.0, color=COLORS["accent"], ls="--", lw=0.8,
               alpha=0.7, label="Ideal (1.0)", zorder=1)

    for i, (h, s) in enumerate(zip(horizons, slopes)):
        color = HORIZON_COLORS[h]
        ax.plot(i, s, "o", color=color, ms=9, mec="white",
                mew=1.2, zorder=3)
        dev = s - 1.0
        va = "bottom" if s >= 1.0 else "top"
        offset = 0.06 if s >= 1.0 else -0.06
        ax.text(i + 0.2, s + offset,
                f"{s:.3f}\n({dev:+.3f})",
                ha="left", va=va, fontsize=6.5, color=color,
                fontweight="bold")

    ymin = min(min(slopes) - 0.15, 0.4)
    ymax_s = max(max(slopes) + 0.15, 1.25)
    ax.set_ylim(ymin, ymax_s)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Calibration Slope", fontsize=8)
    ax.set_title("(c) Calibration Slope (ideal = 1.0)", fontsize=9, pad=8)
    ax.legend(fontsize=6, frameon=True, fancybox=False,
              edgecolor="#CCCCCC", loc="lower right")
    ax.tick_params(direction="in", length=2.5)

    plt.tight_layout(w_pad=1.8)
    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure 4: Ablation Forest Plot
# ═══════════════════════════════════════════════════════════════
def plot_ablation(ablation_df, save_path=None):
    """Forest plot of ablation Δ-metrics with bootstrap CI whiskers
    and two-sided significance annotations."""
    abl_only = ablation_df[ablation_df["config"] != "Full Model"].copy()

    buildup_names = ["AFT Only", "AFT + Direct",
                     "AFT + Direct + Gate", "AFT + Direct + Cox"]
    removal_names = ["w/o Cox Branch", "w/o Direct Heads",
                     "w/o Gate Prior", "w/o IPCW",
                     "w/o Simple Dist.", "w/o Simple Distance",
                     "Thin Features", "Thin Features Only"]

    ordered_configs = []
    for name in buildup_names:
        match = abl_only[abl_only["config"].str.strip() == name]
        if len(match) > 0:
            ordered_configs.append(match.iloc[0])
    for name in removal_names:
        match = abl_only[abl_only["config"].str.strip() == name]
        if len(match) > 0:
            ordered_configs.append(match.iloc[0])
    matched_names = set(r["config"].strip() for r in ordered_configs)
    for _, row in abl_only.iterrows():
        if row["config"].strip() not in matched_names:
            ordered_configs.append(row)

    n_rows = len(ordered_configs)
    n_buildup = sum(1 for r in ordered_configs
                    if r["config"].strip() in buildup_names)

    fig_h = max(3.8, 0.42 * n_rows + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fig_h),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    def _fmt_delta(d):
        if abs(d) < 0.0005:
            return "+0.000"
        return f"{d:+.3f}"

    for ax_idx, (ax, delta_col, p_col, xlabel, title) in enumerate(zip(
            axes,
            ["delta_c_index", "delta_ibs"],
            ["p_c_index", "p_ibs"],
            [r"$\Delta$C-index", r"$\Delta$IBS proxy"],
            ["(a) C-index contribution", "(b) IBS proxy contribution"],
    )):
        xmax = max(abs(r[delta_col]) for r in ordered_configs) + 1e-6

        for i, row in enumerate(ordered_configs):
            d = row[delta_col]
            c = COLORS["positive"] if d >= 0 else COLORS["negative"]
            ax.barh(i, d, height=0.52, color=c, alpha=0.65,
                    edgecolor="white", lw=0.5)

            # CI whiskers — try multiple column name patterns
            for ci_lo_col, ci_hi_col in [
                (f"{delta_col}_ci_lo", f"{delta_col}_ci_hi"),
                (f"{delta_col}_lo", f"{delta_col}_hi"),
            ]:
                if ci_lo_col in row.index and ci_hi_col in row.index:
                    lo = row.get(ci_lo_col, d)
                    hi = row.get(ci_hi_col, d)
                    if pd.notna(lo) and pd.notna(hi):
                        ax.plot([lo, hi], [i, i], color="#444444",
                                lw=0.6, zorder=5)
                        ax.plot([lo, lo], [i - 0.1, i + 0.1],
                                color="#444444", lw=0.6, zorder=5)
                        ax.plot([hi, hi], [i - 0.1, i + 0.1],
                                color="#444444", lw=0.6, zorder=5)
                    break

            p = row.get(p_col, 0.5)
            sig = " *" if (pd.notna(p) and p < 0.05) else ""
            label = f"{_fmt_delta(d)}{sig}"
            x_text = max(d, 0) + xmax * 0.08
            ax.text(x_text, i, label, va="center", fontsize=7,
                    ha="left",
                    fontweight="bold" if sig else "normal")

        # Group separator
        if 0 < n_buildup < n_rows:
            sep_y = n_buildup - 0.5
            ax.axhline(sep_y, color="#AAAAAA", lw=0.8, ls="-", zorder=0)
            if ax_idx == 0:
                ax.text(-xmax * 0.12, n_buildup / 2 - 0.5,
                        "Build-up", fontsize=6, color="#666666",
                        ha="right", va="center", style="italic",
                        rotation=90)
                ax.text(-xmax * 0.12,
                        n_buildup + (n_rows - n_buildup) / 2 - 0.5,
                        "Removal", fontsize=6, color="#666666",
                        ha="right", va="center", style="italic",
                        rotation=90)

        ax.set_yticks(range(n_rows))
        if ax_idx == 0:
            config_labels = [r["config"].strip() for r in ordered_configs]
            ax.set_yticklabels(config_labels, fontsize=7)
        else:
            ax.set_yticklabels([""] * n_rows)

        ax.axvline(0, color="#333333", lw=0.7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9, pad=6)
        ax.tick_params(direction="in", length=2.5)
        ax.set_xlim(-xmax * 0.18, xmax * 1.85)
        ax.set_ylim(-0.5, n_rows - 0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["positive"], alpha=0.65,
              label="Full model better"),
        Patch(facecolor=COLORS["negative"], alpha=0.65,
              label="Module dispensable"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=6.5, frameon=True,
                   fancybox=False, edgecolor="#CCCCCC", loc="lower right")

    fig.text(0.98, 0.02,
             r"Positive $\Delta$ = full model better;  * $p$ < 0.05",
             ha="right", fontsize=6.5, color="#777777", style="italic")

    plt.tight_layout(w_pad=0.6, rect=[0.03, 0.04, 1, 1])
    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure 5: Bootstrap CI Distributions
# ═══════════════════════════════════════════════════════════════
def plot_bootstrap_ci(bs, horizons, save_path=None):
    """Histogram of bootstrapped metric distributions with 95% CI markers."""
    metrics = [("c_index", "C-index"), ("ibs", "Mean Brier\n(IBS proxy)")]
    metrics += [(f"brier_{h}h", f"Brier@{h}h") for h in horizons]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, 2.6))
    if n == 1:
        axes = [axes]

    palette = [COLORS["accent"], COLORS["24h"]] + \
              [HORIZON_COLORS[h] for h in horizons]

    for ax, (key, label), col in zip(axes, metrics, palette):
        vals = bs[key]
        m = np.median(vals)
        lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)

        ax.hist(vals, bins=30, color=col, alpha=0.45,
                edgecolor="white", lw=0.3)
        ymax = ax.get_ylim()[1]

        # CI band
        ax.fill_betweenx([0, ymax * 1.05], lo, hi,
                         alpha=0.15, color=COLORS["accent"])
        ax.axvline(m, color=COLORS["accent"], lw=1.3, zorder=5)
        ax.axvline(lo, color=COLORS["accent"], lw=0.8,
                   ls="--", alpha=0.8, zorder=5)
        ax.axvline(hi, color=COLORS["accent"], lw=0.8,
                   ls="--", alpha=0.8, zorder=5)

        ax.set_xlabel(label, fontsize=7.5)
        ax.set_title(f"{m:.4f}\n[{lo:.4f}, {hi:.4f}]",
                     fontsize=7, pad=6, fontweight="bold")
        ax.set_yticks([])
        ax.tick_params(direction="in", length=2.5, labelsize=6.5)
        ax.locator_params(axis='x', nbins=5)

    plt.tight_layout(w_pad=0.7)
    fig.subplots_adjust(top=0.78)
    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure 6: Prediction Distributions + Risk Stratification
# ═══════════════════════════════════════════════════════════════
def plot_distributions(pred_dict, y_dict, elig_dict, horizons,
                       save_path=None):
    """Out-of-fold predicted probability distributions per horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8),
                             gridspec_kw={"width_ratios": [1, 0.85]})

    # (a) Prediction KDEs
    ax = axes[0]
    from scipy.stats import gaussian_kde
    for h in horizons:
        yp = pred_dict[h]
        color = HORIZON_COLORS[h]
        ax.hist(yp, bins=25, density=True, alpha=0.08,
                color=color, edgecolor="none")
        try:
            kde = gaussian_kde(yp, bw_method="scott")
            xs = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=color, lw=1.2,
                    label=HORIZON_LABELS[h])
        except Exception:
            pass
    ax.set_xlabel("Predicted probability", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("(a) Prediction distributions", fontsize=9, pad=6)
    ax.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC",
              fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(direction="in", length=2.5)

    # (b) Risk stratification
    ax = axes[1]
    if 24 in horizons and 24 in pred_dict and 24 in y_dict:
        p24 = pred_dict[24]
        yt24 = y_dict[24]
        elig24 = elig_dict[24]
        p24_e = p24[elig24]
        yt24_e = yt24[elig24]
        terciles = np.percentile(p24_e, [33.3, 66.7])
        groups = np.digitize(p24_e, terciles)
        labels_g = ["Low risk", "Medium risk", "High risk"]
        gc = [COLORS["positive"], COLORS["24h"], COLORS["accent"]]
        rates = []
        for g in range(3):
            mask_g = groups == g
            rates.append(float(yt24_e[mask_g].mean())
                         if mask_g.sum() > 0 else 0)

        bars = ax.bar(labels_g, rates, color=gc, edgecolor="white",
                      width=0.55, alpha=0.85)
        for b, v in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03,
                    f"{v:.1%}", ha="center", fontsize=8,
                    fontweight="bold")
        ax.set_ylabel("Observed event rate", fontsize=8)
        ax.set_title("(b) Risk stratification (24 h terciles)",
                     fontsize=9, pad=6)
        ax.set_ylim(0, 1.15)
    ax.tick_params(direction="in", length=2.5)

    plt.tight_layout(w_pad=1.2)
    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure 7: Baseline Comparison
# ═══════════════════════════════════════════════════════════════
def plot_baseline_comparison(comparison_df, save_path=None):
    """Horizontal bar chart comparing C-index and IBS across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))

    df_s = comparison_df.sort_values("c_index", ascending=True)

    # (a) C-index
    ax = axes[0]
    colors = [COLORS["accent"] if "Full" in str(c)
              else COLORS["secondary"] for c in df_s["model"]]
    bars = ax.barh(df_s["model"], df_s["c_index"], color=colors,
                   edgecolor="white", height=0.55)
    for b, v in zip(bars, df_s["c_index"]):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=7.5, fontweight="bold")
    xmin = max(0, df_s["c_index"].min() - 0.03)
    ax.set_xlim(xmin, df_s["c_index"].max() + 0.035)
    ax.set_xlabel("C-index (higher is better)", fontsize=8)
    ax.set_title("(a) Concordance Index", fontsize=9, pad=6)
    ax.tick_params(direction="in", length=2.5, labelsize=8)

    # (b) IBS
    ax = axes[1]
    df_s2 = comparison_df.sort_values("ibs", ascending=False)
    colors2 = [COLORS["accent"] if "Full" in str(c)
               else COLORS["secondary"] for c in df_s2["model"]]
    bars = ax.barh(df_s2["model"], df_s2["ibs"], color=colors2,
                   edgecolor="white", height=0.55)
    for b, v in zip(bars, df_s2["ibs"]):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("IBS proxy (lower is better)", fontsize=8)
    ax.set_title("(b) Mean Brier (IBS proxy)", fontsize=9, pad=6)
    ax.tick_params(direction="in", length=2.5, labelsize=8)

    plt.tight_layout(w_pad=1.0)
    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure 8: Recalibration Comparison
# ═══════════════════════════════════════════════════════════════
def plot_recalibration_comparison(recal_df, horizons, save_path=None):
    """Pre- vs post-recalibration Brier, ECE, and calibration slope comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 3.0))

    horizon_rows = recal_df[recal_df["horizon"] != "IBS"].copy()
    x = np.arange(len(horizon_rows))
    width = 0.32
    h_labels = horizon_rows["horizon"].values

    panels = [
        ("pre_brier", "post_brier", "Brier Score", "(a) Brier Score"),
        ("pre_ece", "post_ece", "ECE", "(b) Expected Calibration Error"),
        ("pre_slope", "post_slope", "Calibration Slope",
         "(c) Calibration Slope"),
    ]

    for ax, (pre_col, post_col, ylabel, title) in zip(axes, panels):
        pre_vals = horizon_rows[pre_col].values.astype(float)
        post_vals = horizon_rows[post_col].values.astype(float)

        post_colors = [HORIZON_COLORS[h] for h in horizons[:len(x)]]

        bars_pre = ax.bar(x - width / 2, pre_vals, width,
                          label="Pre-recal", color="#AAAAAA",
                          edgecolor="white", alpha=0.75)
        bars_post = ax.bar(x + width / 2, post_vals, width,
                           label="Post-recal", color=post_colors,
                           edgecolor="white", alpha=0.85)

        if "slope" in pre_col:
            ymax = max(np.nanmax(np.concatenate(
                [pre_vals, post_vals])) * 1.15, 1.35)
            ax.axhline(1.0, color=COLORS["accent"], ls="--",
                       lw=0.8, alpha=0.7)
            ax.set_ylim(0, ymax)
        else:
            all_vals = np.concatenate([pre_vals, post_vals])
            all_vals = all_vals[~np.isnan(all_vals)]
            if len(all_vals) == 0:
                continue
            ymax = np.nanmax(all_vals) * 1.4
            ax.set_ylim(0, ymax)

        # Annotate values — compliant fontsize
        for b, v in zip(bars_pre, pre_vals):
            if not np.isnan(v):
                fmt = f"{v:.3f}" if "slope" in pre_col else f"{v:.4f}"
                ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02,
                        fmt, ha="center", va="bottom",
                        fontsize=6, color="#666666")
        for b, v in zip(bars_post, post_vals):
            if not np.isnan(v):
                fmt = f"{v:.3f}" if "slope" in pre_col else f"{v:.4f}"
                ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02,
                        fmt, ha="center", va="bottom",
                        fontsize=6, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(h_labels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, pad=6)
        ax.legend(fontsize=6, frameon=True, fancybox=False,
                  edgecolor="#CCCCCC", loc="upper right")
        ax.tick_params(direction="in", length=2.5)

    plt.tight_layout(w_pad=1.0)
    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure S1: Monotonicity Verification (Supplementary)
# ═══════════════════════════════════════════════════════════════
def plot_monotonicity(pred_dict, horizons, save_path=None):
    """Scatter plot verifying P(hit by t1) ≤ P(hit by t2) across horizon pairs."""
    pairs = [(horizons[i], horizons[i + 1])
             for i in range(len(horizons) - 1)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(DOUBLE_COL, 3.0))
    if len(pairs) == 1:
        axes = [axes]

    for idx, (ax, (h1, h2)) in enumerate(zip(axes, pairs)):
        ax.scatter(pred_dict[h1], pred_dict[h2], s=3.0, alpha=0.4,
                   color=COLORS["primary"], rasterized=True, zorder=2,
                   edgecolors="none")
        ax.plot([0, 1], [0, 1], "--", color=COLORS["accent"],
                lw=0.8, alpha=0.6, zorder=1)
        v = int(np.sum(pred_dict[h2] < pred_dict[h1] - 1e-6))
        ax.set_xlabel(f"P({HORIZON_LABELS[h1]})", fontsize=8)
        ax.set_ylabel(f"P({HORIZON_LABELS[h2]})", fontsize=8)
        ax.set_title(
            f"({chr(97 + idx)}) {HORIZON_LABELS[h1]} → {HORIZON_LABELS[h2]}  "
            f"(violations: {v})", fontsize=8, pad=6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(direction="in", length=2.5)

    plt.tight_layout()
    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Graphical Abstract
# ═══════════════════════════════════════════════════════════════
def plot_graphical_abstract(metrics, save_path=None):
    """Visual summary of the proposed framework for the journal graphical abstract."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Title
    ax.text(6.5, 7.8,
            "Gate-Informed Survival–Probability Fusion\n"
            "for Multi-Horizon Wildfire Threat Forecasting",
            ha="center", fontsize=11, fontweight="bold",
            color=COLORS["primary"], linespacing=1.4)

    # Pipeline boxes
    boxes = [
        (0.3, 4.0, 3.2, 2.2,
         "Gate Prior\n+\nAFT / Cox\nSurvival Backbone",
         COLORS["module_b"], COLORS["module_b_edge"]),
        (4.3, 4.0, 3.2, 2.2,
         "Calibrated\nHorizon Heads\n12 / 24 / 48 h\n+ Beta Cal.",
         COLORS["module_c"], COLORS["module_c_edge"]),
        (8.3, 4.0, 3.2, 2.2,
         "Simplex Stacking\n+\nMonotone Fusion\n"
         r"$P(12h) \leq P(24h) \leq P(48h)$",
         COLORS["module_d"], COLORS["module_d_edge"]),
    ]
    for bx, by, bw, bh, bt, bfc, bec in boxes:
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                    boxstyle="round,pad=0.12",
                                    fc=bfc, ec=bec, lw=1.2))
        ax.text(bx + bw / 2, by + bh / 2, bt,
                ha="center", va="center",
                fontsize=7, fontweight="bold", color=bec, linespacing=1.3)

    # Arrows
    akw = dict(arrowstyle="-|>", color="#444", lw=1.0)
    ax.annotate("", xy=(4.3, 5.1), xytext=(3.5, 5.1), arrowprops=akw)
    ax.annotate("", xy=(8.3, 5.1), xytext=(7.5, 5.1), arrowprops=akw)

    # Key results
    ci = metrics.get("c_index", 0.941)
    ibs = metrics.get("ibs", 0.043)
    b24 = metrics.get("brier_24h", 0.038)

    result_text = (f"C-index = {ci:.3f}     "
                   f"Mean Brier = {ibs:.3f}     "
                   f"Brier@24h = {b24:.3f}")
    ax.text(6.5, 2.6, result_text,
            ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E1",
                      ec=COLORS["accent"], lw=1.2))

    # Risk stratification highlight
    ax.text(6.5, 1.6,
            "Risk stratification: 0.0% (low) · 6.1% (mid) · 90.8% (high)",
            ha="center", fontsize=7.5, color=COLORS["accent"],
            fontweight="bold")

    # Properties
    ax.text(6.5, 0.9,
            "Censor-aware  ·  Multi-horizon  ·  "
            "Monotone-constrained  ·  Small-sample robust",
            ha="center", fontsize=7, color="#555", style="italic")

    ax.text(6.5, 0.3, "Internal validation only (nested 5-fold CV)",
            ha="center", fontsize=6, color="#999", style="italic")

    _save_mdpi(fig, save_path)


# ═══════════════════════════════════════════════════════════════
# Figure S2: Outer-Fold Prediction Distributions (Supplementary)
# ═══════════════════════════════════════════════════════════════
def plot_outer_fold_distributions(pred_dict, fold_indices, horizons,
                                  save_path=None):
    """Per-fold predicted probability histograms for CV consistency verification."""
    n_horizons = len(horizons)
    fig, axes = plt.subplots(1, n_horizons, figsize=(DOUBLE_COL, 2.6),
                             sharey=True)
    if n_horizons == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        preds = pred_dict[h]
        color = HORIZON_COLORS[h]
        if fold_indices is not None:
            unique_folds = sorted(set(fold_indices))
            for fold_id in unique_folds:
                mask = np.array(fold_indices) == fold_id
                ax.hist(preds[mask], bins=20, density=True,
                        alpha=0.3, color=color, edgecolor="none",
                        label=f"Fold {fold_id}")
        else:
            ax.hist(preds, bins=25, density=True,
                    alpha=0.45, color=color, edgecolor="white", lw=0.3)

        ax.set_xlabel("Predicted probability", fontsize=8)
        ax.set_title(f"{HORIZON_LABELS[h]}", fontsize=9, pad=6)
        ax.tick_params(direction="in", length=2.5)

    axes[0].set_ylabel("Density", fontsize=8)
    if fold_indices is not None:
        axes[-1].legend(fontsize=6, frameon=True, fancybox=False,
                        edgecolor="#CCCCCC", loc="upper right")
    plt.tight_layout(w_pad=0.8)
    _save_mdpi(fig, save_path)
# ================================================================
# §14  Main Pipeline
# ================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Gate-Informed Survival-Probability Fusion — Wildfire Threat Forecasting"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("WIDS_DATA_DIR", ".")),
        help="Directory containing train.csv and test.csv "
             "(default: $WIDS_DATA_DIR or current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("WIDS_OUTPUT_DIR", "paper_output")),
        help="Directory for all outputs and figures "
             "(default: $WIDS_OUTPUT_DIR or ./paper_output).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Global random seed (default: 42)."
    )
    parser.add_argument(
        "--outer-folds", type=int, default=5,
        help="Number of outer CV folds (default: 5).",
    )
    parser.add_argument(
        "--inner-folds", type=int, default=5,
        help="Number of inner CV folds (default: 5).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap iterations for CI and paired tests (default: 1000).",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        n_bootstrap=args.n_bootstrap,
    )
    seed_everything(cfg.random_seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    set_paper_style()

    # ── Load data ──
    print_header("STEP 1: Loading Data")
    train = pd.read_csv(cfg.data_dir / "train.csv")
    test = pd.read_csv(cfg.data_dir / "test.csv")
    print(f"  Train: {train.shape[0]} rows × {train.shape[1]} cols")
    print(f"  Test:  {test.shape[0]} rows × {test.shape[1]} cols")
    print(f"  Events: {int(train[cfg.target_event].sum())} hit / "
          f"{int((train[cfg.target_event] == 0).sum())} censored")

    # ── Feature engineering ──
    print_header("STEP 2: Feature Engineering")
    train_fe = feature_engineering(train)
    test_fe = feature_engineering(test)

    # ── Censor-aware targets ──
    print_header("STEP 3: Censor-Aware Horizon Targets")
    train_fe = create_censor_aware_targets(train_fe, cfg)
    for h in cfg.all_horizons:
        pos = int(train_fe[f"hit_by_{h}h"].sum())
        elig = int(train_fe[f"eligible_{h}h"].sum())
        rate = pos / elig if elig > 0 else 0
        flag = " ← DEGENERATE (supplement only)" if rate > cfg.degenerate_pos_rate else ""
        print(f"  {h}h: {pos}/{elig} positives ({rate:.1%}){flag}")

    feature_sets = build_feature_sets(list(train_fe.columns), cfg)

    # ── Full model: nested evaluation ──
    print_header("STEP 4: Nested Evaluation — Full Model")
    full_result = nested_evaluate(train_fe, feature_sets, FULL_MODEL, cfg)

    payload = _build_payload(train_fe, cfg)
    y_time = train_fe[cfg.target_time].values
    y_event = train_fe[cfg.target_event].values
    full_metrics = compute_all_metrics(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        full_result["oof_preds"],
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        full_result["oof_risk"],
        payload["pair_i"], payload["pair_j"],
        y_time=y_time, y_event=y_event,
        horizons=cfg.main_horizons)

    print(f"\n  Full Model Results (nested OOF, main horizons only):")
    print(f"    C-index (proxy): {full_metrics['c_index_proxy']:.4f}")
    print(f"    C-index (Uno):   {full_metrics.get('c_index_uno', 'N/A')}")
    print(f"    IBS (proxy):     {full_metrics['ibs_proxy']:.4f}")
    print(f"    IBS (continuous):{full_metrics.get('ibs_continuous', 'N/A')}")
    for h in cfg.main_horizons:
        print(f"    Brier@{h}h:   {full_metrics[f'brier_{h}h']:.4f}  "
              f"ECE={full_metrics[f'ece_{h}h']:.4f}  "
              f"slope={full_metrics[f'cal_slope_{h}h']:.3f}")

    # Save pre-recalibration metrics, predictions, AND full result
    # for consistent use in ablation (Step 5) and bootstrap CI (Step 6)
    pre_recal_metrics = dict(full_metrics)
    pre_recal_preds = copy.deepcopy(full_result["oof_preds"])
    pre_recal_result = copy.deepcopy(full_result)  # for ablation & bootstrap

    # ── [R3] Joint Post-Fusion Recalibration (sensitivity analysis) ──
    # Replaces single-horizon temperature scaling with bidirectional
    # monotone-aware Platt recalibration across ALL horizons jointly.
    print_header("STEP 4b: Joint Post-Fusion Recalibration (sensitivity analysis)")
    recal_preds_joint = joint_recalibration(
        full_result["oof_preds"],
        y_dict={h: payload[f"y{h}"] for h in cfg.main_horizons},
        elig_dict={h: payload[f"elig{h}"] for h in cfg.main_horizons},
        horizons=cfg.main_horizons,
        seed=cfg.random_seed)
    full_result["oof_preds"] = recal_preds_joint

    # Also run legacy 12h temperature scaling for comparison
    recal_preds_12h = recalibrate_12h(
        pre_recal_preds.copy(),
        payload["y12"], payload["elig12"],
        seed=cfg.random_seed)

    # Recompute metrics after joint recalibration
    full_metrics = compute_all_metrics(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        full_result["oof_preds"],
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        full_result["oof_risk"],
        payload["pair_i"], payload["pair_j"],
        y_time=y_time, y_event=y_event,
        horizons=cfg.main_horizons)
    print(f"  After joint recalibration:")
    for h in cfg.main_horizons:
        print(f"    Brier@{h}h:   {full_metrics[f'brier_{h}h']:.4f}  "
              f"ECE={full_metrics[f'ece_{h}h']:.4f}  "
              f"slope={full_metrics[f'cal_slope_{h}h']:.3f}")

    # Also compute 12h-only recalibration metrics for comparison table
    recal_12h_metrics = compute_all_metrics(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        recal_preds_12h,
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        full_result["oof_risk"],
        payload["pair_i"], payload["pair_j"],
        y_time=y_time, y_event=y_event,
        horizons=cfg.main_horizons)
    print(f"\n  Legacy 12h temperature scaling (for comparison):")
    for h in cfg.main_horizons:
        print(f"    Brier@{h}h:   {recal_12h_metrics[f'brier_{h}h']:.4f}  "
              f"ECE={recal_12h_metrics[f'ece_{h}h']:.4f}  "
              f"slope={recal_12h_metrics[f'cal_slope_{h}h']:.3f}")

    # Build pre vs post recalibration comparison table (both methods)
    recal_comparison_rows = []
    for h in cfg.main_horizons:
        recal_comparison_rows.append({
            "horizon": f"{h}h",
            "pre_brier": pre_recal_metrics[f"brier_{h}h"],
            "post_joint_brier": full_metrics[f"brier_{h}h"],
            "post_12h_brier": recal_12h_metrics[f"brier_{h}h"],
            "pre_ece": pre_recal_metrics[f"ece_{h}h"],
            "post_joint_ece": full_metrics[f"ece_{h}h"],
            "post_12h_ece": recal_12h_metrics[f"ece_{h}h"],
            "pre_slope": pre_recal_metrics[f"cal_slope_{h}h"],
            "post_joint_slope": full_metrics[f"cal_slope_{h}h"],
            "post_12h_slope": recal_12h_metrics[f"cal_slope_{h}h"],
        })
    recal_comparison_rows.append({
        "horizon": "IBS",
        "pre_brier": pre_recal_metrics["ibs"],
        "post_joint_brier": full_metrics["ibs"],
        "post_12h_brier": recal_12h_metrics["ibs"],
        "pre_ece": np.nan, "post_joint_ece": np.nan, "post_12h_ece": np.nan,
        "pre_slope": np.nan, "post_joint_slope": np.nan, "post_12h_slope": np.nan,
    })
    recal_comparison_df = pd.DataFrame(recal_comparison_rows)
    recal_comparison_df.to_csv(
        cfg.output_dir / "recalibration_comparison.csv", index=False)
    print(f"\n  Pre vs post recalibration comparison:")
    print(recal_comparison_df.to_string(index=False))

    # ── Baseline Comparison ──
    print_header("STEP 4c: Baseline Comparisons")
    print("  Running ETA-only baseline...")
    eta_preds, eta_risk = eta_only_baseline(train_fe, cfg)
    eta_metrics = compute_all_metrics(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        eta_preds,
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        eta_risk,
        payload["pair_i"], payload["pair_j"],
        y_time=y_time, y_event=y_event,
        horizons=cfg.main_horizons)
    print(f"    ETA-only:  C-index={eta_metrics['c_index']:.4f}  "
          f"IBS={eta_metrics['ibs']:.4f}")

    print("  Running per-horizon XGBoost baseline...")
    xgb_preds, xgb_risk = xgb_per_horizon_baseline(train_fe, feature_sets, cfg)
    xgb_metrics = compute_all_metrics(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        xgb_preds,
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        xgb_risk,
        payload["pair_i"], payload["pair_j"],
        y_time=y_time, y_event=y_event,
        horizons=cfg.main_horizons)
    print(f"    XGB-perH:  C-index={xgb_metrics['c_index']:.4f}  "
          f"IBS={xgb_metrics['ibs']:.4f}")

    print("  Running Random Survival Forest baseline...")
    rsf_preds, rsf_risk = rsf_baseline(train_fe, feature_sets, cfg)
    if rsf_preds is not None:
        rsf_metrics = compute_all_metrics(
            {h: payload[f"y{h}"] for h in cfg.main_horizons},
            rsf_preds,
            {h: payload[f"elig{h}"] for h in cfg.main_horizons},
            rsf_risk,
            payload["pair_i"], payload["pair_j"],
            y_time=y_time, y_event=y_event,
            horizons=cfg.main_horizons)
        print(f"    RSF:       C-index={rsf_metrics['c_index']:.4f}  "
              f"IBS={rsf_metrics['ibs']:.4f}")
    else:
        rsf_metrics = {"c_index": np.nan, "ibs": np.nan}
        print("    RSF:       SKIPPED (scikit-survival not installed)")

    print("  Running AFT-only baseline...")
    aft_result = nested_evaluate(train_fe, feature_sets,
                                 VariantConfig(name="aft_baseline", label="AFT Only",
                                               use_gate=False, use_direct=False, use_ipcw=False,
                                               use_simple=False, use_cox=False), cfg)
    aft_metrics = compute_all_metrics(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        aft_result["oof_preds"],
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        aft_result["oof_risk"],
        payload["pair_i"], payload["pair_j"],
        y_time=y_time, y_event=y_event,
        horizons=cfg.main_horizons)
    print(f"    AFT-only:  C-index={aft_metrics['c_index']:.4f}  "
          f"IBS={aft_metrics['ibs']:.4f}")

    # Build comparison table (using pre-recal metrics for Full Model)
    # [R1][R2] Include Uno's C and continuous IBS alongside proxy metrics
    _cmp_keys = ["c_index", "c_index_uno", "ibs", "ibs_continuous"]

    def _safe_pick(m, keys):
        return {k: m.get(k, float("nan")) for k in keys}

    comparison_rows = [
        {"model": "ETA-only (LR)", **_safe_pick(eta_metrics, _cmp_keys)},
        {"model": "Per-horizon XGBoost", **_safe_pick(xgb_metrics, _cmp_keys)},
        {"model": "Random Survival Forest", **_safe_pick(rsf_metrics, _cmp_keys)},
        {"model": "AFT Only", **_safe_pick(aft_metrics, _cmp_keys)},
        {"model": "Full Model", **_safe_pick(pre_recal_metrics, _cmp_keys)},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(cfg.output_dir / "baseline_comparison.csv", index=False)
    print(f"\n  Baseline comparison:")
    print(comparison_df.to_string(index=False))

    # ── Pre-fusion monotonicity violations ──
    print_header("STEP 4d: Pre-Fusion Monotonicity Violations")
    prefusion_stats = count_prefusion_violations(
        pre_recal_result["oof_preds_raw"], cfg.main_horizons)
    for k, v in prefusion_stats.items():
        print(f"    {k}: {v}")
    pd.DataFrame([prefusion_stats]).to_csv(
        cfg.output_dir / "prefusion_violations.csv", index=False)

    # ── Risk stratification with n and Wilson CIs ──
    print_header("STEP 4e: Risk Stratification (24h terciles with CIs)")
    p24_elig = payload["elig24"]
    y24_elig = payload["y24"][p24_elig]
    p24_pred = pre_recal_preds[24][p24_elig]
    tercile_cuts = np.percentile(p24_pred, [33.33, 66.67])
    groups = np.digitize(p24_pred, tercile_cuts)
    for gi, label in enumerate(["Low", "Medium", "High"]):
        mask = groups == gi
        n_grp = mask.sum()
        rate = y24_elig[mask].mean() if n_grp > 0 else 0
        # Wilson confidence interval
        z = 1.96
        denom = 1 + z**2 / n_grp if n_grp > 0 else 1
        centre = (rate + z**2 / (2 * n_grp)) / denom if n_grp > 0 else rate
        margin = z * np.sqrt((rate * (1 - rate) / n_grp + z**2 / (4 * n_grp**2))) / denom if n_grp > 0 else 0
        ci_lo, ci_hi = max(centre - margin, 0), min(centre + margin, 1)
        print(f"    {label}: n={n_grp}, event rate={rate:.3f} "
              f"[{ci_lo:.3f}, {ci_hi:.3f}]")

    # ── Ablation study (uses pre-recalibration predictions for consistency) ──
    print_header("STEP 5: Ablation Study with Paired Δ-Metric Bootstrap")
    ablation_df = run_ablation_study(train_fe, feature_sets, pre_recal_result, cfg)
    ablation_df.to_csv(cfg.output_dir / "ablation_results.csv", index=False)
    print(f"\n  Ablation table saved to {cfg.output_dir / 'ablation_results.csv'}")

    # ── Practical lean variants (full + thin features) ──
    print_header("STEP 5b: Practical Lean Variants (full & thin features)")
    practical_df = run_practical_variants(train_fe, feature_sets, pre_recal_result, cfg)
    practical_df.to_csv(cfg.output_dir / "practical_variants.csv", index=False)
    print(f"\n  Practical variants table saved to {cfg.output_dir / 'practical_variants.csv'}")
    summary_cols = ["config", "c_index", "ibs"] + \
                   [f"brier_{h}h" for h in cfg.main_horizons] + \
                   [f"cal_slope_{h}h" for h in cfg.main_horizons] + \
                   [f"ece_{h}h" for h in cfg.main_horizons]
    avail_cols = [c for c in summary_cols if c in practical_df.columns]
    print(practical_df[avail_cols].to_string(index=False))

    # ── Cox design diagnostic ──
    # Cox PH branch only enters _compose_risk() for ranking (C-index).
    # It does NOT enter the stacking matrix for horizon probabilities.
    # Therefore: AFT Only vs AFT+Cox produce IDENTICAL Brier/ECE/slope
    # (Cox changes only the composite risk signal → C-index).
    # This is BY DESIGN: Cox is a ranking enhancer, not a probability source.
    print_header("STEP 5b-NOTE: Cox Branch Design Diagnostic")
    print("  DESIGN NOTE — Cox PH branch architecture:")
    print("    Cox risk enters _compose_risk() for the composite ranking signal.")
    print("    Cox risk does NOT enter the stacking matrix for P(T≤h).")
    print("    → AFT Only and AFT+Cox have identical Brier/ECE/cal_slope.")
    print("    → AFT+Simple and AFT+Cox+Simple also share Brier/ECE/cal_slope.")
    print("    → Cox's sole contribution is to C-index (discrimination).")
    print("    This is by design: Cox is a ranking enhancer, not a calibration source.")
    print("    Paper must document this explicitly in the Methods section.")

    # Verify this programmatically
    pv_aft_only = practical_df[practical_df["config"] == "Full Model"]
    for pair_a, pair_b in [("AFT Only", "AFT + Cox"),
                           ("AFT + Simple", "AFT + Cox + Simple")]:
        row_a = practical_df[practical_df["config"] == pair_a]
        row_b = practical_df[practical_df["config"] == pair_b]
        if len(row_a) > 0 and len(row_b) > 0:
            brier_match = all(
                abs(row_a.iloc[0].get(f"brier_{h}h", 0) -
                    row_b.iloc[0].get(f"brier_{h}h", 0)) < 1e-8
                for h in cfg.main_horizons)
            ci_diff = abs(row_a.iloc[0]["c_index"] - row_b.iloc[0]["c_index"])
            print(f"    {pair_a} vs {pair_b}: "
                  f"Brier identical={brier_match}, "
                  f"ΔC-index={ci_diff:.4f}")

    # ── Comprehensive paper main table ──
    print_header("STEP 5c: Comprehensive Model Comparison (Paper Main Table)")
    metric_keys = ["c_index", "ibs"] + \
                  [f"brier_{h}h" for h in cfg.main_horizons] + \
                  [f"cal_slope_{h}h" for h in cfg.main_horizons] + \
                  [f"ece_{h}h" for h in cfg.main_horizons]

    key_lean_models = {}
    for _, row in practical_df.iterrows():
        if row["config"] in ("AFT + Cox", "AFT + Cox + Simple"):
            key_lean_models[row["config"]] = row.to_dict()

    main_table_rows = [
        {"model": "ETA-only (LR)",
         **{k: eta_metrics.get(k, np.nan) for k in metric_keys}},
        {"model": "AFT Only",
         **{k: aft_metrics.get(k, np.nan) for k in metric_keys}},
    ]
    for mn in ["AFT + Cox", "AFT + Cox + Simple"]:
        if key_lean_models.get(mn):
            r = key_lean_models[mn]
            main_table_rows.append(
                {"model": mn, **{k: r.get(k, np.nan) for k in metric_keys}})

    main_table_rows.extend([
        {"model": "Per-horizon XGBoost",
         **{k: xgb_metrics.get(k, np.nan) for k in metric_keys}},
        {"model": "Random Survival Forest",
         **{k: rsf_metrics.get(k, np.nan) for k in metric_keys}},
        {"model": "Full Model (prototype)",
         **{k: pre_recal_metrics.get(k, np.nan) for k in metric_keys}},
    ])

    main_table_df = pd.DataFrame(main_table_rows)
    main_table_df.to_csv(cfg.output_dir / "paper_main_table.csv", index=False)
    print("\n  Paper main table (core models):")
    display_cols = ["model", "c_index", "ibs"] + \
                   [f"brier_{h}h" for h in cfg.main_horizons]
    avail_display = [c for c in display_cols if c in main_table_df.columns]
    print(main_table_df[avail_display].to_string(index=False))

    # Narrative recommendation
    print("\n  ── Narrative recommendation ──")
    valid_rows = main_table_df.dropna(subset=["c_index"])
    best_ci_row = valid_rows.loc[valid_rows["c_index"].idxmax()]
    best_ibs_row = valid_rows.loc[valid_rows["ibs"].idxmin()]
    print(f"    Best C-index: {best_ci_row['model']} ({best_ci_row['c_index']:.4f})")
    print(f"    Best IBS:     {best_ibs_row['model']} ({best_ibs_row['ibs']:.4f})")

    lean_names = {"AFT + Cox", "AFT + Cox + Simple", "AFT Only"}
    lean_rows = valid_rows[valid_rows["model"].isin(lean_names)]
    if len(lean_rows) > 0:
        best_lean = lean_rows.loc[lean_rows["c_index"].idxmax()]
        full_ci = pre_recal_metrics["c_index"]
        lean_ci = best_lean["c_index"]
        if lean_ci >= full_ci - 0.005:
            print(f"    → RECOMMENDATION: '{best_lean['model']}' "
                  f"(C={lean_ci:.4f}) is competitive with Full Model "
                  f"(C={full_ci:.4f}). Consider as practical main model.")
        else:
            print(f"    → NOTE: Full Model (C={full_ci:.4f}) exceeds "
                  f"best lean ({best_lean['model']}, C={lean_ci:.4f}) by "
                  f"{full_ci - lean_ci:.4f}.")

    if not np.isnan(rsf_metrics.get("c_index", np.nan)):
        rsf_ci = rsf_metrics["c_index"]
        rsf_ibs = rsf_metrics["ibs"]
        full_ci = pre_recal_metrics["c_index"]
        full_ibs = pre_recal_metrics["ibs"]
        if rsf_ibs < full_ibs and rsf_ci >= full_ci - 0.01:
            print(f"    → NOTE: RSF (C={rsf_ci:.4f}, IBS={rsf_ibs:.4f}) "
                  f"is a strong practical comparator.")

    # ── Multi-seed stability check ──
    print_header("STEP 5d: Multi-Seed Stability Check (5 seeds)")
    stability_df = multi_seed_stability(
        train_fe, feature_sets, cfg, n_repeats=5)
    stability_df.to_csv(cfg.output_dir / "multi_seed_stability.csv", index=False)
    print(f"\n  Stability results saved to {cfg.output_dir / 'multi_seed_stability.csv'}")

    # ── Grouped CV (if incident-level grouping exists) ──
    print_header("STEP 5e: Grouped CV (incident-level)")
    group_col = detect_group_column(train_fe, cfg)
    if group_col is not None:
        print(f"  Detected group column: '{group_col}' "
              f"({train_fe[group_col].nunique()} unique groups)")
        grouped_df = run_grouped_cv_comparison(
            train_fe, feature_sets, cfg, group_col)
        grouped_df.to_csv(cfg.output_dir / "grouped_cv_results.csv", index=False)
        print(f"\n  Grouped CV results saved.")
        print(grouped_df[["model", "cv_type", "c_index", "ibs"]].to_string(index=False))

        # Compare with standard CV results
        print("\n  ── Grouped vs Standard CV comparison ──")
        standard_models = {"Full Model": pre_recal_metrics,
                           "AFT + Cox + Simple": key_lean_models.get("AFT + Cox + Simple"),
                           "AFT + Cox": key_lean_models.get("AFT + Cox")}
        for _, grow in grouped_df.iterrows():
            mn = grow["model"]
            std_m = standard_models.get(mn)
            if std_m is not None:
                std_ci = std_m.get("c_index", std_m.get("c_index", np.nan))
                grp_ci = grow["c_index"]
                drop = std_ci - grp_ci
                print(f"    {mn}: standard C={std_ci:.4f} → grouped C={grp_ci:.4f} "
                      f"(Δ={drop:+.4f})")
    else:
        print("  No incident-level group column detected in dataset.")
        print("  Columns available:", [c for c in train_fe.columns
                                       if c not in [cfg.target_time, cfg.target_event]])
        print("  → Grouped CV skipped. If your data has a fire incident ID,")
        print("    add it to detect_group_column() candidate patterns.")
        print("  → Currently relying on multi-seed stability check instead.")

    # ── [R5] Fold-Level Metrics (Table S4) ──
    print_header("STEP 5f: Fold-Level Metrics (Supplementary Table S4)")
    fold_metrics_df = compute_fold_level_metrics(
        train_fe, pre_recal_preds, pre_recal_result["oof_risk"], cfg)
    fold_metrics_df.to_csv(cfg.output_dir / "fold_level_metrics.csv", index=False)
    print(f"  Fold-level metrics saved to {cfg.output_dir / 'fold_level_metrics.csv'}")
    print(fold_metrics_df.to_string(index=False))

    # ── [R9] Feature Importance ──
    print_header("STEP 5g: Feature Importance (AFT gain-based)")
    feat_imp_df = compute_feature_importance(train_fe, feature_sets, cfg)
    feat_imp_df.to_csv(cfg.output_dir / "feature_importance.csv", index=False)
    print(f"  Feature importance saved. Top 15:")
    print(feat_imp_df.head(15).to_string(index=False))

    # ── [R4] Decision Utility Metrics ──
    print_header("STEP 5h: Decision Utility & DCA (Supplementary)")
    y_dict_du = {h: payload[f"y{h}"] for h in cfg.main_horizons}
    e_dict_du = {h: payload[f"elig{h}"] for h in cfg.main_horizons}
    decision_df = compute_decision_utility(
        y_dict_du, pre_recal_preds, e_dict_du,
        cfg.main_horizons, cfg.decision_thresholds)
    decision_df.to_csv(cfg.output_dir / "decision_utility.csv", index=False)
    print(f"  Decision utility saved to {cfg.output_dir / 'decision_utility.csv'}")
    print(decision_df.to_string(index=False))

    # DCA for each horizon
    dca_rows = []
    for h in cfg.main_horizons:
        elig = e_dict_du[h]
        yt_h = y_dict_du[h][elig]
        yp_h = pre_recal_preds[h][elig]
        dca_h = decision_curve_analysis(yt_h, yp_h)
        dca_h["horizon"] = f"{h}h"
        dca_rows.append(dca_h)
    dca_df = pd.concat(dca_rows, ignore_index=True)
    dca_df.to_csv(cfg.output_dir / "decision_curve_analysis.csv", index=False)
    print(f"  DCA saved to {cfg.output_dir / 'decision_curve_analysis.csv'}")

    # ── [R6] Lambda Sensitivity Sweep ──
    print_header("STEP 5i: Lambda Sensitivity Sweep (Supplementary Table S2)")
    lambda_df = run_lambda_sweep(train_fe, feature_sets, cfg)
    lambda_df.to_csv(cfg.output_dir / "lambda_sensitivity.csv", index=False)
    print(f"\n  Lambda sensitivity saved to {cfg.output_dir / 'lambda_sensitivity.csv'}")
    print(lambda_df[["lambda", "c_index", "ibs"]].to_string(index=False))

    # ── Bootstrap CI (uses pre-recalibration predictions — primary analysis) ──
    print_header("STEP 6: Bootstrap Confidence Intervals (pre-recalibration)")
    bs = bootstrap_ci(
        {h: payload[f"y{h}"] for h in cfg.main_horizons},
        pre_recal_preds,
        {h: payload[f"elig{h}"] for h in cfg.main_horizons},
        pre_recal_result["oof_risk"],
        payload["pair_i"], payload["pair_j"],
        horizons=cfg.main_horizons,
        n_boot=cfg.n_bootstrap,
        seed=cfg.random_seed,
        y_time=y_time,        # [R1][R2] enables Uno's C and continuous IBS
        y_event=y_event)
    # Primary proxy metrics
    for k in ["c_index", "ibs"] + [f"brier_{h}h" for h in cfg.main_horizons]:
        v = bs[k]
        print(f"  {k:20s}: {np.mean(v):.4f} "
              f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")
    # [R1][R2] Standard survival metrics (when computable)
    for k in ["c_index_uno", "ibs_continuous"]:
        v = bs.get(k, np.array([]))
        if len(v) > 0:
            print(f"  {k:20s}: {np.mean(v):.4f} "
                  f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")

    # ── Calibration summary (PRE-recalibration — primary analysis) ──
    print_header("STEP 7: Calibration Summary (pre-recalibration)")
    cal_rows = []
    for h in cfg.main_horizons:
        elig = payload[f"elig{h}"]
        yt = payload[f"y{h}"][elig]
        yp = pre_recal_preds[h][elig]
        brier_val = float(np.mean((yt - yp) ** 2))
        ece, mce = compute_ece_mce(yt, yp)
        slope, intercept = calibration_slope_intercept(yt, yp)
        cal_rows.append({
            "horizon": f"{h}h", "n_eligible": int(elig.sum()),
            "brier": brier_val, "ECE": ece, "MCE": mce,
            "cal_slope": slope, "cal_intercept": intercept,
        })
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(cfg.output_dir / "calibration_summary.csv", index=False)
    print(cal_df.to_string(index=False))

    # ── MDPI Fire Journal Figures (8 main figures + Graphical Abstract) ──
    print_header("STEP 8: Generating MDPI Fire Figures (600 DPI, TIFF + PNG)")
    y_dict = {h: payload[f"y{h}"] for h in cfg.main_horizons}
    e_dict = {h: payload[f"elig{h}"] for h in cfg.main_horizons}

    print("  Figure 0: Prediction timeline...")
    plot_timeline(save_path=cfg.fig_dir / "Figure_0_timeline.pdf")

    print("  Figure 1: Framework schematic...")
    plot_framework(save_path=cfg.fig_dir / "Figure_1_framework.pdf")

    print("  Figure 2: Reliability diagrams with Wilson CI (pre-recal)...")
    plot_reliability(y_dict, pre_recal_preds, e_dict,
                     cfg.main_horizons,
                     save_path=cfg.fig_dir / "Figure_2_reliability.pdf")

    print("  Figure 3: Calibration summary (Brier + ECE + Slope)...")
    plot_calibration_summary(cal_df, cfg.main_horizons,
                             save_path=cfg.fig_dir / "Figure_3_calibration.pdf")

    print("  Figure S1: Monotonicity verification (supplementary)...")
    plot_monotonicity(pre_recal_preds, list(cfg.main_horizons),
                      save_path=cfg.fig_dir / "Figure_S1_monotonicity.pdf")

    print("  Figure 4: Ablation contribution summary (ΔC-index + ΔIBS)...")
    plot_ablation(ablation_df,
                  save_path=cfg.fig_dir / "Figure_4_ablation.pdf")

    print("  Figure 5: Bootstrap CI distributions (pre-recalibration)...")
    plot_bootstrap_ci(bs, cfg.main_horizons,
                      save_path=cfg.fig_dir / "Figure_5_bootstrap.pdf")

    print("  Figure 6: Prediction distributions + risk stratification...")
    plot_distributions(pre_recal_preds, y_dict, e_dict,
                       cfg.main_horizons,
                       save_path=cfg.fig_dir / "Figure_6_distributions.pdf")

    print("  Figure 7: Baseline comparison (including lean variants)...")
    plot_table = main_table_df.dropna(subset=["c_index", "ibs"]).copy()
    plot_baseline_comparison(plot_table,
                             save_path=cfg.fig_dir / "Figure_7_baselines.pdf")

    print("  Graphical Abstract...")
    plot_graphical_abstract(pre_recal_metrics,
                            save_path=cfg.fig_dir / "Graphical_Abstract.pdf")

    print("  Figure 8: Pre vs post recalibration comparison...")
    plot_recalibration_comparison(recal_comparison_df, cfg.main_horizons,
                                  save_path=cfg.fig_dir / "Figure_8_recalibration.pdf")

    print("  All figures generated (TIFF @ 600 DPI + PNG @ 300 DPI + SVG + PDF).")

    # ── Export metrics ──
    metrics_df = pd.DataFrame([full_metrics])
    metrics_df.to_csv(cfg.output_dir / "full_model_metrics.csv", index=False)

    print_header("Pipeline complete")
    print(f"  Output directory : {cfg.output_dir}")
    print(f"  Figures directory: {cfg.fig_dir}")
    print(f"\n  Generated outputs:")
    _outputs = [
        ("paper_main_table.csv",        "Main results table"),
        ("ablation_results.csv",         "Ablation study (Table S3)"),
        ("baseline_comparison.csv",      "External baseline comparison"),
        ("lambda_sensitivity.csv",       "Stacking lambda sweep (Table S2)"),
        ("fold_level_metrics.csv",       "Per-fold metrics (Table S4)"),
        ("feature_importance.csv",       "AFT gain-based importances"),
        ("decision_utility.csv",         "Utility metrics at operational thresholds"),
        ("decision_curve_analysis.csv",  "DCA net benefit curves"),
        ("practical_variants.csv",       "Lean variant results"),
        ("multi_seed_stability.csv",     "5-seed stability check"),
        ("full_model_metrics.csv",       "Full model OOF metrics"),
        ("calibration_summary.csv",      "Calibration summary"),
    ]
    for fname, desc in _outputs:
        print(f"    {fname:<38s} {desc}")
    if group_col is not None:
        print(f"    {'grouped_cv_results.csv':<38s} Incident-level grouped CV")
    print(f"\n  Pre-fusion monotonicity violations:")
    print(f"    12h→24h: {prefusion_stats.get('violation_rate_12_24', 'n/a')}")
    print(f"    24h→48h: {prefusion_stats.get('violation_rate_24_48', 'n/a')}")


if __name__ == "__main__":
    main()