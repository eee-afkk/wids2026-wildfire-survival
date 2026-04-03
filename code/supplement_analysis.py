# -*- coding: utf-8 -*-
"""
Supplementary Code S1
=====================
supplement_code_s1.py

Supplementary analysis code for:
  "A Survival-Probability Fusion Prototype for Censor-Aware
   Multi-Horizon Wildfire Threat Forecasting and Decision-Utility Evaluation"
  Submitted to Fire (MDPI).

This script implements all supplementary analyses referenced in the manuscript:
  [P1] Incident-level grouped cross-validation using NIFC WFIGS fire identifiers
  [P2] GroupKFold vs. standard StratifiedKFold comparison (optimistic bias quantification)
  [P3] Temporal blocked cross-validation (Leave-One-Month-Out)
  [P4] Informative censoring sensitivity analysis (worst/best-case + Aalen-Johansen)
  [P5] TRIPOD-style data quality report (missing rates, distributions, KM curve)
  [P6] All outputs as journal-ready CSV tables + 300 DPI white-background PDF figures

External data:
  NIFC WFIGS Interagency Fire Perimeters (public, no login required)
  Download: https://data-nifc.opendata.arcgis.com/

Usage:
    python supplement_code_s1.py --train path/to/train.csv --nifc WFIGS.csv --output ./output

    NOTE: Real WiDS 2026 training data is REQUIRED.
          If train.csv is not found, the script raises FileNotFoundError.

Dependencies:
    numpy, pandas, scipy, scikit-learn, lifelines, xgboost, matplotlib

Reproducibility:
    All random seeds are fixed (seed = 42).
    Python 3.10+, scikit-learn 1.3+, XGBoost 2.0+, lifelines 0.28+.
"""


from __future__ import annotations

# (Platform-specific console encoding configuration omitted)
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# -- Matplotlib font: use Arial/Helvetica for SCI journal figures -------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines import AalenJohansenFitter, KaplanMeierFitter
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
HORIZONS = [12, 24, 48]
SEED = 42
rng = np.random.default_rng(SEED)

LOCAL_BASE = Path(
    # NOTE: Update this path to match your actual local folder name.
    # The folder name shown here is an English translation of the original.
    # Your actual folder may still have the Chinese name; just replace the string below.
    r"./data"  # Update to your local data directory
)
NIFC_FILENAME = "WFIGS_Interagency_Perimeters_-848118526729381764.csv"
NIFC_LOCAL_PATH = LOCAL_BASE / NIFC_FILENAME

WESTERN_STATES = [
    "US-CA", "US-OR", "US-WA", "US-NV", "US-AZ",
    "US-ID", "US-MT", "US-CO", "US-UT", "US-NM",
]

MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ============================================================================
# Section 1  NIFC data loading
# ============================================================================

def load_nifc_data(
    csv_path: Path,
    min_acres: float = 10.0,
    states: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the locally downloaded NIFC WFIGS CSV and return a clean DataFrame.

    Relevant columns in the raw file:
      attr_UniqueFireIdentifier  — national unique fire ID, e.g. '2022-WAPCS-000256'
      attr_IncidentName          — human-readable fire name
      attr_InitialLatitude/Lon   — point of origin coordinates
      attr_POOState              — state code (format 'US-CA')
      attr_IncidentSize          — final fire size (acres)
      attr_FireDiscoveryDateTime — ignition timestamp (string)
      attr_ContainmentDateTime   — containment timestamp (string)
      attr_FireCause             — cause category
      attr_POOCounty             — county
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NIFC WFIGS file not found: {csv_path}\n"
            f"Please verify the NIFC CSV has been downloaded to this path."
        )

    if verbose:
        print(f"  [NIFC] Reading: {csv_path.name} ...")

    df = pd.read_csv(csv_path, low_memory=False)
    if verbose:
        print(f"  [NIFC] Raw records: {len(df):,}  columns: {df.shape[1]}")

    # State filter
    if states is None:
        states = WESTERN_STATES
    df = df[df["attr_POOState"].isin(states)].copy()

    # Minimum size filter
    df = df[df["attr_IncidentSize"].fillna(0) >= min_acres].copy()

    # Coordinate filter
    df = df.dropna(subset=["attr_InitialLatitude", "attr_InitialLongitude"])
    df["lat"] = pd.to_numeric(df["attr_InitialLatitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["attr_InitialLongitude"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # Sanity-check bounding box (contiguous western US)
    df = df[df["lat"].between(25, 50) & df["lon"].between(-130, -100)]

    # Parse timestamps
    df["discovery_dt"] = pd.to_datetime(
        df["attr_FireDiscoveryDateTime"], errors="coerce"
    )
    df["containment_dt"] = pd.to_datetime(
        df["attr_ContainmentDateTime"], errors="coerce"
    )

    # Rename to tidy output schema
    rename = {
        "attr_UniqueFireIdentifier": "fire_id",
        "attr_IncidentName": "incident_name",
        "attr_POOState": "state",
        "attr_POOCounty": "county",
        "attr_IncidentSize": "final_acres",
        "attr_FireCause": "fire_cause",
    }
    keep = {k: v for k, v in rename.items() if k in df.columns}
    df = df[list(keep.keys()) + ["lat", "lon", "discovery_dt", "containment_dt"]]
    df = df.rename(columns=keep)

    # Derived fields
    df["duration_days"] = (
        (df["containment_dt"] - df["discovery_dt"])
        .dt.total_seconds() / 86400
    ).clip(lower=0)
    df["year"] = df["discovery_dt"].dt.year
    df["month"] = df["discovery_dt"].dt.month

    if verbose:
        print(f"  [NIFC] After filtering: {len(df):,} records "
              f"(western US, >= {min_acres} acres, with coordinates)")
        print(f"  [NIFC] Year range: {df['year'].min()}–{df['year'].max()}")
        print(f"  [NIFC] Unique fire IDs: {df['fire_id'].nunique():,}")

    return df.reset_index(drop=True)


# ============================================================================
# Section 2  Proxy incident-ID assignment
# ============================================================================

def _haversine_km(
    lat1: float, lon1: float,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Haversine great-circle distance (km) from one point to an array of points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _cluster_fallback(df: pd.DataFrame, verbose: bool = True) -> np.ndarray:
    """
    Feature-space hierarchical clustering fallback when coordinates are absent.
    Produces proxy incident groups that approximate 'same fire event' structure.

    For 221 observations and 5-fold GroupKFold, we need enough groups
    (>=25 ideally) so each fold has >=5 groups for stable estimation.
    """
    candidates = [
        # Priority: distance & speed features (most physically meaningful for grouping)
        "dist_min_ci_0_5h", "closing_speed_abs_m_per_h",
        "area_growth_rate_ha_per_h", "centroid_speed_m_per_h",
        "alignment_abs", "radial_growth_rate_m_per_h",
        "event_start_month", "along_track_speed",
        "area_first_ha", "spread_bearing_deg",
        "dist_change_ci_0_5h", "projected_advance_m",
        # fallback: old names from synthetic schema
        "min_distance_to_perimeter", "dist_log", "closing_speed_abs",
        "centroid_speed", "area_growth_rate", "month_sin", "month_cos",
        "urgency", "eta_hours",
    ]
    available = [c for c in candidates if c in df.columns]
    if not available:
        available = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
    if not available:
        return np.zeros(len(df), dtype=int)

    X = df[available].copy().fillna(df[available].median())
    X = StandardScaler().fit_transform(X.values)

    # Target more groups for stable GroupKFold (5 folds → need >= 25 groups)
    n_target = max(25, min(45, len(df) // 5))
    Z = linkage(X, method="ward")
    t_cut = Z[-(n_target - 1), 2] if len(Z) >= n_target else Z[-1, 2]
    labels = fcluster(Z, t=t_cut, criterion="distance") - 1

    n_groups = len(np.unique(labels))
    grp_sizes = pd.Series(labels).value_counts()
    if verbose:
        print(f"  [Cluster] Hierarchical clustering: "
              f"{n_groups} proxy incident groups "
              f"(target={n_target})")
        print(f"  [Cluster] Group sizes: min={grp_sizes.min()}, "
              f"median={int(grp_sizes.median())}, max={grp_sizes.max()}")
        print(f"  [Cluster] Features used for grouping: {available}")
    return labels.astype(int)


def assign_incident_ids_from_nifc(
    wids_df: pd.DataFrame,
    nifc_df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    wids_month_col: Optional[str] = "start_month",
    max_dist_km: float = 80.0,
    max_month_diff: int = 2,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match WiDS fire-zone pair observations to real NIFC fire incidents.

    Matching logic:
      1. If WiDS data has coordinate columns → nearest-neighbour in (lat, lon)
         constrained by spatial threshold (max_dist_km) and optional month window.
      2. If no coordinates → feature-space clustering fallback.

    Returns
    -------
    incident_ids   : shape (n,) int, group label per observation
    matched_fire_id: shape (n,) str, NIFC UniqueFireIdentifier (empty if unmatched)
    """
    n = len(wids_df)
    incident_ids = np.full(n, -1, dtype=int)
    matched_fire_id = np.array([""] * n, dtype=object)

    if nifc_df.empty:
        if verbose:
            print("  [Match] NIFC data empty — using clustering fallback ...")
        incident_ids = _cluster_fallback(wids_df, verbose)
        return incident_ids, matched_fire_id

    has_latlon = (
        lat_col in wids_df.columns
        and lon_col in wids_df.columns
        and wids_df[lat_col].notna().mean() > 0.5
    )

    if has_latlon:
        if verbose:
            print(f"  [Match] Coordinate-based matching "
                  f"(threshold = {max_dist_km} km) ...")
        nifc_lats = nifc_df["lat"].values
        nifc_lons = nifc_df["lon"].values
        nifc_months = nifc_df["month"].values if "month" in nifc_df.columns \
                      else np.zeros(len(nifc_df))
        nifc_ids_arr = nifc_df["fire_id"].values

        wids_month = (
            wids_df[wids_month_col].values
            if wids_month_col and wids_month_col in wids_df.columns
            else np.zeros(n)
        )

        for i in range(n):
            lat_i = wids_df[lat_col].iloc[i]
            lon_i = wids_df[lon_col].iloc[i]
            if pd.isna(lat_i) or pd.isna(lon_i):
                continue
            dists = _haversine_km(lat_i, lon_i, nifc_lats, nifc_lons)

            if wids_month[i] > 0 and np.any(nifc_months > 0):
                mdiff = np.abs(nifc_months - wids_month[i])
                mdiff = np.minimum(mdiff, 12 - mdiff)   # wrap-around
                valid = (dists <= max_dist_km) & (mdiff <= max_month_diff)
            else:
                valid = dists <= max_dist_km

            if valid.any():
                best = np.argmin(np.where(valid, dists, np.inf))
                incident_ids[i] = best
                matched_fire_id[i] = nifc_ids_arr[best]

        n_matched = (incident_ids >= 0).sum()
        if verbose:
            print(f"  [Match] Matched: {n_matched}/{n} "
                  f"({100 * n_matched / n:.1f}%)")

        # Unmatched observations: assign via clustering
        unmatched = incident_ids < 0
        if unmatched.any():
            cl = _cluster_fallback(
                wids_df[unmatched].reset_index(drop=True), verbose=False
            )
            incident_ids[unmatched] = cl + len(nifc_df)
    else:
        if verbose:
            print("  [Match] No coordinate columns found — using clustering fallback ...")
        incident_ids = _cluster_fallback(wids_df, verbose)

    n_groups = len(np.unique(incident_ids))
    if verbose:
        print(f"  [Match] Final incident groups: {n_groups}")
    return incident_ids, matched_fire_id


# ============================================================================
# Section 3  Grouped CV comparison  [Priority P1 — highest]
# ============================================================================

def run_grouped_cv_comparison(
    X: np.ndarray,
    y_event: np.ndarray,
    t_obs: np.ndarray,
    incident_ids: np.ndarray,
    y_horizons: Dict[int, Tuple[np.ndarray, np.ndarray]],
    n_outer: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare two cross-validation strategies on a Lean AFT+Simple model:
      stratified_cv : standard StratifiedKFold (replicates the paper's setup)
      grouped_cv    : GroupKFold by NIFC incident ID (leak-controlled)

    A meaningful drop in C-index under grouped_cv indicates optimistic bias
    from intra-incident correlation in the standard CV and should be reported.
    """
    results = []
    strat_labels = _make_strat_labels(y_event, t_obs)

    n_groups = len(np.unique(incident_ids))
    actual_folds = min(n_outer, n_groups)
    if actual_folds < n_outer and verbose:
        print(f"  [GroupedCV] Note: n_groups={n_groups} < target folds={n_outer}; "
              f"using {actual_folds} folds")

    strategies = {
        "stratified_cv": StratifiedKFold(n_splits=n_outer, shuffle=True,
                                          random_state=SEED),
        "grouped_cv": GroupKFold(n_splits=actual_folds),
    }

    for strat_name, cv in strategies.items():
        if verbose:
            print(f"\n  [GroupedCV] Running {strat_name} ...")

        splits = (
            list(cv.split(X, strat_labels))
            if strat_name == "stratified_cv"
            else list(cv.split(X, y_event, groups=incident_ids))
        )

        for fold_i, (tr_idx, va_idx) in enumerate(splits):
            bst = _fit_aft_simple(X[tr_idx], y_event[tr_idx], t_obs[tr_idx])
            log_t_pred = bst.predict(xgb.DMatrix(X[va_idx]))
            risk = -log_t_pred

            c_idx = _concordance(risk, y_event[va_idx], t_obs[va_idx])

            briers = []
            for h in HORIZONS:
                y_h, elig_h = y_horizons[h]
                elig_va = elig_h[va_idx]
                if elig_va.sum() < 5:
                    continue
                p_h = _aft_cdf(log_t_pred, h)
                briers.append(
                    float(np.mean((p_h[elig_va] - y_h[va_idx][elig_va]) ** 2))
                )

            mean_brier = float(np.mean(briers)) if briers else np.nan
            n_grp_va = len(np.unique(incident_ids[va_idx]))

            results.append({
                "split_strategy": strat_name,
                "fold": fold_i + 1,
                "n_val": len(va_idx),
                "n_incident_groups_val": n_grp_va,
                "c_index": round(c_idx, 4),
                "mean_brier": round(mean_brier, 4),
            })
            if verbose:
                print(f"    Fold {fold_i+1}: C={c_idx:.4f}  "
                      f"IBS={mean_brier:.4f}  n_groups={n_grp_va}")

        sub = [r for r in results if r["split_strategy"] == strat_name]
        c_vals = [r["c_index"] for r in sub]
        b_vals = [r["mean_brier"] for r in sub if not np.isnan(r["mean_brier"])]
        if verbose:
            print(f"  [{strat_name}] "
                  f"C = {np.mean(c_vals):.4f} +/- {np.std(c_vals):.4f}  "
                  f"IBS = {np.mean(b_vals):.4f} +/- {np.std(b_vals):.4f}")

    return pd.DataFrame(results)


# ============================================================================
# Section 4  Temporal blocked CV  [Priority P6]
# ============================================================================

def run_temporal_blocked_cv(
    X: np.ndarray,
    y_event: np.ndarray,
    t_obs: np.ndarray,
    month_col: np.ndarray,
    y_horizons: Dict[int, Tuple[np.ndarray, np.ndarray]],
    min_val_size: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Leave-One-Month-Out temporal blocked cross-validation.
    Conservative alternative to incident-level grouping when incident IDs
    are unavailable; prevents future data from leaking into training.

    Months with fewer than min_val_size observations are skipped
    (too few for stable metric estimation).
    """
    months = np.unique(month_col[~np.isnan(month_col)]).astype(int)
    if verbose:
        print(f"\n  [TemporalCV] Months present: "
              f"{[MONTH_ABBR.get(m, str(m)) for m in sorted(months)]}")
        # Show per-month counts
        for m in sorted(months):
            n_m = int((month_col == m).sum())
            status = "" if n_m >= min_val_size else f"  (SKIPPED: n < {min_val_size})"
            print(f"    {MONTH_ABBR.get(m, str(m)):3s}: n={n_m:3d}{status}")

    rows = []
    for m in sorted(months):
        va_mask = month_col == m
        tr_mask = ~va_mask
        n_va = int(va_mask.sum())
        n_tr = int(tr_mask.sum())
        if n_va < min_val_size or n_tr < 20:
            continue

        bst = _fit_aft_simple(X[tr_mask], y_event[tr_mask], t_obs[tr_mask])
        log_t_pred = bst.predict(xgb.DMatrix(X[va_mask]))
        risk = -log_t_pred
        c_idx = _concordance(risk, y_event[va_mask], t_obs[va_mask])

        briers = []
        for h in HORIZONS:
            y_h, elig_h = y_horizons[h]
            elig_va = elig_h[va_mask]
            if elig_va.sum() < 3:
                continue
            p_h = _aft_cdf(log_t_pred, h)
            briers.append(
                float(np.mean((p_h[elig_va] - y_h[va_mask][elig_va]) ** 2))
            )

        rows.append({
            "hold_out_month": int(m),
            "month_label": MONTH_ABBR.get(m, str(m)),
            "n_val": n_va,
            "c_index": round(c_idx, 4),
            "mean_brier": round(float(np.mean(briers)) if briers else np.nan, 4),
        })
        if verbose:
            print(f"    {MONTH_ABBR.get(m, str(m)):3s} "
                  f"(n={n_va:3d}): "
                  f"C={c_idx:.4f}  IBS={rows[-1]['mean_brier']:.4f}")

    if verbose and rows:
        c_vals = [r["c_index"] for r in rows]
        n_vals = [r["n_val"] for r in rows]
        # Weighted mean by sample size
        wt_mean = np.average(c_vals, weights=n_vals)
        print(f"\n  [TemporalCV] Unweighted mean C = {np.mean(c_vals):.4f}")
        print(f"  [TemporalCV] N-weighted mean C = {wt_mean:.4f}")

    return pd.DataFrame(rows)


# ============================================================================
# Section 5  Informative censoring sensitivity analysis  [Priority P4]
# ============================================================================

def censoring_sensitivity_analysis(
    t_obs: np.ndarray,
    y_event: np.ndarray,
    y_horizons: Dict[int, Tuple[np.ndarray, np.ndarray]],
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Two-component sensitivity analysis for informative censoring.

    (A) Worst / best-case event-rate bounding
        For observations censored before horizon h (unknown true status),
        assume all would have been events (worst-case) or none (best-case).
        The resulting range quantifies the sensitivity of event rates to the
        non-informative censoring assumption.

    (B) Aalen-Johansen competing-risk CIF
        Proxy 'containment-censored' observations (censoring time < median
        censoring time) as a competing event (type 2). Compare the KM-derived
        P(T<=h) against the AJ cause-specific CIF for the primary event.
        A non-trivial gap suggests informative censoring may bias survival
        model estimates.
    """
    results = {}

    # -- (A) Worst / best-case bounding --------------------------------------
    if verbose:
        print("\n  [CensorSens] (A) Worst/best-case event-rate bounding ...")
    rows_A = []
    for h in HORIZONS:
        y_h, elig_h = y_horizons[h]
        cens_before = (y_event == 0) & (t_obs < h)
        n_cens = int(cens_before.sum())
        y_elig = y_h[elig_h]
        n_elig = int(elig_h.sum())
        base_rate = float(y_elig.mean())
        worst = float(np.append(y_elig, np.ones(n_cens)).mean())
        best = float(np.append(y_elig, np.zeros(n_cens)).mean())
        rows_A.append({
            "horizon": f"{h}h",
            "n_eligible_base": n_elig,
            "n_censored_before_h": n_cens,
            "base_event_rate": round(base_rate, 4),
            "worst_case_rate": round(worst, 4),
            "best_case_rate": round(best, 4),
            "sensitivity_range": round(worst - best, 4),
        })
        if verbose:
            print(f"    {h}h  base={base_rate:.3f}  "
                  f"worst={worst:.3f}  best={best:.3f}  "
                  f"range={worst - best:.3f}  "
                  f"(n_censored_before={n_cens})")
    results["bounding"] = pd.DataFrame(rows_A)

    # -- (B) Aalen-Johansen competing-risk CIF -------------------------------
    if verbose:
        print("  [CensorSens] (B) Aalen-Johansen competing-risk CIF ...")

    med_cens = float(np.median(t_obs[y_event == 0]))
    # Proxy: observations censored before the median censoring time are
    # labelled as 'containment-censored' (competing event type = 2)
    event_type = np.where(
        y_event == 1, 1,
        np.where((y_event == 0) & (t_obs < med_cens), 2, 0)
    )
    n_competing = int((event_type == 2).sum())

    kmf = KaplanMeierFitter()
    kmf.fit(t_obs, event_observed=y_event)

    aj_ok = False
    try:
        ajf = AalenJohansenFitter(calculate_variance=False)
        ajf.fit(t_obs, event_type, event_of_interest=1)
        aj_ok = True
    except Exception as exc:
        if verbose:
            print(f"    AJ fitting failed: {exc}")

    rows_B = []
    for h in HORIZONS:
        km_cif = float(1 - kmf.survival_function_at_times([h]).values[0])
        if aj_ok:
            try:
                aj_cif = float(ajf.cumulative_density_at_times([h]).values[0])
            except Exception:
                aj_cif = np.nan
        else:
            aj_cif = np.nan
        delta = round(aj_cif - km_cif, 4) if not np.isnan(aj_cif) else np.nan
        rows_B.append({
            "horizon": f"{h}h",
            "km_cif": round(km_cif, 4),
            "aj_cif_event1": round(aj_cif, 4) if not np.isnan(aj_cif) else np.nan,
            "delta_aj_minus_km": delta,
            "n_competing_events_proxy": n_competing,
            "competing_proxy_threshold_h": round(med_cens, 1),
        })
        if aj_ok and not np.isnan(aj_cif) and verbose:
            print(f"    {h}h  KM={km_cif:.4f}  "
                  f"AJ={aj_cif:.4f}  delta={delta:+.4f}")

    results["competing_risk"] = pd.DataFrame(rows_B)
    return results


# ============================================================================
# Section 6  TRIPOD-style data quality report  [Priority P5]
# ============================================================================

def generate_data_quality_report(
    df_raw: pd.DataFrame,
    t_obs: np.ndarray,
    y_event: np.ndarray,
    y_horizons: Dict[int, Tuple[np.ndarray, np.ndarray]],
    fig_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    TRIPOD+AI-compliant data quality summary:
      - Missing rate per variable
      - Kaplan-Meier overall survival curve with horizon markers
      - Event-time vs censoring-time density overlay
      - Eligible sample size and event rate per prediction horizon
    """
    if verbose:
        print("\n  [DataQuality] Generating TRIPOD-style data quality report ...")

    miss = df_raw.isnull().sum()
    miss_pct = (miss / len(df_raw) * 100).round(2)
    miss_df = pd.DataFrame({
        "variable": miss.index,
        "missing_count": miss.values,
        "missing_pct": miss_pct.values,
        "dtype": df_raw.dtypes.astype(str).values,
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    # -- Figure 1: Missing rates ---------------------------------------------
    miss_pos = miss_df[miss_df["missing_pct"] > 0].head(25)
    if len(miss_pos) > 0:
        fig, ax = plt.subplots(figsize=(8, max(3, len(miss_pos) * 0.32)))
        ax.barh(miss_pos["variable"], miss_pos["missing_pct"],
                color="#4878CF", edgecolor="white", linewidth=0.4)
        ax.axvline(5, color="red", ls="--", lw=0.8, label="5% threshold")
        ax.set_xlabel("Missing rate (%)")
        ax.set_title("Missing Data Rate per Variable")
        ax.legend(fontsize=7)
        plt.tight_layout()
        _savefig(fig, fig_dir / "figS_missing_rates.pdf")

    # -- Figure 2: KM curve + time distributions -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    kmf = KaplanMeierFitter()
    kmf.fit(t_obs, event_observed=y_event)
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2166AC",
                               label=f"All observations (n={len(t_obs)})")
    colors_h = ["#D6604D", "#F4A582", "#92C5DE"]
    for h, ls, col in zip(HORIZONS, ["--", "-.", ":"], colors_h):
        ax.axvline(h, color=col, ls=ls, lw=1.0, label=f"{h} h horizon")
    ax.set_xlabel("Time from prediction origin (h)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title("Kaplan-Meier Overall Survival Function")
    ax.legend(fontsize=7)

    ax = axes[1]
    t_ev = t_obs[y_event == 1]
    t_ce = t_obs[y_event == 0]
    ax.hist(t_ev, bins=25, alpha=0.75, color="#D6604D", density=True,
            label=f"Event times (n={len(t_ev)})")
    ax.hist(t_ce, bins=25, alpha=0.55, color="#4393C3", density=True,
            label=f"Censoring times (n={len(t_ce)})")
    for h in HORIZONS:
        ax.axvline(h, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Observed time (h)")
    ax.set_ylabel("Density")
    ax.set_title("Event-Time vs Censoring-Time Distributions")
    ax.legend(fontsize=7)
    plt.tight_layout()
    _savefig(fig, fig_dir / "figS_km_time_dist.pdf")

    # -- Figure 3: Horizon eligibility summary --------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    h_labels, elig_ns, ev_ns, rates = [], [], [], []
    for h in HORIZONS:
        y_h, elig_h = y_horizons[h]
        h_labels.append(f"{h} h")
        elig_ns.append(int(elig_h.sum()))
        ev_ns.append(int(y_h[elig_h].sum()))
        rates.append(float(y_h[elig_h].mean() * 100))

    x = np.arange(len(h_labels))
    w = 0.35
    ax.bar(x - w/2, elig_ns, width=w, color="#4878CF", alpha=0.85,
           label="Eligible n")
    ax.bar(x + w/2, ev_ns, width=w, color="#D6604D", alpha=0.85,
           label="Events (positives)")
    ax2 = ax.twinx()
    ax2.plot(x, rates, "ko-", ms=5, label="Event rate (%)")
    for xi, r in zip(x, rates):
        ax2.text(xi, r + 0.5, f"{r:.1f}%", ha="center", fontsize=7)
    ax2.set_ylabel("Event rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Count")
    ax.set_title("Eligible Sample Size and Event Rate per Horizon")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper left")
    plt.tight_layout()
    _savefig(fig, fig_dir / "figS_horizon_eligibility.pdf")

    if verbose:
        n_miss = (miss_df["missing_pct"] > 0).sum()
        print(f"    Variables with missing data: {n_miss}")
        if n_miss == 0:
            print("    NOTE: All features are complete (no missing values).")
            print("    The paper notes median/mode imputation was applied within")
            print("    each outer training fold; the pre-imputed data was already")
            print("    clean in this dataset version.")
        # Additional distribution summary
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
        n_numeric = len(numeric_cols)
        print(f"    Numeric features: {n_numeric}")
        print(f"    Total observations: {len(df_raw)}")
        print(f"    Event rate: {y_event.mean():.1%} ({y_event.sum()}/{len(y_event)})")
    return miss_df


# ============================================================================
# Section 7  NIFC geographic context figure (Supplementary)
# ============================================================================

def plot_nifc_context(
    nifc_df: pd.DataFrame, fig_dir: Path, verbose: bool = True
) -> None:
    """
    Figure for the supplementary materials:
    Geographic distribution of NIFC fire incidents used for incident-ID
    assignment, alongside annual event counts for the western US.
    """
    if nifc_df.empty:
        return

    df_p = nifc_df[
        nifc_df["lon"].between(-126, -102) &
        nifc_df["lat"].between(31, 50)
    ]
    if df_p.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter map
    ax = axes[0]
    acres = df_p["final_acres"].clip(upper=50000).fillna(100)
    sizes = np.sqrt(acres) * 0.8
    sc = ax.scatter(
        df_p["lon"], df_p["lat"],
        s=sizes, alpha=0.4,
        c=df_p["year"] if "year" in df_p.columns else None,
        cmap="YlOrRd", edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Ignition year")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(
        f"NIFC WFIGS Fire Incidents — Western US (n = {len(df_p):,})\n"
        "Bubble size proportional to sqrt(final area); color = year"
    )
    ax.text(0.02, 0.02, "Source: NIFC WFIGS public open data",
            transform=ax.transAxes, fontsize=6, style="italic", color="gray")

    # Annual counts
    ax = axes[1]
    if "year" in df_p.columns:
        yr_cnt = df_p.groupby("year").size()
        ax.bar(yr_cnt.index, yr_cnt.values, color="#4878CF", alpha=0.85)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of incidents")
        ax.set_title(
            "Annual Wildfire Incidents — Western US\n"
            "(>= 10 acres, with point-of-origin coordinates)"
        )
        for yr, cnt in yr_cnt.items():
            ax.text(yr, cnt + 3, str(cnt), ha="center", fontsize=7)

    plt.tight_layout()
    _savefig(fig, fig_dir / "figS_nifc_geographic_context.pdf")
    if verbose:
        print(f"  [NIFC] Geographic context figure saved "
              f"({len(df_p):,} western-US records).")


# ============================================================================
# Section 8  Result figures
# ============================================================================

def plot_grouped_cv_comparison(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Boxplot comparison of C-index and IBS proxy:
    Standard stratified CV vs incident-grouped CV.
    Figure for Supplementary Materials (Table S_new).
    """
    if df.empty:
        return

    strats = list(df["split_strategy"].unique())
    label_map = {
        "stratified_cv": "Standard\nStratified CV",
        "grouped_cv": "Proxy Incident-Grouped CV\n(feature-space clustering)",
    }
    color_map = {"stratified_cv": "#4878CF", "grouped_cv": "#D6604D"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for col_i, (metric, title, direction) in enumerate([
        ("c_index",    "Discrimination (C-index)",     "higher is better"),
        ("mean_brier", "Probability error (IBS proxy)", "lower is better"),
    ]):
        ax = axes[col_i]
        data_list = [
            df[df["split_strategy"] == s][metric].dropna().values
            for s in strats
        ]
        labels_list = [label_map.get(s, s) for s in strats]

        bp = ax.boxplot(
            data_list, patch_artist=True, widths=0.45,
            medianprops=dict(color="black", lw=2),
        )
        for patch, s in zip(bp["boxes"], strats):
            patch.set_facecolor(color_map.get(s, "gray"))
            patch.set_alpha(0.7)

        for k, (dat, s) in enumerate(zip(data_list, strats)):
            jit = rng.uniform(-0.10, 0.10, len(dat))
            ax.scatter(np.ones(len(dat)) * (k + 1) + jit, dat,
                       color=color_map.get(s, "gray"), s=22, zorder=5, alpha=0.85)
            if len(dat):
                ax.text(
                    k + 1, np.mean(dat),
                    f"  mean={np.mean(dat):.4f}\n  SD={np.std(dat):.4f}",
                    va="center", fontsize=7,
                )

        ax.set_xticks(range(1, len(strats) + 1))
        ax.set_xticklabels(labels_list)
        ax.set_ylabel("C-index" if metric == "c_index" else "Mean Brier score")
        ax.set_title(title)
        ax.text(0.97, 0.03, direction, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7,
                style="italic", color="gray")

    fig.suptitle(
        "Cross-Validation Strategy Comparison: "
        "Standard CV vs Incident-Grouped CV\n"
        "(Proxy grouping via feature-space clustering)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    _savefig(fig, fig_dir / "figS_grouped_cv_comparison.pdf")


def plot_temporal_cv(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Bar chart of per-month C-index and IBS for Leave-One-Month-Out CV.
    """
    if df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    months = df["hold_out_month"].values
    x_labels = [MONTH_ABBR.get(m, str(m)) for m in months]

    for ax, metric, color, ylabel, direction in [
        (axes[0], "c_index",    "#4878CF", "C-index",         "higher is better"),
        (axes[1], "mean_brier", "#D6604D", "Mean Brier score", "lower is better"),
    ]:
        vals = df[metric].values
        ax.bar(range(len(months)), vals, color=color, alpha=0.8)
        mean_val = np.nanmean(vals)
        ax.axhline(mean_val, color="navy", ls="--", lw=1.0,
                   label=f"Mean = {mean_val:.4f}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.text(0.99, 0.95, direction, transform=ax.transAxes,
                ha="right", va="top", fontsize=7,
                style="italic", color="gray")
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v + 0.003, f"{v:.3f}",
                        ha="center", fontsize=6.5)

    axes[0].set_title("Temporal Blocked CV — Leave-One-Month-Out")
    axes[1].set_xticks(range(len(months)))
    axes[1].set_xticklabels(x_labels)
    axes[1].set_xlabel("Held-out month")

    # Sample size annotations below x-axis
    for i, n_val in enumerate(df["n_val"].values):
        axes[1].text(
            i, -0.005, f"n={n_val}", ha="center", fontsize=6,
            transform=axes[1].get_xaxis_transform(), color="gray",
        )

    plt.tight_layout()
    _savefig(fig, fig_dir / "figS_temporal_blocked_cv.pdf")


def plot_censoring_sensitivity(
    sens: Dict[str, pd.DataFrame], fig_dir: Path
) -> None:
    """
    Two-panel figure:
    (a) Worst/best-case event-rate bounds per horizon.
    (b) Kaplan-Meier vs Aalen-Johansen cumulative incidence.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A
    ax = axes[0]
    bdf = sens.get("bounding", pd.DataFrame())
    if not bdf.empty:
        hors = bdf["horizon"].tolist()
        x = np.arange(len(hors))
        ax.bar(x, bdf["base_event_rate"], width=0.4, color="#4878CF",
               alpha=0.85, label="Base rate (eligible subset)")
        centers = (bdf["worst_case_rate"] + bdf["best_case_rate"]) / 2
        half = (bdf["worst_case_rate"] - bdf["best_case_rate"]) / 2
        ax.errorbar(x, centers, yerr=half.values,
                    fmt="none", ecolor="#D6604D",
                    elinewidth=2.5, capsize=8,
                    label="Worst/best-case range")
        for xi, row in zip(x, bdf.itertuples()):
            ax.text(xi, row.worst_case_rate + 0.012,
                    f"range={row.sensitivity_range:.3f}",
                    ha="center", fontsize=7, color="#D6604D")
        ax.set_xticks(x)
        ax.set_xticklabels(hors)
        ax.set_ylabel("Event rate")
        ax.set_title(
            "(a) Informative Censoring Sensitivity\n"
            "Worst/Best-case Event-Rate Bounds"
        )
        ax.legend(fontsize=7)

    # Panel B
    ax = axes[1]
    cdf = sens.get("competing_risk", pd.DataFrame())
    if (not cdf.empty
            and "km_cif" in cdf.columns
            and "aj_cif_event1" in cdf.columns):
        hors = cdf["horizon"].tolist()
        x = np.arange(len(hors))
        w = 0.35
        ax.bar(x - w/2, cdf["km_cif"], width=w, color="#4878CF",
               alpha=0.85, label="KM-derived P(T <= h)")
        ax.bar(x + w/2, cdf["aj_cif_event1"].fillna(0), width=w,
               color="#D6604D", alpha=0.85,
               label="Aalen-Johansen CIF")
        ax.set_xticks(x)
        ax.set_xticklabels(hors)
        ax.set_ylabel("Cumulative incidence")
        ax.set_title(
            "(b) Competing-Risk Framing\n"
            "KM vs Aalen-Johansen CIF"
        )
        ax.legend(fontsize=7)
        for k, row in cdf.iterrows():
            d = row.get("delta_aj_minus_km")
            if pd.notna(d):
                ymax = max(row.get("km_cif", 0),
                           row.get("aj_cif_event1") or 0)
                ax.text(k, ymax + 0.012,
                        f"delta={d:+.3f}",
                        ha="center", fontsize=7, color="navy")

    plt.tight_layout()
    _savefig(fig, fig_dir / "figS_censoring_sensitivity.pdf")


# ============================================================================
# Section 9  Manuscript summary table
# ============================================================================

def build_summary_table(
    grouped_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    paper_c: float = 0.941,
    paper_ibs: float = 0.041,
) -> pd.DataFrame:
    """
    Table S_new: Validation strategy comparison.
    Suitable for direct inclusion in the supplementary materials.

    NOTE: Lean AFT+Simple results are NOT directly comparable to the
    paper's full model because they use a single AFT (depth=3) vs
    the full multi-module stacking architecture. The purpose of this
    table is to show the RELATIVE difference between standard vs
    grouped/temporal CV, not the absolute model performance.
    """
    rows = [
        {
            "Validation strategy": (
                "Full model — nested stratified CV (Table 2, Uno-C)"
            ),
            "n_folds": 5,
            "C_mean": paper_c, "C_SD": "—",
            "IBS_mean": paper_ibs, "IBS_SD": "—",
            "Delta_C": "ref", "Delta_IBS": "ref",
            "Notes": "Full multi-module architecture; see Table 2 in manuscript",
        }
    ]

    for strat in (["stratified_cv", "grouped_cv"]
                  if not grouped_df.empty else []):
        sub = grouped_df[grouped_df["split_strategy"] == strat]
        c_arr = sub["c_index"].dropna().values
        b_arr = sub["mean_brier"].dropna().values
        c_m = float(np.mean(c_arr)) if len(c_arr) else np.nan
        c_s = float(np.std(c_arr)) if len(c_arr) else np.nan
        b_m = float(np.mean(b_arr)) if len(b_arr) else np.nan
        b_s = float(np.std(b_arr)) if len(b_arr) else np.nan
        label_map = {
            "stratified_cv": (
                "Lean AFT — standard stratified CV (supplementary baseline)"
            ),
            "grouped_cv": (
                "Lean AFT — proxy incident-grouped CV (feature-space clustering)"
            ),
        }
        # For grouped vs standard comparison, show Delta relative to
        # standard CV (same model), not vs paper full model
        ref_c = None
        if strat == "grouped_cv" and not grouped_df.empty:
            std_sub = grouped_df[grouped_df["split_strategy"] == "stratified_cv"]
            if len(std_sub):
                ref_c = float(std_sub["c_index"].mean())
                ref_b = float(std_sub["mean_brier"].mean())

        rows.append({
            "Validation strategy": label_map.get(strat, strat),
            "n_folds": len(sub),
            "C_mean": round(c_m, 4) if not np.isnan(c_m) else "—",
            "C_SD": round(c_s, 4) if not np.isnan(c_s) else "—",
            "IBS_mean": round(b_m, 4) if not np.isnan(b_m) else "—",
            "IBS_SD": round(b_s, 4) if not np.isnan(b_s) else "—",
            "Delta_C": (
                round(c_m - ref_c, 4)
                if ref_c is not None and not np.isnan(c_m)
                else ("ref" if strat == "stratified_cv" else "—")
            ),
            "Delta_IBS": (
                round(b_m - ref_b, 4)
                if ref_c is not None and not np.isnan(b_m)
                else ("ref" if strat == "stratified_cv" else "—")
            ),
            "Notes": (
                "Lean AFT (depth=3, single model); supplementary robustness check"
                if strat == "stratified_cv"
                else "Delta relative to standard CV (same lean model)"
            ),
        })

    if not temporal_df.empty:
        c_m = float(temporal_df["c_index"].mean())
        b_m = float(temporal_df["mean_brier"].mean())
        # N-weighted mean
        n_vals = temporal_df["n_val"].values
        c_wt = float(np.average(temporal_df["c_index"].values, weights=n_vals))
        rows.append({
            "Validation strategy": (
                "Lean AFT — temporal blocked CV (Leave-One-Month-Out)"
            ),
            "n_folds": len(temporal_df),
            "C_mean": round(c_m, 4),
            "C_SD": round(float(temporal_df["c_index"].std()), 4),
            "IBS_mean": round(b_m, 4),
            "IBS_SD": round(float(temporal_df["mean_brier"].std()), 4),
            "Delta_C": "—",
            "Delta_IBS": "—",
            "Notes": (
                f"Conservative temporal block; N-weighted mean C = {c_wt:.4f}; "
                f"months with n < 10 excluded"
            ),
        })

    return pd.DataFrame(rows)


# ============================================================================
# Section 10  Helper functions
# ============================================================================

def build_horizon_targets(
    t_obs: np.ndarray, y_event: np.ndarray,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Build censor-aware binary targets and eligibility masks per horizon.
    Eligible if: event occurred at any time OR censored at t >= h.
    Y_h = 1 if event occurred at T <= h, else 0.
    """
    result = {}
    for h in HORIZONS:
        elig = (y_event == 1) | (t_obs >= h)
        y_h = ((y_event == 1) & (t_obs <= h)).astype(int)
        result[h] = (y_h, elig)
    return result


def _make_strat_labels(y_event: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
    t_bins = pd.cut(t_obs, bins=5, labels=False).astype(str)
    return np.array([f"{e}_{b}" for e, b in zip(y_event, t_bins)])


def _fit_aft_simple(
    X_tr: np.ndarray, y_ev: np.ndarray, t_tr: np.ndarray,
) -> xgb.Booster:
    """Train a lean XGBoost-AFT model (normal distribution, depth=3)."""
    lb = t_tr.astype(float)
    ub = np.where(y_ev == 1, t_tr, np.inf).astype(float)
    dtrain = xgb.DMatrix(X_tr)
    dtrain.set_float_info("label_lower_bound", lb)
    dtrain.set_float_info("label_upper_bound", ub)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "max_depth": 3, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "seed": SEED,
    }
    return xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)


def _aft_cdf(log_t_pred: np.ndarray, h: float) -> np.ndarray:
    """Convert AFT log-time predictions to P(T <= h) via normal CDF."""
    from scipy.stats import norm
    return norm.cdf((np.log(h) - log_t_pred) / 1.0)


def _concordance(
    risk: np.ndarray, ev: np.ndarray, t: np.ndarray,
) -> float:
    """Pairwise concordance index proxy (Harrell-like)."""
    concordant = discordant = 0
    for i in range(len(risk)):
        for j in range(i + 1, len(risk)):
            if ev[i] == 1 and ev[j] == 1 and t[i] != t[j]:
                if (t[i] < t[j]) == (risk[i] > risk[j]):
                    concordant += 1
                else:
                    discordant += 1
            elif ev[i] == 1 and ev[j] == 0 and t[j] >= t[i]:
                concordant += risk[i] > risk[j]
                discordant += risk[i] < risk[j]
            elif ev[j] == 1 and ev[i] == 0 and t[i] >= t[j]:
                concordant += risk[j] > risk[i]
                discordant += risk[j] < risk[i]
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _section(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# (generate_synthetic_data removed — this script now requires real data)


# ============================================================================
# Section 11  Main entry point
# ============================================================================

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _section("Supplementary Improvement Analysis v4 — Fire (MDPI) Submission")
    print(f"  Output directory : {out_dir.resolve()}")
    print(f"  Random seed      : {SEED}")

    # -- Load NIFC data -------------------------------------------------------
    _section("Section 1  Load NIFC WFIGS Data")
    nifc_path = Path(args.nifc) if args.nifc else NIFC_LOCAL_PATH
    nifc_df = load_nifc_data(nifc_path, min_acres=10.0, verbose=True)
    plot_nifc_context(nifc_df, fig_dir, verbose=True)

    # -- Load WiDS training data ----------------------------------------------
    _section("Section 2  Load WiDS Training Data")
    train_path = Path(args.train)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data file not found: {train_path}\n"
            f"Please verify train.csv is in the correct directory."
        )
    print(f"  Reading: {train_path}")
    df_raw = pd.read_csv(train_path, low_memory=False)
    print(f"  Loaded {len(df_raw)} observations, {df_raw.shape[1]} columns")

    # -- Identify outcome columns (real WiDS column names) --------------------
    # time_to_hit_hours = survival time; event = event indicator
    T_COL = "time_to_hit_hours"
    EV_COL = "event"
    MONTH_COL = "event_start_month"

    if T_COL not in df_raw.columns:
        raise KeyError(
            f"Time column not found in training data '{T_COL}'; available columns: {list(df_raw.columns)}"
        )
    if EV_COL not in df_raw.columns:
        raise KeyError(
            f"Event column not found in training data '{EV_COL}'; available columns: {list(df_raw.columns)}"
        )

    t_obs = df_raw[T_COL].values.astype(float)
    y_event = df_raw[EV_COL].values.astype(int)

    print(f"  Events : {y_event.sum()}/{len(y_event)} "
          f"({100 * y_event.mean():.1f}%)")
    print(f"  t_obs  : [{t_obs.min():.1f}, {t_obs.max():.1f}] h")

    # Feature matrix — exclude non-feature columns
    exclude = {
        T_COL, EV_COL, "event_id",
        "lat", "lon", "true_incident_id", "fire_id",
        "proxy_incident_id", "matched_nifc_fire_id",
    }
    feat_cols = [
        c for c in df_raw.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_raw[c])
    ]
    X_df = df_raw[feat_cols].copy()
    for c in X_df.columns:
        X_df[c] = X_df[c].fillna(X_df[c].median())
    X = X_df.values.astype(float)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features used: {feat_cols}")

    y_horizons = build_horizon_targets(t_obs, y_event)
    for h, (y_h, elig) in y_horizons.items():
        print(f"  {h}h: eligible={elig.sum():3d}  "
              f"events={y_h[elig].sum():2d}  "
              f"rate={y_h[elig].mean():.3f}")

    month_col = (
        df_raw[MONTH_COL].values.astype(float)
        if MONTH_COL in df_raw.columns
        else np.full(len(df_raw), np.nan)
    )

    # -- Assign proxy incident IDs --------------------------------------------
    _section("Section 3  Assign NIFC Proxy Incident IDs")
    incident_ids, matched_fire_ids = assign_incident_ids_from_nifc(
        wids_df=df_raw, nifc_df=nifc_df,
        lat_col="lat", lon_col="lon",
        wids_month_col=MONTH_COL,
        max_dist_km=80.0, max_month_diff=2,
        verbose=True,
    )
    df_raw["proxy_incident_id"] = incident_ids
    df_raw["matched_nifc_fire_id"] = matched_fire_ids
    df_raw[["proxy_incident_id", "matched_nifc_fire_id"]].to_csv(
        out_dir / "wids_incident_id_assignments.csv",
        index=False, encoding="utf-8-sig",
    )

    if "true_incident_id" in df_raw.columns:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(
            df_raw["true_incident_id"].values, incident_ids
        )
        print(f"  ARI vs ground-truth incident IDs: {ari:.4f} "
              f"(1.0 = perfect)")

    # -- Grouped CV comparison ------------------------------------------------
    _section("Section 4  Proxy Incident-Grouped CV vs Standard CV  [Priority P1]")
    n_groups = len(np.unique(incident_ids))
    n_folds = min(5, n_groups)
    grouped_df = run_grouped_cv_comparison(
        X=X, y_event=y_event, t_obs=t_obs,
        incident_ids=incident_ids,
        y_horizons=y_horizons,
        n_outer=n_folds, verbose=True,
    )
    grouped_df.to_csv(
        out_dir / "grouped_cv_comparison.csv",
        index=False, encoding="utf-8-sig",
    )
    plot_grouped_cv_comparison(grouped_df, fig_dir)

    # -- Temporal blocked CV --------------------------------------------------
    _section("Section 5  Temporal Blocked CV (Leave-One-Month-Out)  [P6]")
    if not np.all(np.isnan(month_col)):
        temporal_df = run_temporal_blocked_cv(
            X=X, y_event=y_event, t_obs=t_obs,
            month_col=month_col, y_horizons=y_horizons, verbose=True,
        )
        temporal_df.to_csv(
            out_dir / "temporal_blocked_cv.csv",
            index=False, encoding="utf-8-sig",
        )
        plot_temporal_cv(temporal_df, fig_dir)
    else:
        print("  No month column — skipping temporal blocked CV.")
        temporal_df = pd.DataFrame()

    # -- Censoring sensitivity ------------------------------------------------
    _section("Section 6  Informative Censoring Sensitivity  [Priority P4]")
    sens = censoring_sensitivity_analysis(
        t_obs=t_obs, y_event=y_event,
        y_horizons=y_horizons, verbose=True,
    )
    for k, df_s in sens.items():
        df_s.to_csv(
            out_dir / f"censoring_sensitivity_{k}.csv",
            index=False, encoding="utf-8-sig",
        )
    plot_censoring_sensitivity(sens, fig_dir)

    # -- Data quality report --------------------------------------------------
    _section("Section 7  TRIPOD-Style Data Quality Report  [Priority P5]")
    miss_df = generate_data_quality_report(
        df_raw=df_raw, t_obs=t_obs, y_event=y_event,
        y_horizons=y_horizons, fig_dir=fig_dir, verbose=True,
    )
    miss_df.to_csv(
        out_dir / "data_quality_missing_rates.csv",
        index=False, encoding="utf-8-sig",
    )

    # -- Summary table --------------------------------------------------------
    _section("Section 8  Validation Strategy Summary Table (Supplementary)")
    summary = build_summary_table(grouped_df, temporal_df)
    summary.to_csv(
        out_dir / "validation_strategy_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    display_cols = [
        "Validation strategy", "C_mean", "C_SD",
        "IBS_mean", "IBS_SD", "Delta_C", "Delta_IBS",
    ]
    print(summary[display_cols].to_string(index=False))

    # -- File listing ---------------------------------------------------------
    _section("DONE — Output file listing")
    for f in sorted(out_dir.glob("**/*")):
        if f.is_file():
            kb = f.stat().st_size / 1024
            print(f"  {str(f.relative_to(out_dir)):55s}  {kb:6.1f} KB")

    # -- Key findings ---------------------------------------------------------
    _section("Key findings for manuscript revision")
    if not grouped_df.empty:
        s = grouped_df[grouped_df["split_strategy"] == "stratified_cv"]["c_index"]
        g = grouped_df[grouped_df["split_strategy"] == "grouped_cv"]["c_index"]
        if len(s) and len(g):
            dc = g.mean() - s.mean()
            print(f"  1. Grouped CV vs Standard CV (same lean AFT model):")
            print(f"     Standard C = {s.mean():.4f} ± {s.std():.4f}")
            print(f"     Grouped  C = {g.mean():.4f} ± {g.std():.4f}")
            print(f"     Delta_C = {dc:+.4f}")
            if dc < -0.02:
                print("     -> Meaningful C-index drop under grouped CV.")
                print("        Recommend: report both results in supplement,")
                print("        discuss potential optimistic bias in Discussion.")
            elif dc < 0:
                print("     -> Modest C-index drop: some within-cluster "
                      "correlation present.")
                print("        Recommend: report in supplementary as robustness check.")
            else:
                print("     -> C-index stable: within-cluster leakage appears "
                      "limited at this data scale.")

    if not temporal_df.empty:
        c_vals = temporal_df["c_index"].values
        n_vals = temporal_df["n_val"].values
        wt_mean = np.average(c_vals, weights=n_vals)
        print(f"\n  2. Temporal blocked CV (fire-season months only):")
        print(f"     Unweighted mean C = {np.mean(c_vals):.4f}")
        print(f"     N-weighted mean C = {wt_mean:.4f}")
        print(f"     Months evaluated: {len(temporal_df)}")
        print("     -> Use N-weighted mean in supplement for fairer comparison.")

    bdf = sens.get("bounding", pd.DataFrame())
    if not bdf.empty:
        rng_max = bdf["sensitivity_range"].max()
        h_max = bdf.loc[bdf["sensitivity_range"].idxmax(), "horizon"]
        print(f"\n  3. Informative censoring sensitivity:")
        for _, row in bdf.iterrows():
            print(f"     {row['horizon']}: range = {row['sensitivity_range']:.3f} "
                  f"(n_censored_before = {row['n_censored_before_h']})")
        print(f"     Max range = {rng_max:.3f} ({h_max})")
        if rng_max > 0.05:
            print("     -> Non-trivial sensitivity; quantify in Limitations section.")
            print("        Suggested text: 'Informative censoring sensitivity analysis")
            print("        showed event-rate ranges of up to {:.1f} percentage points".format(
                rng_max * 100))
            print("        at the 48 h horizon, suggesting that the non-informative")
            print("        censoring assumption may moderately affect estimates.'")
        else:
            print("     -> Within acceptable range.")
    print()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Supplementary improvement analysis for Fire (MDPI) submission. "
            "Uses real NIFC WFIGS data for incident-level grouped CV. "
            "Requires real WiDS training CSV — no synthetic fallback."
        )
    )
    parser.add_argument(
        "--train",
        default=str(LOCAL_BASE / "train.csv"),
        help="Path to WiDS 2026 training CSV. "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--nifc", default=None,
        help="Path to the NIFC WFIGS CSV file. "
             "If omitted, the hardcoded LOCAL_BASE path is used.",
    )
    parser.add_argument(
        "--output",
        default=str(LOCAL_BASE / "supplement_output"),
        help="Output directory for tables and figures.",
    )
    args = parser.parse_args()
    main(args)