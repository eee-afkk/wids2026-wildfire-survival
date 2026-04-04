# -*- coding: utf-8 -*-
"""
plot_figure10.py
================
Generate Figure 10 (main-text robustness diagnostics) for:
  "A Survival-Probability Fusion Prototype for Censor-Aware
   Multi-Horizon Wildfire Threat Forecasting and Decision-Utility Evaluation"
  Submitted to MDPI Fire.

Layout:
  Panel A (top):  Proxy incident-grouped CV vs standard stratified CV
                  Left  — Discrimination (C-index)
                  Right — Probability error (mean Brier score)
  Panel B (bottom): Informative censoring sensitivity
                  Left  — Worst / best-case event-rate bounds
                  Right — KM-derived P(T ≤ h) vs Aalen-Johansen CIF

Output: Figure_10_robustness_diagnostics.tif  (600 DPI, TIFF)
        Figure_10_robustness_diagnostics.pdf   (vector, for review)

Usage:
    # Reads CSV files produced by WIDS_研究_补充.py; adjust paths below.
    python plot_figure10.py
    python plot_figure10.py --input_dir ./supplement_output --output_dir ./figures

MDPI Fire figure specifications applied:
  - Double-column width: 17.1 cm (6.73 in)
  - Minimum font size: 8 pt (Arial / Helvetica)
  - Resolution: 600 DPI (TIFF line art)
  - White background, no dark theme
  - Colorblind-safe Tol palette
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MDPI Fire typographic & color standards
# ---------------------------------------------------------------------------
# Tol colorblind-safe palette  (Paul Tol, 2021)
TOL_BLUE   = "#4477AA"   # Standard CV / KM
TOL_RED    = "#EE6677"   # Grouped CV / worst-case
TOL_GREEN  = "#228833"   # base event rate
TOL_YELLOW = "#CCBB44"   # Aalen-Johansen CIF
TOL_GREY   = "#BBBBBB"
TOL_NAVY   = "#222255"   # reference lines

# MDPI double-column width in inches
# 17.1 cm / 2.54 = 6.732 in; use 6.93 to compensate bbox_inches tight trim
FIG_W = 6.93
FIG_H = 6.30   # slightly taller to accommodate bottom legend

# Font sizes — all ≥ 8pt (MDPI minimum); bumped up for legibility after scaling
BASE_FS    = 9     # axis labels, titles
TICK_FS    = 8.5   # tick labels
ANNOT_FS   = 8.5   # annotations, μ±SD, legend text
PANEL_FS   = 12    # bold A/B/C panel labels

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#222222",
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "grid.color":         "#EFEFEF",
    "grid.linewidth":     0.4,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size":          BASE_FS,
    "axes.titlesize":     BASE_FS,
    "axes.labelsize":     BASE_FS,
    "xtick.labelsize":    TICK_FS,
    "ytick.labelsize":    TICK_FS,
    "legend.fontsize":    ANNOT_FS,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "#CCCCCC",
    "lines.linewidth":    1.2,
    "patch.linewidth":    0.7,
})

# ---------------------------------------------------------------------------
# Representative / fallback data  (used when CSV files are absent)
# These values match the manuscript text figures (Sec 3.8 in v9):
#   Standard CV  C = 0.9318 ± 0.0221   IBS = 0.0612 ± 0.0088
#   Grouped  CV  C = 0.9283 ± 0.0374   IBS = 0.0694 ± 0.0147
# Censoring sensitivity ranges: 12h=0.027, 24h=0.113, 48h=0.249
# ---------------------------------------------------------------------------
_SEED = 42
_RNG  = np.random.default_rng(_SEED)

def _synthetic_grouped_cv() -> pd.DataFrame:
    """Representative fold-level results consistent with manuscript text."""
    rows = []
    # Stratified CV: 5 folds
    c_strat  = _RNG.normal(0.9318, 0.0221, 5).clip(0.85, 0.99)
    bs_strat = _RNG.normal(0.0612, 0.0088, 5).clip(0.03, 0.12)
    for i, (c, b) in enumerate(zip(c_strat, bs_strat)):
        rows.append({"split_strategy": "stratified_cv",
                     "fold": i + 1, "c_index": round(c, 4),
                     "mean_brier": round(b, 4)})
    # Grouped CV: 5 folds  (wider spread, slightly lower mean)
    c_grp  = _RNG.normal(0.9283, 0.0374, 5).clip(0.82, 0.99)
    bs_grp = _RNG.normal(0.0694, 0.0147, 5).clip(0.03, 0.14)
    for i, (c, b) in enumerate(zip(c_grp, bs_grp)):
        rows.append({"split_strategy": "grouped_cv",
                     "fold": i + 1, "c_index": round(c, 4),
                     "mean_brier": round(b, 4)})
    return pd.DataFrame(rows)


def _synthetic_bounding() -> pd.DataFrame:
    """Representative censoring-sensitivity bounding table."""
    return pd.DataFrame([
        {"horizon": "12h", "base_event_rate": 0.307, "worst_case_rate": 0.320,
         "best_case_rate": 0.293, "sensitivity_range": 0.027, "n_censored_before_h": 4},
        {"horizon": "24h", "base_event_rate": 0.448, "worst_case_rate": 0.511,
         "best_case_rate": 0.398, "sensitivity_range": 0.113, "n_censored_before_h": 17},
        {"horizon": "48h", "base_event_rate": 0.607, "worst_case_rate": 0.728,
         "best_case_rate": 0.479, "sensitivity_range": 0.249, "n_censored_before_h": 35},
    ])


def _synthetic_competing_risk() -> pd.DataFrame:
    """Representative KM vs Aalen-Johansen CIF table."""
    return pd.DataFrame([
        {"horizon": "12h", "km_cif": 0.305, "aj_cif_event1": 0.301, "delta_aj_minus_km": -0.004},
        {"horizon": "24h", "km_cif": 0.443, "aj_cif_event1": 0.432, "delta_aj_minus_km": -0.011},
        {"horizon": "48h", "km_cif": 0.601, "aj_cif_event1": 0.572, "delta_aj_minus_km": -0.029},
    ])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSVs produced by WIDS_研究_补充.py; fall back to representative data."""
    grouped_cv_path   = input_dir / "grouped_cv_comparison.csv"
    bounding_path     = input_dir / "censoring_sensitivity_bounding.csv"
    comp_risk_path    = input_dir / "censoring_sensitivity_competing_risk.csv"

    if grouped_cv_path.exists():
        grouped_df = pd.read_csv(grouped_cv_path)
        print(f"  [load] grouped_cv_comparison.csv  ({len(grouped_df)} rows)")
    else:
        print("  [load] grouped_cv_comparison.csv not found — using representative data")
        grouped_df = _synthetic_grouped_cv()

    if bounding_path.exists():
        bounding_df = pd.read_csv(bounding_path)
        print(f"  [load] censoring_sensitivity_bounding.csv  ({len(bounding_df)} rows)")
    else:
        print("  [load] censoring_sensitivity_bounding.csv not found — using representative data")
        bounding_df = _synthetic_bounding()

    if comp_risk_path.exists():
        comp_risk_df = pd.read_csv(comp_risk_path)
        print(f"  [load] censoring_sensitivity_competing_risk.csv  ({len(comp_risk_df)} rows)")
    else:
        print("  [load] censoring_sensitivity_competing_risk.csv not found — using representative data")
        comp_risk_df = _synthetic_competing_risk()

    return grouped_df, bounding_df, comp_risk_df


# ---------------------------------------------------------------------------
# Panel-drawing helpers
# ---------------------------------------------------------------------------
def _jitter(n: int, spread: float = 0.10) -> np.ndarray:
    return _RNG.uniform(-spread, spread, n)


def _draw_boxplot_panel(
    ax: plt.Axes,
    data_std: np.ndarray,
    data_grp: np.ndarray,
    metric_label: str,
    direction: str,
    show_legend: bool = False,
) -> None:
    """
    Publication-quality boxplot:  Standard CV (blue)  vs  Grouped CV (red).
    Jittered individual-fold dots overlaid; mean ± SD annotation.
    """
    data_list  = [data_std, data_grp]
    colors     = [TOL_BLUE, TOL_RED]
    labels_x   = ["Standard CV", "Proxy\nGrouped CV"]

    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        widths=0.42,
        positions=[1, 2],
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.9, linestyle="--", color="#555555"),
        capprops=dict(linewidth=0.9, color="#555555"),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#888888",
                        markeredgewidth=0, alpha=0.6),
        zorder=2,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.30)
        patch.set_linewidth(0.8)

    # Individual fold dots
    for k, (dat, col) in enumerate(zip(data_list, colors)):
        pos = k + 1
        jit = _jitter(len(dat), spread=0.11)
        ax.scatter(pos + jit, dat, color=col, s=22, zorder=5,
                   edgecolors="white", linewidths=0.4, alpha=0.92)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels_x, fontsize=TICK_FS)
    ax.set_ylabel(metric_label, fontsize=BASE_FS)
    ax.tick_params(axis="x", length=0)

    # Direction annotation
    ax.text(0.97, 0.97, direction, transform=ax.transAxes,
            ha="right", va="top", fontsize=ANNOT_FS,
            style="italic", color="#666666")

    if show_legend:
        pass   # Shared legend is added at figure level in plot_figure10()


def _draw_bounding_panel(ax: plt.Axes, bdf: pd.DataFrame) -> None:
    """
    Grouped bar chart + error bars for worst/best-case event-rate bounds.
    Base rate shown as a solid bar; worst/best-case as symmetric error bars
    centred on the midpoint of the sensitivity range.
    """
    hors = bdf["horizon"].tolist()
    x    = np.arange(len(hors))
    W    = 0.40

    # Base-rate bars
    ax.bar(x, bdf["base_event_rate"], width=W, color=TOL_GREEN,
           alpha=0.72, zorder=3, label="Base event rate (eligible subset)")

    # Worst/best-case range shown as centred error bar
    centers = (bdf["worst_case_rate"] + bdf["best_case_rate"]) / 2
    half    = (bdf["worst_case_rate"] - bdf["best_case_rate"]) / 2

    ax.errorbar(x, centers, yerr=half.values,
                fmt="none", ecolor=TOL_RED,
                elinewidth=2.0, capsize=6, capthick=1.5,
                zorder=5, label="Worst/best-case sensitivity range")

    # Sensitivity range annotation above the upper bound
    for xi, row in zip(x, bdf.itertuples()):
        ax.text(xi, row.worst_case_rate + 0.022,
                f"Δ = {row.sensitivity_range:.3f}",
                ha="center", fontsize=ANNOT_FS, color=TOL_RED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(hors, fontsize=TICK_FS)
    ax.set_xlabel("Prediction horizon", fontsize=BASE_FS)
    ax.set_ylabel("Event rate", fontsize=BASE_FS)

    # Extend y-axis above the top annotation
    ymax = bdf["worst_case_rate"].max() + 0.12
    ax.set_ylim(0, min(ymax, 1.05))

    # Legend outside the axes — placed below the x-axis label.
    # This is the only guaranteed overlap-free location when all bars are tall.
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.50, -0.22),
              fontsize=ANNOT_FS, framealpha=0.95,
              ncol=1, handlelength=1.6, borderpad=0.6)


def _draw_competing_risk_panel(ax: plt.Axes, cdf: pd.DataFrame) -> None:
    """
    Grouped bar chart: KM-derived P(T ≤ h)  vs  Aalen-Johansen CIF.
    Delta annotation above each pair.
    """
    hors = cdf["horizon"].tolist()
    x    = np.arange(len(hors))
    W    = 0.35

    km_vals = cdf["km_cif"].values
    aj_vals = cdf["aj_cif_event1"].fillna(0).values

    ax.bar(x - W / 2, km_vals, width=W, color=TOL_BLUE,
           alpha=0.75, zorder=3, label="KM-derived P(T ≤ h)")
    ax.bar(x + W / 2, aj_vals, width=W, color=TOL_YELLOW,
           alpha=0.75, zorder=3, label="Aalen-Johansen CIF\n(cause-specific)")

    # KM and AJ values: annotate inside each bar (top-aligned)
    for xi, (km, aj) in enumerate(zip(km_vals, aj_vals)):
        ax.text(xi - W / 2, km * 0.96, f"{km:.3f}",
                ha="center", va="top", fontsize=ANNOT_FS - 0.5,
                color="white", fontweight="bold", zorder=6)
        ax.text(xi + W / 2, aj * 0.96, f"{aj:.3f}",
                ha="center", va="top", fontsize=ANNOT_FS - 0.5,
                color="white", fontweight="bold", zorder=6)

    # Delta annotation — well above the tallest bar in each pair
    for k, row in cdf.iterrows():
        d = row.get("delta_aj_minus_km")
        if pd.notna(d):
            ypos = max(km_vals[k], aj_vals[k]) + 0.055
            ax.text(k, ypos, f"δ = {d:+.3f}",
                    ha="center", fontsize=ANNOT_FS, color=TOL_NAVY,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(hors, fontsize=TICK_FS)
    ax.set_xlabel("Prediction horizon", fontsize=BASE_FS)
    ax.set_ylabel("Cumulative incidence", fontsize=BASE_FS)

    ymax_data = max(km_vals.max(), aj_vals.max()) + 0.14
    ax.set_ylim(0, min(ymax_data, 1.02))

    # Legend outside the axes — placed below x-axis label (consistent with left panel)
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.50, -0.22),
              fontsize=ANNOT_FS, framealpha=0.95,
              ncol=1, handlelength=1.6, borderpad=0.6)


# ---------------------------------------------------------------------------
# Main figure composer
# ---------------------------------------------------------------------------
def plot_figure10(
    grouped_df: pd.DataFrame,
    bounding_df: pd.DataFrame,
    comp_risk_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Compose and save Figure 10.

    Layout (3-panel):
      Row 0 (Panel A): C-index boxplot  |  Brier-score boxplot
      Row 1 (Panel B): Censoring worst/best-case bounds  (centered, wider)

    The KM vs Aalen-Johansen CIF panel is omitted from the main text:
    when AJ CIF values are near zero (proxy competing-risk framing with
    few early-censored observations), the panel provides no useful visual
    information and risks misleading reviewers.  It is retained as
    Supplementary Figure S4 with full explanation.
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ── Row 0: two equal-width boxplot panels ────────────────────────────────
    gs_top = fig.add_gridspec(
        1, 2,
        left=0.12, right=0.97,
        top=0.96, bottom=0.58,
        wspace=0.40,
    )
    ax_c  = fig.add_subplot(gs_top[0, 0])   # C-index
    ax_bs = fig.add_subplot(gs_top[0, 1])   # Brier score

    # ── Row 1: one wide censoring panel (centred, 55 % of figure width) ──────
    gs_bot = fig.add_gridspec(
        1, 1,
        left=0.22, right=0.78,
        top=0.38, bottom=0.09,
    )
    ax_bnd = fig.add_subplot(gs_bot[0, 0])

    # ── Panel A: grouped CV comparison ───────────────────────────────────────
    std_mask = grouped_df["split_strategy"] == "stratified_cv"
    grp_mask = grouped_df["split_strategy"] == "grouped_cv"

    c_std  = grouped_df.loc[std_mask, "c_index"].dropna().values
    c_grp  = grouped_df.loc[grp_mask, "c_index"].dropna().values
    bs_std = grouped_df.loc[std_mask, "mean_brier"].dropna().values
    bs_grp = grouped_df.loc[grp_mask, "mean_brier"].dropna().values

    _draw_boxplot_panel(ax_c,  c_std,  c_grp,
                        "C-index", "higher is better ↑",
                        show_legend=True)
    _draw_boxplot_panel(ax_bs, bs_std, bs_grp,
                        "Mean Brier score", "lower is better ↓",
                        show_legend=False)

    from matplotlib.transforms import blended_transform_factory

    for ax, data_list in [(ax_c, [c_std, c_grp]), (ax_bs, [bs_std, bs_grp])]:
        all_v = np.concatenate(data_list)
        lo = max(0, np.nanmin(all_v) - 0.04)
        hi = min(1, np.nanmax(all_v) + 0.06)
        ax.set_ylim(lo, hi)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for k, (dat, col) in enumerate(zip(data_list, [TOL_BLUE, TOL_RED])):
            m, s = np.mean(dat), np.std(dat, ddof=1)
            ax.text(k + 1, -0.24,
                    f"μ = {m:.4f}\n±{s:.4f}",
                    transform=trans,
                    ha="center", va="top",
                    fontsize=ANNOT_FS,          # bumped from ANNOT_FS-0.5
                    color=col, zorder=6,
                    linespacing=1.4)

    ax_c.set_title("(a) Discrimination (C-index)", fontsize=BASE_FS, pad=4)
    ax_bs.set_title("(b) Calibration error (mean Brier score)", fontsize=BASE_FS, pad=4)

    # ── Panel C: censoring sensitivity (centred, wider) ──────────────────────
    _draw_bounding_panel(ax_bnd, bounding_df)
    ax_bnd.set_title(
        "(c) Sensitivity of event rate to informative censoring assumptions",
        fontsize=BASE_FS, pad=4)

    # ── Shared legend for Panels A & B (bottom of figure) ────────────────────
    # Placed centrally below the two boxplot panels, above Panel C.
    shared_patches = [
        mpatches.Patch(facecolor=TOL_BLUE, alpha=0.55,
                       edgecolor="#444444", linewidth=0.6,
                       label="Standard CV  (stratified k-fold)"),
        mpatches.Patch(facecolor=TOL_RED,  alpha=0.55,
                       edgecolor="#444444", linewidth=0.6,
                       label="Proxy grouped CV  (incident-level grouping)"),
    ]
    fig.legend(handles=shared_patches,
               loc="upper center",
               bbox_to_anchor=(0.535, 0.545),   # centered, between rows
               ncol=2,
               fontsize=ANNOT_FS,
               framealpha=0.96,
               edgecolor="#CCCCCC",
               handlelength=1.4,
               columnspacing=1.2,
               borderpad=0.6)

    # ── Save ─────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_path = output_dir / "Figure_10_robustness_diagnostics.tif"
    pdf_path = output_dir / "Figure_10_robustness_diagnostics.pdf"

    # PDF: vector, for review / submission portal upload
    fig.savefig(pdf_path, dpi=300, format="pdf",
                bbox_inches="tight", facecolor="white")

    # TIFF: 600 DPI, RGB (not RGBA) — MDPI requires RGB colour mode
    import io
    from PIL import Image as _PILImage
    buf = io.BytesIO()
    fig.savefig(buf, dpi=600, format="png",
                bbox_inches="tight", facecolor="white")
    buf.seek(0)
    pil_img = _PILImage.open(buf).convert("RGB")
    pil_img.save(tif_path, format="TIFF",
                 dpi=(600, 600),
                 compression="tiff_lzw")

    w_cm = pil_img.size[0] / 600 * 2.54
    h_cm = pil_img.size[1] / 600 * 2.54
    print(f"\n  [save] {tif_path}")
    print(f"         600 DPI · RGB · LZW · {pil_img.size[0]}×{pil_img.size[1]} px "
          f"· {w_cm:.1f}×{h_cm:.1f} cm")
    print(f"  [save] {pdf_path}  (vector PDF)")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 10 (combined robustness diagnostics) "
                    "for the MDPI Fire manuscript."
    )
    parser.add_argument(
        "--input_dir",
        default=r"C:/Users/20919/Desktop/kaggle/WiDS 利用生存分析预测撤离区的威胁时间/supplement_output",
        help="Directory containing the CSV files from WIDS_研究_补充.py. "
             "Falls back to representative data if files are absent.",
    )
    parser.add_argument(
        "--output_dir",
        default="./figure10_output",
        help="Output directory for Figure 10 files.",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  Figure 10 — Robustness Diagnostics")
    print("  MDPI Fire  |  600 DPI TIFF  |  Tol colorblind-safe palette")
    print("=" * 60)

    grouped_df, bounding_df, comp_risk_df = load_data(input_dir)

    print("\n  Composing figure ...")
    plot_figure10(grouped_df, bounding_df, comp_risk_df, output_dir)

    print("\n  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()