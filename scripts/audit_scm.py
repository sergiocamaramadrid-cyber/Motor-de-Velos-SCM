"""
scripts/audit_scm.py — Visual audit for VIF and numerical-stability diagnostics.

Reads the CSV files produced by :func:`src.scm_analysis._write_audit_metrics`
(``<out_dir>/audit/vif_table.csv``, ``<out_dir>/audit/stability_metrics.csv``,
and ``<out_dir>/audit/audit_features.csv``) and writes:

* ``<out_dir>/audit/vif_table.png``         — bar chart of VIF per feature
* ``<out_dir>/audit/stability_metrics.png`` — condition-number summary panel
* ``<out_dir>/audit/residual_vs_hinge.csv`` — per-bin summary of residual vs hinge
* ``<out_dir>/audit/residual_vs_hinge.png`` — scatter + binned-median plot

Unit note
---------
All features in the audit table are dimensionless log10 quantities (dex).
Residuals derived from them are therefore also in **dex**, not in km/s.

Usage
-----
After running the pipeline::

    python scripts/audit_scm.py --out results/

Explicit audit-dir override::

    python scripts/audit_scm.py --audit-dir results/audit/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Thresholds (match _write_audit_metrics)
# ---------------------------------------------------------------------------

_VIF_WARN = 5.0    # moderate collinearity
_VIF_SEVERE = 10.0  # severe collinearity (rule of thumb)
_KAPPA_WARN = 30.0  # moderate — same threshold used in stability_metrics.csv
_KAPPA_SEVERE = 100.0  # severe

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_color(vif: float) -> str:
    """Return a colour that signals the severity of a VIF value."""
    if vif >= _VIF_SEVERE:
        return "#d62728"   # red — severe
    if vif >= _VIF_WARN:
        return "#ff7f0e"   # orange — moderate
    return "#2ca02c"       # green — ok


def plot_vif(vif_df: pd.DataFrame, out_path: Path) -> None:
    """Save a horizontal bar chart of VIF values to *out_path* (PNG).

    Parameters
    ----------
    vif_df : pd.DataFrame
        Must have columns ``feature`` and ``VIF``.
    out_path : Path
        Destination PNG file.
    """
    features = vif_df["feature"].tolist()
    vifs = vif_df["VIF"].tolist()

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.6 * len(features) + 1.0)))

    colours = [_bar_color(v) for v in vifs]
    bars = ax.barh(features, vifs, color=colours, edgecolor="black", linewidth=0.6)

    # Reference lines
    ax.axvline(_VIF_WARN, color="#ff7f0e", linestyle="--", linewidth=0.9,
               label=f"VIF = {_VIF_WARN:.0f} (moderate)")
    ax.axvline(_VIF_SEVERE, color="#d62728", linestyle="--", linewidth=0.9,
               label=f"VIF = {_VIF_SEVERE:.0f} (severe)")

    # Annotate bar values (handle inf)
    for bar, v in zip(bars, vifs):
        label = "∞" if np.isinf(v) else f"{v:.2f}"
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=9)

    ax.set_xlabel("VIF  (dimensionless — features are in dex)")
    ax.set_title("Variance Inflation Factor (VIF) audit\n"
                 "Features: logM, log_gbar, log_j, hinge  [all in dex]")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stability(sm_df: pd.DataFrame, out_path: Path) -> None:
    """Save a condition-number summary panel to *out_path* (PNG).

    Parameters
    ----------
    sm_df : pd.DataFrame
        Must have columns ``metric``, ``value``, ``status``, ``notes``.
    out_path : Path
        Destination PNG file.
    """
    row = sm_df[sm_df["metric"] == "condition_number_kappa"]
    if row.empty:
        return
    kappa = float(row["value"].iloc[0])
    status = str(row["status"].iloc[0])
    notes = str(row["notes"].iloc[0])

    # Colour encodes severity
    if kappa >= _KAPPA_SEVERE:
        bar_col = "#d62728"
        sev_label = "severe (κ ≥ 100)"
    elif kappa >= _KAPPA_WARN:
        bar_col = "#ff7f0e"
        sev_label = "moderate (30 ≤ κ < 100)"
    else:
        bar_col = "#2ca02c"
        sev_label = "stable (κ < 30)"

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(["κ  (condition number)"], [kappa], color=bar_col,
            edgecolor="black", linewidth=0.6)
    ax.axvline(_KAPPA_WARN, color="#ff7f0e", linestyle="--", linewidth=0.9,
               label=f"κ = {_KAPPA_WARN:.0f} (moderate)")
    ax.axvline(_KAPPA_SEVERE, color="#d62728", linestyle="--", linewidth=0.9,
               label=f"κ = {_KAPPA_SEVERE:.0f} (severe)")
    ax.text(kappa + max(kappa * 0.02, 0.5), 0, f"{kappa:.1f}",
            va="center", ha="left", fontsize=10)

    ax.set_xlabel("κ  (dimensionless — computed on z-scored dex features)")
    ax.set_title(
        f"Numerical stability: condition number κ\n"
        f"Status: {status}  ({sev_label})\n"
        f"Note: {notes}",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residual_vs_hinge(
    features_df: pd.DataFrame,
    out_csv: Path,
    out_png: Path,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Save a scatter + binned-median plot of residual_dex vs hinge.

    Also writes a per-bin summary CSV to *out_csv*.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must have columns ``hinge`` and ``residual_dex``.
    out_csv : Path
        Destination CSV (per-bin summary).
    out_png : Path
        Destination PNG file.
    n_bins : int
        Number of equal-width bins along the hinge axis.

    Returns
    -------
    pd.DataFrame
        Per-bin summary with columns
        ``hinge_bin_centre``, ``residual_median``, ``residual_std``, ``n``.
    """
    hinge = features_df["hinge"].values
    residual = features_df["residual_dex"].values

    # Bin along the hinge axis
    h_min, h_max = float(np.nanmin(hinge)), float(np.nanmax(hinge))
    if h_max <= h_min:
        h_max = h_min + 1.0
    edges = np.linspace(h_min, h_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = np.digitize(hinge, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_med, bin_std, bin_n = [], [], []
    for i in range(n_bins):
        mask = bin_idx == i
        vals = residual[mask]
        if len(vals) >= 2:
            bin_med.append(float(np.median(vals)))
            bin_std.append(float(np.std(vals, ddof=1)))
        elif len(vals) == 1:
            bin_med.append(float(vals[0]))
            bin_std.append(float("nan"))
        else:
            bin_med.append(float("nan"))
            bin_std.append(float("nan"))
        bin_n.append(int(mask.sum()))

    summary = pd.DataFrame({
        "hinge_bin_centre": centres,
        "residual_median": bin_med,
        "residual_std": bin_std,
        "n": bin_n,
    })
    summary.to_csv(out_csv, index=False)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))

    # Scatter (subsample for clarity if large)
    _MAX_SCATTER = 5000
    if len(hinge) > _MAX_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(hinge), _MAX_SCATTER, replace=False)
        ax.scatter(hinge[idx], residual[idx], s=4, alpha=0.3,
                   color="#aec7e8", label="radial points (sample)")
    else:
        ax.scatter(hinge, residual, s=4, alpha=0.3,
                   color="#aec7e8", label="radial points")

    # Binned median ± 1σ
    valid = ~np.isnan(np.array(bin_med, dtype=float))
    bm = np.array(bin_med, dtype=float)
    bs = np.array(bin_std, dtype=float)
    bc = centres
    ax.plot(bc[valid], bm[valid], "o-", color="#1f77b4",
            linewidth=1.5, markersize=5, label="binned median")
    eb_mask = valid & ~np.isnan(bs)
    ax.errorbar(bc[eb_mask], bm[eb_mask], yerr=bs[eb_mask],
                fmt="none", ecolor="#1f77b4", alpha=0.5, linewidth=1.0)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("hinge  [dex]  (= max(0, log₁₀(a₀) − log₁₀(ḡ)))")
    ax.set_ylabel("residual_dex  [dex]  (= log₁₀(g_obs) − log₁₀(ḡ))")
    ax.set_title("Residual vs deep-regime hinge\n"
                 "hinge > 0 ↔ deep-MOND regime (ḡ < a₀)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return summary

def _format_report(vif_df: pd.DataFrame, sm_df: pd.DataFrame) -> list[str]:
    """Return list of lines for the text report."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — Internal Audit (VIF + Numerical Stability)",
        "  Features are log10 quantities — residuals are in dex, not km/s",
        _SEP,
        "",
        "  VIF table:",
        f"  {'Feature':<12}  {'VIF':>10}  {'Status':<10}",
        "  " + "-" * 38,
    ]
    for _, row in vif_df.iterrows():
        v = row["VIF"]
        v_str = "∞" if np.isinf(v) else f"{v:.4f}"
        if np.isinf(v) or v >= _VIF_SEVERE:
            st = "⚠  severe"
        elif v >= _VIF_WARN:
            st = "!  moderate"
        else:
            st = "ok"
        lines.append(f"  {row['feature']:<12}  {v_str:>10}  {st}")

    lines += ["", "  Stability metrics:"]
    for _, row in sm_df.iterrows():
        v = row["value"]
        lines.append(f"  {row['metric']}: {v:.4f}  [{row['status']}]")
        lines.append(f"    {row['notes']}")

    lines += ["", _SEP]
    return lines


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_audit(audit_dir: Path) -> None:
    """Read audit CSVs, produce PNG charts and print a text report.

    Parameters
    ----------
    audit_dir : Path
        Directory containing ``vif_table.csv``, ``stability_metrics.csv``,
        and optionally ``audit_features.csv``.
    """
    vif_path = audit_dir / "vif_table.csv"
    sm_path = audit_dir / "stability_metrics.csv"

    missing = [p for p in (vif_path, sm_path) if not p.exists()]
    if missing:
        paths = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Audit CSV(s) not found: {paths}\n"
            "Run the pipeline first:  python -m src.scm_analysis "
            "--data-dir data/SPARC --out results/"
        )

    vif_df = pd.read_csv(vif_path)
    sm_df = pd.read_csv(sm_path)

    # --- Text report ---
    for line in _format_report(vif_df, sm_df):
        print(line)

    # --- VIF chart ---
    vif_png = audit_dir / "vif_table.png"
    plot_vif(vif_df, vif_png)
    print(f"  VIF chart saved → {vif_png}")

    # --- Stability chart ---
    sm_png = audit_dir / "stability_metrics.png"
    plot_stability(sm_df, sm_png)
    print(f"  Stability chart saved → {sm_png}")

    # --- Residual vs hinge (requires audit_features.csv from the pipeline) ---
    features_path = audit_dir / "audit_features.csv"
    if features_path.exists():
        feat_df = pd.read_csv(features_path)
        if {"hinge", "residual_dex"}.issubset(feat_df.columns):
            rvh_csv = audit_dir / "residual_vs_hinge.csv"
            rvh_png = audit_dir / "residual_vs_hinge.png"
            plot_residual_vs_hinge(feat_df, rvh_csv, rvh_png)
            print(f"  Residual-vs-hinge CSV saved → {rvh_csv}")
            print(f"  Residual-vs-hinge plot saved → {rvh_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visual audit: VIF collinearity + condition-number diagnostics.\n"
            "Reads audit CSVs written by run_pipeline() and saves PNG charts."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--out", metavar="DIR",
        help="Pipeline output directory (audit/ subdirectory is read from here).",
    )
    src.add_argument(
        "--audit-dir", metavar="DIR", dest="audit_dir",
        help="Audit directory directly (overrides --out).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.audit_dir:
        audit_dir = Path(args.audit_dir)
    else:
        audit_dir = Path(args.out) / "audit"

    run_audit(audit_dir)


if __name__ == "__main__":
    main()
