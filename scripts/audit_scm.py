"""
scripts/audit_scm.py — Visual audit for VIF and numerical-stability diagnostics.

Reads the CSV files produced by :func:`src.scm_analysis._write_audit_metrics`
(``<out_dir>/audit/vif_table.csv`` and ``<out_dir>/audit/stability_metrics.csv``)
and writes:

* ``<out_dir>/audit/vif_table.png``      — bar chart of VIF per feature
* ``<out_dir>/audit/stability_metrics.png`` — condition-number summary panel

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


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

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
        Directory containing ``vif_table.csv`` and ``stability_metrics.csv``.
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
