"""
scripts/vif_report.py — Formatted VIF diagnostic report for the SCM audit.

Reads the per-galaxy VIF table produced by ``run_pipeline()`` and prints a
human-readable multicollinearity report with interpretation labels and verdicts
for each model feature.

Theory
------
The VIF for feature *i* measures how much of its variance is explained by the
other predictors in the regression matrix:

    VIF_i = 1 / (1 − R_i²)

The design matrix audited here is:

    [const,  logM,  log_gbar,  log_j,  hinge]

where  hinge = max(0, log g_0 - log g_bar)  and log g_0 = -10.45 (frozen).

The ``hinge`` column is the key audited term: it captures whether the
near-flat-regime correction carries independent information or is merely a
disguised linear combination of logM and log_gbar.

VIF interpretation table
------------------------
VIF       Label                   Symbol
1–2       independent             ✔
2–5       moderate (acceptable)   ✔
5–10      strong (watch)          ⚠
>10       structural (serious)    ✖

Usage
-----
Default paths::

    python scripts/vif_report.py

Explicit CSV::

    python scripts/vif_report.py --csv results/audit/vif_table.csv

Write report to file as well::

    python scripts/vif_report.py --out results/audit/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIF_CSV_DEFAULT = "results/audit/vif_table.csv"
_SEP = "=" * 64


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def vif_verdict(vif: float) -> tuple[str, str]:
    """Return (label, symbol) for a VIF value.

    Parameters
    ----------
    vif : float
        Variance Inflation Factor (must be ≥ 1 for well-conditioned data).

    Returns
    -------
    tuple[str, str]
        ``(label, symbol)`` where symbol is one of ``"✔"``, ``"⚠"``, ``"✖"``.
    """
    if vif > 10.0:
        return "structural multicollinearity (serious)", "✖"
    if vif > 5.0:
        return "strong correlation (watch)", "⚠"
    if vif > 2.0:
        return "moderate correlation (acceptable)", "✔"
    return "independent", "✔"


def format_vif_report(vif_df: pd.DataFrame, csv_path: str) -> list[str]:
    """Format the VIF table as a human-readable report.

    Parameters
    ----------
    vif_df : pd.DataFrame
        Must contain columns ``variable`` and ``VIF``.
    csv_path : str
        Source CSV path (for the header).

    Returns
    -------
    list[str]
        Lines of the report (no trailing newlines).
    """
    lines = [
        _SEP,
        "  Motor de Velos SCM — VIF Multicollinearity Report",
        _SEP,
        f"  CSV          : {csv_path}",
        "",
        "  VIF interpretation:",
        "    1–2   independent                      ✔",
        "    2–5   moderate correlation (acceptable) ✔",
        "    5–10  strong correlation (watch)        ⚠",
        "    >10   structural multicollinearity      ✖",
        "",
        f"  {'Variable':<12}  {'VIF':>8}   {'Verdict':<44}  Symbol",
        "  " + "-" * 60,
    ]

    hinge_row = None
    for _, row in vif_df.iterrows():
        var = str(row["variable"])
        v = float(row["VIF"])
        label, symbol = vif_verdict(v)
        lines.append(f"  {var:<12}  {v:>8.3f}   {label:<44}  {symbol}")
        if var == "hinge":
            hinge_row = (var, v, label, symbol)

    lines.append("")
    lines.append("  --- Audit verdict ---")
    if hinge_row is not None:
        _, v, label, symbol = hinge_row
        if symbol == "✔":
            verdict = (
                f"  hinge VIF = {v:.3f}  →  {symbol}  independent — "
                "the near-flat correction carries independent information."
            )
        elif symbol == "⚠":
            verdict = (
                f"  hinge VIF = {v:.3f}  →  {symbol}  partial collinearity — "
                "monitor but not structurally redundant."
            )
        else:
            verdict = (
                f"  hinge VIF = {v:.3f}  →  {symbol}  redundant — "
                "hinge is a near-linear combination of logM / log_gbar."
            )
        lines.append(verdict)
    else:
        lines.append("  hinge column not found in VIF table.")

    lines.append(_SEP)
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a formatted VIF multicollinearity report.",
    )
    parser.add_argument(
        "--csv", default=VIF_CSV_DEFAULT, metavar="PATH",
        help=f"VIF table CSV (default: {VIF_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Write vif_report.log to this directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Run the VIF report and print results.

    Returns the VIF DataFrame so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"VIF table not found: {csv_path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first."
        )

    vif_df = pd.read_csv(csv_path)
    required = {"variable", "VIF"}
    missing = required - set(vif_df.columns)
    if missing:
        raise ValueError(
            f"VIF table missing required columns: {missing}.\n"
            "Regenerate with an updated run_pipeline() that emits audit/vif_table.csv."
        )

    report_lines = format_vif_report(vif_df, str(csv_path))
    for line in report_lines:
        print(line)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vif_report.log").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        print(f"\n  Report written to {out_dir / 'vif_report.log'}")

    return vif_df


if __name__ == "__main__":
    main()
