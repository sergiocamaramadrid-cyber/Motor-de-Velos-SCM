"""
scripts/audit_scm.py — Tamper-proof audit entry-point for Motor de Velos SCM.

The script forces the analyst to declare **upfront** which dataset they are
using, then applies pre-specified (hard-coded) pass/fail criteria so that the
verdict cannot be adjusted after inspecting the data.

Two input modes (mutually exclusive — you must pick exactly one):

  --global-csv FILE   Per-galaxy summary table (per_galaxy_summary.csv).
                      Columns required: galaxy, chi2_reduced, upsilon_disk,
                      n_points, Vflat_kms, M_bar_BTFR_Msun.

  --radial-csv FILE   Per-radial-point table (universal_term_comparison_full.csv).
                      Columns required: log_g_bar, log_g_obs.

Pre-specified pass/fail thresholds (cannot be changed at run-time):

  Global mode
  -----------
  CHI2_MEDIAN_PASS   : median(chi2_reduced) < 3.0
  CHI2_FRAC_PASS     : fraction of galaxies with chi2_reduced < 2.0 ≥ 0.30

  Radial mode
  -----------
  SLOPE_EXPECTED     : 0.5  (deep-MOND / deep-velos prediction)
  MIN_DEEP_POINTS    : 10   (minimum radial points in the deep regime to
                             produce a meaningful regression)
  SLOPE_SIGMA_TOL    : 3    (pass if |β − 0.5| ≤ SLOPE_SIGMA_TOL × stderr)

Outputs written to --out:
  audit_metrics.csv   — all computed statistics
  audit_verdict.txt   — human-readable PASS/FAIL verdict with provenance

Exit codes:
  0  PASS
  1  FAIL
  2  Configuration / file error

Usage
-----
With the global aggregate table::

    python scripts/audit_scm.py \\
        --global-csv results/per_galaxy_summary.csv \\
        --out results/audit/global

With the per-radial-point table::

    python scripts/audit_scm.py \\
        --radial-csv results/universal_term_comparison_full.csv \\
        --out results/audit/radial
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Pre-specified audit thresholds — DO NOT CHANGE AFTER DATA INSPECTION
# ---------------------------------------------------------------------------

# Global-mode thresholds
CHI2_MEDIAN_PASS: float = 3.0     # median chi2_reduced must be below this
CHI2_FRAC_GOOD_THRESHOLD: float = 2.0   # per-galaxy chi2 threshold for "good fit"
CHI2_FRAC_PASS: float = 0.30     # fraction of galaxies with "good fit" required

# Radial-mode thresholds
SLOPE_EXPECTED: float = 0.5      # deep-MOND / deep-velos slope prediction
SLOPE_SIGMA_TOL: int = 3         # |β − 0.5| ≤ SLOPE_SIGMA_TOL × stderr → PASS
MIN_DEEP_POINTS: int = 10        # minimum deep-regime points for a valid regression
DEEP_THRESHOLD_DEFAULT: float = 0.3   # g_bar / a0 below which a point is "deep"
A0_DEFAULT: float = 1.2e-10      # characteristic acceleration (m/s²)

_SEP = "=" * 70


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file (tamper-evident checksum)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Global-mode audit
# ---------------------------------------------------------------------------

_GLOBAL_REQUIRED_COLS = {"galaxy", "chi2_reduced", "upsilon_disk", "n_points"}


def audit_global(csv_path: Path, a0: float = A0_DEFAULT) -> dict:
    """Audit using the per-galaxy summary table.

    Parameters
    ----------
    csv_path : Path
        Path to per_galaxy_summary.csv.
    a0 : float
        Characteristic acceleration (for reference only; not used in verdict).

    Returns
    -------
    dict
        All computed metrics plus ``verdict`` ("PASS" or "FAIL") and
        ``verdict_reason``.
    """
    df = pd.read_csv(csv_path)
    missing = _GLOBAL_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Global CSV missing required columns: {sorted(missing)}\n"
            f"Found: {sorted(df.columns.tolist())}"
        )

    n_galaxies = len(df)
    chi2 = df["chi2_reduced"].dropna().values
    ud = df["upsilon_disk"].dropna().values

    n_finite_chi2 = int(np.isfinite(chi2).sum())
    chi2_finite = chi2[np.isfinite(chi2)]

    median_chi2 = float(np.median(chi2_finite)) if len(chi2_finite) else float("nan")
    mean_chi2 = float(np.mean(chi2_finite)) if len(chi2_finite) else float("nan")
    p25_chi2 = float(np.percentile(chi2_finite, 25)) if len(chi2_finite) else float("nan")
    p75_chi2 = float(np.percentile(chi2_finite, 75)) if len(chi2_finite) else float("nan")

    n_good = int((chi2_finite < CHI2_FRAC_GOOD_THRESHOLD).sum())
    frac_good = n_good / n_finite_chi2 if n_finite_chi2 > 0 else float("nan")

    median_ud = float(np.median(ud[np.isfinite(ud)])) if len(ud) else float("nan")
    p5_ud = float(np.percentile(ud[np.isfinite(ud)], 5)) if len(ud) else float("nan")
    p95_ud = float(np.percentile(ud[np.isfinite(ud)], 95)) if len(ud) else float("nan")

    # --- Apply pre-specified criteria ---
    crit_chi2_median = (np.isfinite(median_chi2) and median_chi2 < CHI2_MEDIAN_PASS)
    crit_frac_good = (np.isfinite(frac_good) and frac_good >= CHI2_FRAC_PASS)
    passed = crit_chi2_median and crit_frac_good

    reasons = []
    if not np.isfinite(median_chi2):
        reasons.append("median chi2_reduced is not finite (no valid data?)")
    elif not crit_chi2_median:
        reasons.append(
            f"median chi2_reduced = {median_chi2:.4f} ≥ {CHI2_MEDIAN_PASS} (threshold)"
        )
    if not np.isfinite(frac_good):
        reasons.append("fraction of good-fit galaxies is not finite")
    elif not crit_frac_good:
        reasons.append(
            f"fraction(chi2 < {CHI2_FRAC_GOOD_THRESHOLD}) = {frac_good:.3f} "
            f"< {CHI2_FRAC_PASS} (threshold)"
        )
    if passed:
        verdict_reason = (
            f"median chi2_reduced = {median_chi2:.4f} < {CHI2_MEDIAN_PASS}  AND  "
            f"fraction(chi2 < {CHI2_FRAC_GOOD_THRESHOLD}) = {frac_good:.3f} "
            f"≥ {CHI2_FRAC_PASS}"
        )
    else:
        verdict_reason = "; ".join(reasons)

    return {
        "mode": "global",
        "n_galaxies": n_galaxies,
        "n_finite_chi2": n_finite_chi2,
        "chi2_median": median_chi2,
        "chi2_mean": mean_chi2,
        "chi2_p25": p25_chi2,
        "chi2_p75": p75_chi2,
        "chi2_threshold_good": CHI2_FRAC_GOOD_THRESHOLD,
        "n_good_fit": n_good,
        "frac_good_fit": frac_good,
        "upsilon_disk_median": median_ud,
        "upsilon_disk_p5": p5_ud,
        "upsilon_disk_p95": p95_ud,
        "criterion_chi2_median": f"median chi2 < {CHI2_MEDIAN_PASS}",
        "criterion_frac_good": f"frac(chi2<{CHI2_FRAC_GOOD_THRESHOLD}) >= {CHI2_FRAC_PASS}",
        "crit_chi2_median_pass": bool(crit_chi2_median),
        "crit_frac_good_pass": bool(crit_frac_good),
        "verdict": "PASS" if passed else "FAIL",
        "verdict_reason": verdict_reason,
    }


# ---------------------------------------------------------------------------
# Radial-mode audit
# ---------------------------------------------------------------------------

_RADIAL_REQUIRED_COLS = {"log_g_bar", "log_g_obs"}


def audit_radial(csv_path: Path, a0: float = A0_DEFAULT,
                 deep_threshold: float = DEEP_THRESHOLD_DEFAULT) -> dict:
    """Audit using the per-radial-point table.

    Parameters
    ----------
    csv_path : Path
        Path to universal_term_comparison_full.csv.
    a0 : float
        Characteristic acceleration (m/s²) used for the deep-regime mask.
    deep_threshold : float
        A point is "deep" if g_bar < deep_threshold × a0.

    Returns
    -------
    dict
        All computed metrics plus ``verdict`` and ``verdict_reason``.
    """
    df = pd.read_csv(csv_path)
    missing = _RADIAL_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Radial CSV missing required columns: {sorted(missing)}\n"
            f"Found: {sorted(df.columns.tolist())}\n"
            "Regenerate with run_pipeline() to include log_g_bar / log_g_obs columns."
        )

    log_gbar = df["log_g_bar"].to_numpy(dtype=float)
    log_gobs = df["log_g_obs"].to_numpy(dtype=float)

    # Only use finite pairs
    finite_mask = np.isfinite(log_gbar) & np.isfinite(log_gobs)
    log_gbar = log_gbar[finite_mask]
    log_gobs = log_gobs[finite_mask]

    n_total = len(log_gbar)
    g_bar = 10.0 ** log_gbar
    deep_mask = g_bar < deep_threshold * a0
    n_deep = int(deep_mask.sum())
    deep_frac = n_deep / max(n_total, 1)

    if n_deep < 2:
        return {
            "mode": "radial",
            "a0": a0,
            "deep_threshold": deep_threshold,
            "n_total": n_total,
            "n_deep": n_deep,
            "deep_frac": deep_frac,
            "slope": float("nan"),
            "intercept": float("nan"),
            "stderr": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "delta_from_mond": float("nan"),
            "log_g0_pred": float("nan"),
            "rms_dex": float("nan"),
            "criterion_deep_points": f"n_deep >= {MIN_DEEP_POINTS}",
            "criterion_slope": f"|β − {SLOPE_EXPECTED}| ≤ {SLOPE_SIGMA_TOL}×stderr",
            "crit_deep_points_pass": False,
            "crit_slope_pass": False,
            "verdict": "FAIL",
            "verdict_reason": (
                f"Insufficient deep-regime points: {n_deep} "
                f"(need ≥{MIN_DEEP_POINTS}).  "
                "Dataset may not sample the deep regime."
            ),
        }

    slope, intercept, r_value, p_value, stderr = linregress(
        log_gbar[deep_mask], log_gobs[deep_mask]
    )
    slope = float(slope)
    intercept = float(intercept)
    r_value = float(r_value)
    p_value = float(p_value)
    stderr = float(stderr)
    delta = slope - SLOPE_EXPECTED
    log_g0_pred = 2.0 * intercept  # under pure MOND: intercept = 0.5·log10(g0)

    # rms_dex over all points (log_g_obs − log_g_bar as a simple spread metric)
    rms_dex = float(np.sqrt(np.mean((log_gobs - log_gbar) ** 2)))

    # --- Apply pre-specified criteria ---
    crit_deep_points = n_deep >= MIN_DEEP_POINTS
    crit_slope = stderr > 0 and abs(delta) <= SLOPE_SIGMA_TOL * stderr

    passed = crit_deep_points and crit_slope

    reasons = []
    if not crit_deep_points:
        reasons.append(
            f"n_deep = {n_deep} < {MIN_DEEP_POINTS} (unreliable regression)"
        )
    if not crit_slope:
        if stderr <= 0:
            reasons.append("slope stderr = 0 (degenerate fit)")
        else:
            nsig = abs(delta) / stderr
            reasons.append(
                f"β = {slope:.4f}, |β − {SLOPE_EXPECTED}| = {abs(delta):.4f} "
                f"= {nsig:.1f}σ > {SLOPE_SIGMA_TOL}σ"
            )
    if passed:
        nsig = abs(delta) / stderr if stderr > 0 else float("nan")
        verdict_reason = (
            f"n_deep = {n_deep} ≥ {MIN_DEEP_POINTS}  AND  "
            f"β = {slope:.4f}, |β − {SLOPE_EXPECTED}| = {abs(delta):.4f} "
            f"= {nsig:.1f}σ ≤ {SLOPE_SIGMA_TOL}σ"
        )
    else:
        verdict_reason = "; ".join(reasons)

    return {
        "mode": "radial",
        "a0": a0,
        "deep_threshold": deep_threshold,
        "n_total": n_total,
        "n_deep": n_deep,
        "deep_frac": deep_frac,
        "slope": slope,
        "intercept": intercept,
        "stderr": stderr,
        "r_value": r_value,
        "p_value": p_value,
        "delta_from_mond": delta,
        "log_g0_pred": log_g0_pred,
        "rms_dex": rms_dex,
        "criterion_deep_points": f"n_deep >= {MIN_DEEP_POINTS}",
        "criterion_slope": f"|β − {SLOPE_EXPECTED}| ≤ {SLOPE_SIGMA_TOL}×stderr",
        "crit_deep_points_pass": bool(crit_deep_points),
        "crit_slope_pass": bool(crit_slope),
        "verdict": "PASS" if passed else "FAIL",
        "verdict_reason": verdict_reason,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_global_report(metrics: dict, csv_path: Path,
                           sha: str, commit: str) -> list[str]:
    m = metrics
    lines = [
        _SEP,
        "  Motor de Velos SCM — Audit Report (GLOBAL / per-galaxy mode)",
        _SEP,
        f"  Input file   : {csv_path}",
        f"  SHA-256      : {sha}",
        f"  Git commit   : {commit}",
        "",
        "  Pre-specified thresholds (fixed before data inspection)",
        f"    Criterion 1: median(chi2_reduced) < {CHI2_MEDIAN_PASS}",
        f"    Criterion 2: fraction(chi2 < {CHI2_FRAC_GOOD_THRESHOLD}) "
        f">= {CHI2_FRAC_PASS}",
        "",
        f"  N galaxies            : {m['n_galaxies']}",
        f"  N with finite chi2    : {m['n_finite_chi2']}",
        f"  chi2_reduced  median  : {m['chi2_median']:.4f}",
        f"  chi2_reduced  mean    : {m['chi2_mean']:.4f}",
        f"  chi2_reduced  Q25/Q75 : {m['chi2_p25']:.4f} / {m['chi2_p75']:.4f}",
        f"  Galaxies with chi2 < {m['chi2_threshold_good']:.1f}"
        f"  : {m['n_good_fit']} / {m['n_finite_chi2']}"
        f"  (frac = {m['frac_good_fit']:.3f})",
        f"  upsilon_disk  median  : {m['upsilon_disk_median']:.3f}",
        f"  upsilon_disk  p5/p95  : {m['upsilon_disk_p5']:.3f} / "
        f"{m['upsilon_disk_p95']:.3f}",
        "",
        f"  Criterion 1 (chi2 median)    : {'PASS' if m['crit_chi2_median_pass'] else 'FAIL'}",
        f"  Criterion 2 (frac good fit)  : {'PASS' if m['crit_frac_good_pass'] else 'FAIL'}",
        "",
        f"  *** VERDICT: {m['verdict']} ***",
        f"  Reason: {m['verdict_reason']}",
        _SEP,
    ]
    return lines


def _format_radial_report(metrics: dict, csv_path: Path,
                           sha: str, commit: str) -> list[str]:
    m = metrics
    slope_s = f"{m['slope']:.4f}" if np.isfinite(m.get("slope", float("nan"))) else "N/A"
    stderr_s = f"{m['stderr']:.4f}" if np.isfinite(m.get("stderr", float("nan"))) else "N/A"
    delta_s = f"{m['delta_from_mond']:+.4f}" if np.isfinite(m.get("delta_from_mond", float("nan"))) else "N/A"
    lines = [
        _SEP,
        "  Motor de Velos SCM — Audit Report (RADIAL / per-point mode)",
        _SEP,
        f"  Input file   : {csv_path}",
        f"  SHA-256      : {sha}",
        f"  Git commit   : {commit}",
        "",
        "  Pre-specified thresholds (fixed before data inspection)",
        f"    Criterion 1: n_deep >= {MIN_DEEP_POINTS}",
        f"    Criterion 2: |β − {SLOPE_EXPECTED}| ≤ {SLOPE_SIGMA_TOL}×stderr",
        f"    deep mask   : g_bar < {m['deep_threshold']} × a0  "
        f"(a0 = {m['a0']:.2e} m/s²)",
        "",
        f"  Total radial points   : {m['n_total']}",
        f"  Deep-regime points    : {m['n_deep']}",
        f"  Deep fraction         : {m['deep_frac']:.3f}",
        "",
        f"  Slope β               : {slope_s}",
        f"  Expected (MOND)       : {SLOPE_EXPECTED}",
        f"  Stderr                : {stderr_s}",
        f"  Δ from 0.5            : {delta_s}",
        "",
        f"  Criterion 1 (n_deep)  : {'PASS' if m['crit_deep_points_pass'] else 'FAIL'}",
        f"  Criterion 2 (slope)   : {'PASS' if m['crit_slope_pass'] else 'FAIL'}",
        "",
        f"  *** VERDICT: {m['verdict']} ***",
        f"  Reason: {m['verdict_reason']}",
        _SEP,
    ]
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tamper-proof audit for the Motor de Velos SCM model.\n"
            "You must declare upfront which dataset you are auditing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--global-csv", metavar="FILE", dest="global_csv",
        help=(
            "Per-galaxy summary CSV (per_galaxy_summary.csv).  "
            "Audit mode: global aggregate."
        ),
    )
    src.add_argument(
        "--radial-csv", metavar="FILE", dest="radial_csv",
        help=(
            "Per-radial-point CSV (universal_term_comparison_full.csv).  "
            "Audit mode: radial deep-slope."
        ),
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="Output directory for audit_metrics.csv and audit_verdict.txt.",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=(
            f"Deep-regime threshold x = g_bar/a0 (radial mode only; "
            f"default: {DEEP_THRESHOLD_DEFAULT})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the audit and return an exit code (0=PASS, 1=FAIL, 2=error)."""
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = _git_commit()

    try:
        if args.global_csv:
            csv_path = Path(args.global_csv)
            if not csv_path.exists():
                print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
                return 2
            sha = _sha256(csv_path)
            metrics = audit_global(csv_path, a0=args.a0)
            report_lines = _format_global_report(metrics, csv_path, sha, commit)
        else:
            csv_path = Path(args.radial_csv)
            if not csv_path.exists():
                print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
                return 2
            sha = _sha256(csv_path)
            metrics = audit_radial(
                csv_path, a0=args.a0, deep_threshold=args.deep_threshold
            )
            report_lines = _format_radial_report(metrics, csv_path, sha, commit)
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Print to stdout
    for line in report_lines:
        print(line)

    # Write verdict text
    verdict_path = out_dir / "audit_verdict.txt"
    verdict_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Write metrics CSV
    metrics_path = out_dir / "audit_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"\n  Results written to {out_dir}")

    return 0 if metrics["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
