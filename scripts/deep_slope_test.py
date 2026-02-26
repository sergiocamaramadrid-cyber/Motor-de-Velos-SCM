"""
scripts/deep_slope_test.py — Deep-regime slope test for the RAR relation.

Theory
------
In the deep-MOND / deep-velos regime (g_bar ≪ a0):

    g_obs ≈ sqrt(g_bar · a0)

In log-space this is a straight line:

    log10(g_obs) = 0.5 · log10(g_bar) + 0.5 · log10(a0)

so the **expected slope β = 0.5** regardless of interpolation choice, ν-model
or AICc winner.  This is a *structural* test.

What the results mean
---------------------
β ≈ 0.50 ± 0.02  → MOND/deep-velos regime confirmed
β < 0.44          → model deviates structurally in deep regime
Deep points = 0   → dataset does not sample the deep regime at all

Usage
-----
With default paths::

    python scripts/deep_slope_test.py

Explicit options::

    python scripts/deep_slope_test.py \\
        --csv  results/universal_term_comparison_full.csv \\
        --g0   1.2e-10 \\
        --deep-threshold 0.3 \\
        --out  results/diagnostics/deep_slope_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G0_DEFAULT = 1.2e-10        # characteristic acceleration (m/s²)
DEEP_THRESHOLD_DEFAULT = 0.3
EXPECTED_SLOPE = 0.5        # MOND / deep-velos prediction
MIN_DEEP_POINTS = 10        # minimum for a meaningful regression
CSV_DEFAULT = "results/universal_term_comparison_full.csv"
_SEP = "=" * 64

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def deep_slope(log_gbar: np.ndarray, log_gobs: np.ndarray,
               g0: float = G0_DEFAULT,
               deep_threshold: float = DEEP_THRESHOLD_DEFAULT) -> dict:
    """Fit the deep-regime slope β in log10(g_obs) = β · log10(g_bar) + const.

    Parameters
    ----------
    log_gbar : array_like
        log10 of baryonic centripetal acceleration (m/s²) per radial point.
    log_gobs : array_like
        log10 of observed centripetal acceleration (m/s²) per radial point.
    g0 : float
        Characteristic acceleration (m/s²).
    deep_threshold : float
        Fraction of g0 below which a point is "deep": g_bar < threshold × g0.

    Returns
    -------
    dict with keys:
        n_total          — total number of radial points
        n_deep           — number of deep-regime points
        deep_frac        — n_deep / n_total
        slope            — OLS slope β (nan if n_deep < 2, i.e. mathematically
                           impossible to fit a line; a reliability warning is
                           included in 'verdict' if n_deep < MIN_DEEP_POINTS)
        intercept        — OLS intercept
        stderr           — standard error of slope
        r_value          — Pearson r
        p_value          — two-tailed p-value
        delta_from_mond  — slope − 0.5
        verdict          — descriptive string
        log_g0_pred      — implied log10(a0) from intercept (= 2 × intercept
                           under pure MOND: log10(g_obs) = 0.5·log10(g_bar) + 0.5·log10(g0))
    """
    log_gbar = np.asarray(log_gbar, dtype=float)
    log_gobs = np.asarray(log_gobs, dtype=float)

    g_bar = 10.0 ** log_gbar
    deep_mask = g_bar < deep_threshold * g0
    n_total = len(log_gbar)
    n_deep = int(deep_mask.sum())

    if n_deep < 2:  # mathematical minimum for linregress
        return {
            "n_total": n_total,
            "n_deep": n_deep,
            "deep_frac": n_deep / max(n_total, 1),
            "slope": float("nan"),
            "intercept": float("nan"),
            "stderr": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "delta_from_mond": float("nan"),
            "verdict": (
                "⚠️  Insufficient deep-regime points "
                f"(need ≥{MIN_DEEP_POINTS}, got {n_deep}). "
                "Dataset may not sample deep regime."
            ),
            "log_g0_pred": float("nan"),
        }

    slope, intercept, r_value, p_value, stderr = linregress(
        log_gbar[deep_mask], log_gobs[deep_mask]
    )
    delta = slope - EXPECTED_SLOPE

    if n_deep < MIN_DEEP_POINTS:
        verdict = (
            f"⚠️  Only {n_deep} deep points — result may not be reliable "
            f"(need ≥{MIN_DEEP_POINTS})."
        )
    elif abs(delta) <= 2 * stderr:
        verdict = "✅  β consistent with MOND/deep-velos (β ≈ 0.5)"
    elif slope < EXPECTED_SLOPE - 3 * stderr:
        verdict = f"⚠️  β = {slope:.3f} — significant structural deviation below 0.5"
    elif slope > EXPECTED_SLOPE + 3 * stderr:
        verdict = f"⚠️  β = {slope:.3f} — significant structural deviation above 0.5"
    else:
        verdict = f"ℹ️  β = {slope:.3f} — mild deviation from 0.5 (within 3σ)"

    # Under pure MOND: log10(g_obs) = 0.5·log10(g_bar) + 0.5·log10(g0)
    # so  intercept = 0.5·log10(g0)  → log10(g0) = 2·intercept
    log_g0_pred = 2.0 * intercept

    return {
        "n_total": n_total,
        "n_deep": n_deep,
        "deep_frac": n_deep / max(n_total, 1),
        "slope": float(slope),
        "intercept": float(intercept),
        "stderr": float(stderr),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "delta_from_mond": float(delta),
        "verdict": verdict,
        "log_g0_pred": float(log_g0_pred),
    }


def format_report(result: dict, g0: float, threshold: float,
                  csv_path: str) -> list[str]:
    """Format the deep-slope test report as a list of lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — Deep-Regime Slope Test",
        _SEP,
        f"  CSV          : {csv_path}",
        f"  g0           : {g0:.2e} m/s²",
        f"  deep mask    : g_bar < {threshold} × g0",
        f"  Min deep pts : {MIN_DEEP_POINTS}",
        "",
        f"  Total radial points  : {result['n_total']}",
        f"  Deep-regime points   : {result['n_deep']}",
        f"  Deep fraction        : {result['deep_frac']:.3f}",
    ]
    if not np.isnan(result["slope"]):
        lines += [
            "",
            f"  Slope β              : {result['slope']:.4f}",
            f"  Expected (MOND)      : {EXPECTED_SLOPE:.4f}",
            f"  Std err              : {result['stderr']:.4f}",
            f"  Δ from 0.5           : {result['delta_from_mond']:+.4f}",
            f"  Pearson r            : {result['r_value']:.4f}",
            f"  p-value              : {result['p_value']:.2e}",
            f"  Intercept            : {result['intercept']:.4f}",
            f"  Implied log10(g0)    : {result['log_g0_pred']:.4f}",
            f"  Implied g0 (m/s²)  : {10**result['log_g0_pred']:.3e}",
        ]
    lines += [
        "",
        f"  Verdict: {result['verdict']}",
        _SEP,
    ]
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep-regime slope test: β in log(g_obs) = β·log(g_bar) + const."
    )
    parser.add_argument(
        "--csv", default=CSV_DEFAULT,
        help=f"Per-radial-point CSV (default: {CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--g0", type=float, default=G0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {G0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        dest="deep_threshold",
        help=(f"Fraction of g0 defining deep regime (default: "
              f"{DEEP_THRESHOLD_DEFAULT})."),
    )
    parser.add_argument(
        "--out", default=None, metavar="DIR",
        help="Write deep_slope_test.csv and .log to this directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the deep-slope test and print results.

    Returns the result dict so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first."
        )

    with open(csv_path) as fh:
        df = pd.read_csv(fh)
    required = {"log_g_bar", "log_g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}.\n"
            "Regenerate with an updated run_pipeline() that emits per-radial-point rows."
        )

    result = deep_slope(
        df["log_g_bar"].to_numpy(),
        df["log_g_obs"].to_numpy(),
        g0=args.g0,
        deep_threshold=args.deep_threshold,
    )

    report_lines = format_report(result, args.g0, args.deep_threshold, str(csv_path))
    for line in report_lines:
        print(line)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write CSV result
        pd.DataFrame([result]).to_csv(out_dir / "deep_slope_test.csv", index=False)

        # Write log
        (out_dir / "deep_slope_test.log").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )
        print(f"\n  Results written to {out_dir}")

    return result


if __name__ == "__main__":
    main()
