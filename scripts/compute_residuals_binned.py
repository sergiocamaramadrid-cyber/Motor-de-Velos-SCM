"""compute_residuals_binned.py — Bin SCM/RAR model residuals by g_bar.

Reads an acceleration-pair dataset (g_bar, g_obs columns in m/s^2),
fits the Motor-de-Velos acceleration scale g0 (a0) from the data,
emits fitting telemetry, computes log-residuals
  Δ = log10(g_obs) − log10(g_pred)
where g_pred is the RAR interpolation formula, then bins those residuals
in equal-width log10(g_bar) bins and writes the summary statistics to
``results/diagnostics/residuals_binned_v02.csv``.

Telemetría del Ajuste
---------------------
After fitting the script prints:

    [SCM v0.2] g0_hat=<value>  g0_lo=<lower_CI>  g0_hi=<upper_CI>
    [INFO] g_bar quantiles: q10=<p10>  q50=<p50>  q90=<p90>  [m/s^2]
    [WARN] g0_hat hit LOWER bound again
    [WARN] g0_hat hit UPPER bound

The WARN lines are only printed when the fitted log10(g0) is within
G0_BORDER_TOL decades of either bound of the search interval.

Reprint Notarial de Residuos
-----------------------------
After writing the CSV the script prints the full binned-residuals table
to stdout (Reprint Notarial).

Usage
-----
    python scripts/compute_residuals_binned.py [--data PATH] [--out PATH]
         [--bins N] [--min-count N]

Defaults
--------
    --data      data/sparc_rar_sample.csv
    --out       results/diagnostics/residuals_binned_v02.csv
    --bins      15
    --min-count 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Motor-de-Velos / MOND acceleration scale  [m s⁻²]
A0_SI: float = 1.2e-10

#: Bounds for g0 fitting search interval [m s⁻²]
#: Lower bound expanded to 1e-16 (log10 = -16.0) — v0.2b fix.
#: The previous bound of 1e-12 caused g0_hat to be clamped at the lower
#: boundary; physical analysis shows the optimal g0 can be ~1e-13 or below.
G0_BOUNDS: tuple[float, float] = (1e-16, 1e-8)

#: Log10-distance from a bound that triggers a WARN (v0.2b: log10 tolerance)
G0_BORDER_TOL: float = 0.05


# ---------------------------------------------------------------------------
# RAR model (SI units — accelerations in m s⁻²)
# ---------------------------------------------------------------------------

def nu_rar(x: np.ndarray) -> np.ndarray:
    """RAR interpolation function  ν(x) = 1 / (1 − exp(−√x)).

    Parameters
    ----------
    x : array_like
        Dimensionless ratio g_bar / a0.

    Returns
    -------
    nu : ndarray
        Interpolation factor  ≥ 1.
    """
    x = np.asarray(x, dtype=float)
    sqrtx = np.sqrt(np.clip(x, 1e-30, None))
    return 1.0 / (1.0 - np.exp(-sqrtx))


def g_pred_rar(g_bar: np.ndarray, a0: float = A0_SI) -> np.ndarray:
    """Predicted observed acceleration from the RAR formula.

    g_pred = g_bar · ν(g_bar / a0)

    Parameters
    ----------
    g_bar : array_like
        Baryonic centripetal acceleration [m s⁻²].
    a0 : float, optional
        Motor-de-Velos acceleration scale [m s⁻²].

    Returns
    -------
    g_pred : ndarray
        Predicted g_obs values [m s⁻²].
    """
    g_bar = np.asarray(g_bar, dtype=float)
    return g_bar * nu_rar(g_bar / a0)


# ---------------------------------------------------------------------------
# g0 fitting
# ---------------------------------------------------------------------------

def _neg_log_likelihood(log10_a0: float, g_bar: np.ndarray, g_obs: np.ndarray) -> float:
    """Negative log-likelihood for Gaussian errors on log-residuals.

    Minimising this w.r.t. log10(a0) gives the MLE of the Motor-de-Velos
    acceleration scale.

    Parameters
    ----------
    log10_a0 : float
        log10 of the trial acceleration scale.
    g_bar  : ndarray
        Baryonic acceleration data [m s⁻²].
    g_obs  : ndarray
        Observed acceleration data [m s⁻²].

    Returns
    -------
    float  Negative log-likelihood (to be minimised).
    """
    a0 = 10.0 ** log10_a0
    g_pred = g_pred_rar(g_bar, a0=a0)
    # log-residuals
    delta = np.log10(g_obs) - np.log10(g_pred)
    return float(np.sum(delta ** 2))


def g0_touches_bounds(
    g0_hat: float,
    bounds: tuple[float, float],
    tol_log10: float = G0_BORDER_TOL,
) -> tuple[bool, bool]:
    """Return (touches_lower, touches_upper) using log10-space proximity.

    Guards against non-finite or non-positive g0_hat and validates bounds
    before performing the log10 comparison.

    Parameters
    ----------
    g0_hat    : float  Fitted acceleration scale [m s⁻²].
    bounds    : (lo, hi)  Search interval [m s⁻²]; must satisfy 0 < lo < hi.
    tol_log10 : float  Proximity threshold in log10 decades (default G0_BORDER_TOL).

    Returns
    -------
    (touches_lower, touches_upper) : (bool, bool)
        Both False when g0_hat is non-finite or <= 0.

    Raises
    ------
    ValueError  If bounds are invalid (non-finite, <= 0, or lo >= hi).
    """
    lo, hi = float(bounds[0]), float(bounds[1])

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError(f"Invalid bounds: {bounds}")

    if (not np.isfinite(g0_hat)) or g0_hat <= 0:
        return False, False

    log10_g0 = float(np.log10(g0_hat))
    log10_lo = float(np.log10(lo))
    log10_hi = float(np.log10(hi))

    touches_lower = abs(log10_g0 - log10_lo) < float(tol_log10)
    touches_upper = abs(log10_g0 - log10_hi) < float(tol_log10)
    return bool(touches_lower), bool(touches_upper)


def fit_g0(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    bounds: tuple[float, float] = G0_BOUNDS,
) -> dict:
    """Fit the Motor-de-Velos acceleration scale g0 from data.

    Uses scalar minimisation of the sum of squared log-residuals over
    log10(g0) in the interval ``bounds``.  The optimizer operates
    entirely in log10 space: bounds passed as (-16.0, -8.0) for the
    default G0_BOUNDS of (1e-16, 1e-8).

    Parameters
    ----------
    g_bar  : array_like   Baryonic acceleration [m s⁻²].
    g_obs  : array_like   Observed acceleration [m s⁻²].
    bounds : (lo, hi)     Search interval for g0 [m s⁻²].

    Returns
    -------
    dict with keys:
        g0_hat   – MLE of the acceleration scale [m s⁻²]
        g0_lo    – lower 68 % CI approximation (Δχ²=1) [m s⁻²]
        g0_hi    – upper 68 % CI approximation [m s⁻²]
        touches_lower – bool, True if log10(g0_hat) is within G0_BORDER_TOL
                        decades of the lower bound
        touches_upper – bool, True if log10(g0_hat) is within G0_BORDER_TOL
                        decades of the upper bound
        success  – bool, True if optimisation converged
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    # Optimizer works in log10 space: bounds are (-16.0, -8.0) for default G0_BOUNDS
    log10_lo = np.log10(bounds[0])
    log10_hi = np.log10(bounds[1])

    result = minimize_scalar(
        _neg_log_likelihood,
        bounds=(log10_lo, log10_hi),
        method="bounded",
        args=(g_bar, g_obs),
    )

    g0_hat = float(10.0 ** result.x)

    # Approximate CI from the curvature (Δχ²=1 ↔ 1σ for single parameter)
    delta_log10 = 0.5 / np.sqrt(max(result.fun, 1e-30))
    log10_hat = result.x
    g0_lo = float(10.0 ** np.clip(log10_hat - delta_log10, log10_lo, log10_hi))
    g0_hi = float(10.0 ** np.clip(log10_hat + delta_log10, log10_lo, log10_hi))

    # Sanity guard: detect if g0_hat is within G0_BORDER_TOL decades of either bound
    touches_lower, touches_upper = g0_touches_bounds(g0_hat, bounds)

    return {
        "g0_hat": g0_hat,
        "g0_lo": g0_lo,
        "g0_hi": g0_hi,
        "touches_lower": touches_lower,
        "touches_upper": touches_upper,
        "success": result.success if hasattr(result, "success") else True,
    }


# ---------------------------------------------------------------------------
# Telemetría del Ajuste
# ---------------------------------------------------------------------------

def print_telemetry(
    fit: dict,
    g_bar: np.ndarray,
) -> None:
    """Print SCM v0.2 fitting telemetry to stdout.

    Emits:
      [SCM v0.2] g0_hat=...  g0_lo=...  g0_hi=...
      [INFO] g_bar quantiles: q10=...  q50=...  q90=...  [m/s^2]
      [WARN] g0_hat hit LOWER bound again
      [WARN] g0_hat hit UPPER bound

    Parameters
    ----------
    fit   : dict returned by :func:`fit_g0`.
    g_bar : array of baryonic accelerations used in the fit [m s⁻²].
    """
    print(
        f"[SCM v0.2] g0_hat={fit['g0_hat']:.4e}"
        f"  g0_lo={fit['g0_lo']:.4e}"
        f"  g0_hi={fit['g0_hi']:.4e}"
    )

    q10, q50, q90 = np.quantile(g_bar, [0.10, 0.50, 0.90])
    print(
        f"[INFO] g_bar quantiles: q10={q10:.4e}  q50={q50:.4e}  q90={q90:.4e}  [m/s^2]"
    )

    if fit["touches_lower"]:
        print("[WARN] g0_hat hit LOWER bound again")
    if fit["touches_upper"]:
        print("[WARN] g0_hat hit UPPER bound")


# ---------------------------------------------------------------------------
# Residual computation and binning
# ---------------------------------------------------------------------------

def compute_log_residuals(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    a0: float = A0_SI,
) -> np.ndarray:
    """Log-residuals  Δ = log10(g_obs) − log10(g_pred).

    Parameters
    ----------
    g_bar : array_like
        Baryonic acceleration [m s⁻²].
    g_obs : array_like
        Observed centripetal acceleration [m s⁻²].
    a0 : float, optional
        Motor-de-Velos acceleration scale.

    Returns
    -------
    delta : ndarray
        Log-residuals in dex.
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)
    g_pred = g_pred_rar(g_bar, a0=a0)
    return np.log10(g_obs) - np.log10(g_pred)


def bin_residuals(
    g_bar: np.ndarray,
    delta: np.ndarray,
    n_bins: int = 15,
    min_count: int = 3,
) -> pd.DataFrame:
    """Bin log-residuals into equal-width log10(g_bar) bins.

    Parameters
    ----------
    g_bar     : array_like  Baryonic acceleration [m s⁻²].
    delta     : array_like  Log-residuals (same length as g_bar).
    n_bins    : int         Number of equal-width bins in log10(g_bar).
    min_count : int         Minimum number of points to keep a bin.

    Returns
    -------
    pd.DataFrame  Columns: g_bar_center, median_residual, mad_residual, count
    """
    g_bar = np.asarray(g_bar, dtype=float)
    delta = np.asarray(delta, dtype=float)

    log_gb = np.log10(g_bar)
    edges = np.linspace(log_gb.min(), log_gb.max(), n_bins + 1)

    rows = []
    for i in range(n_bins):
        mask = (log_gb >= edges[i]) & (log_gb < edges[i + 1])
        if i == n_bins - 1:
            mask = (log_gb >= edges[i]) & (log_gb <= edges[i + 1])
        count = int(mask.sum())
        if count < min_count:
            continue
        center_log = 0.5 * (edges[i] + edges[i + 1])
        median_r = float(np.median(delta[mask]))
        mad_r = float(np.median(np.abs(delta[mask] - median_r)))
        rows.append(
            {
                "g_bar_center": float(10.0 ** center_log),
                "median_residual": median_r,
                "mad_residual": mad_r,
                "count": count,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Load a g_bar / g_obs CSV (skipping comment lines starting with '#').

    Parameters
    ----------
    path : Path  CSV file with columns g_bar, g_obs (and optionally g_err).

    Returns
    -------
    pd.DataFrame  With at least columns g_bar, g_obs.

    Raises
    ------
    ValueError  If required columns are missing.
    """
    df = pd.read_csv(path, comment="#")
    required = {"g_bar", "g_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset {path} is missing required column(s): {sorted(missing)}"
        )
    return df


def write_csv(binned: pd.DataFrame, out_path: Path, n_bins_effective: int) -> None:
    """Write binned residuals CSV with header metadata.

    Parameters
    ----------
    binned          : DataFrame from :func:`bin_residuals`.
    out_path        : Destination path (parent directories are created).
    n_bins_effective: Number of bins that passed the min-count filter.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        f"# bins_effective: {n_bins_effective}",
        f"# g_bar_center_min: {binned['g_bar_center'].min():e}",
        f"# g_bar_center_max: {binned['g_bar_center'].max():e}",
    ]

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header_lines) + "\n")
        binned.to_csv(fh, index=False, float_format="%e")


# ---------------------------------------------------------------------------
# Reprint Notarial de Residuos
# ---------------------------------------------------------------------------

def reprint_notarial(binned: pd.DataFrame, out_path: Path) -> None:
    """Print the full binned-residuals table to stdout (Reprint Notarial).

    Parameters
    ----------
    binned   : DataFrame from :func:`bin_residuals`.
    out_path : Path where the CSV was written (printed in the header).
    """
    separator = "=" * 72
    print(separator)
    print(f"  REPRINT NOTARIAL DE RESIDUOS — {out_path}")
    print(separator)
    print(binned.to_string(index=False))
    print(separator)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/compute_residuals_binned.py",
        description=(
            "Bin SCM/RAR model residuals by g_bar and write diagnostics CSV.\n\n"
            "Prints fitting telemetry ([SCM v0.2], [INFO], [WARN]) and a full\n"
            "Reprint Notarial of the binned-residuals table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        metavar="PATH",
        default="data/sparc_rar_sample.csv",
        help="Input CSV with g_bar and g_obs columns (default: data/sparc_rar_sample.csv).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default="results/diagnostics/residuals_binned_v02.csv",
        help=(
            "Output CSV path "
            "(default: results/diagnostics/residuals_binned_v02.csv)."
        ),
    )
    parser.add_argument(
        "--bins",
        metavar="N",
        type=int,
        default=15,
        help="Number of log-spaced g_bar bins (default: 15).",
    )
    parser.add_argument(
        "--min-count",
        metavar="N",
        type=int,
        default=3,
        help="Minimum points per bin to include it (default: 3).",
    )
    parser.add_argument(
        "--a0",
        metavar="A0",
        type=float,
        default=None,
        help=(
            "Fixed acceleration scale [m/s^2].  If omitted, g0 is fitted "
            "from the data and telemetry is printed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    out_path = Path(args.out)

    # Load data
    df = load_dataset(data_path)
    g_bar = df["g_bar"].to_numpy()
    g_obs = df["g_obs"].to_numpy()

    # Fit g0 or use fixed value, then print telemetry
    if args.a0 is not None:
        a0_used = args.a0
    else:
        fit = fit_g0(g_bar, g_obs)
        print_telemetry(fit, g_bar)
        a0_used = fit["g0_hat"]

    # Compute log-residuals and bin
    delta = compute_log_residuals(g_bar, g_obs, a0=a0_used)
    binned = bin_residuals(g_bar, delta, n_bins=args.bins, min_count=args.min_count)

    # Write CSV
    write_csv(binned, out_path, n_bins_effective=len(binned))
    print(f"[INFO] Residuals written to {out_path}")

    # Reprint Notarial de Residuos
    reprint_notarial(binned, out_path)


if __name__ == "__main__":
    main()
