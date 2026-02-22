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
    [WARN] g0_hat tocó el límite <lower|upper>: revisa los bordes

The WARN line is only printed when the fitted g0 is within 1 % of
either bound of the search interval.

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
G0_BOUNDS: tuple[float, float] = (1e-12, 1e-8)

#: Fraction of bound proximity that triggers a WARN
G0_BORDER_TOL: float = 0.01


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

def _neg_log_likelihood(log_a0: float, g_bar: np.ndarray, g_obs: np.ndarray) -> float:
    """Negative log-likelihood for Gaussian errors on log-residuals.

    Minimising this w.r.t. log(a0) gives the MLE of the Motor-de-Velos
    acceleration scale.

    Parameters
    ----------
    log_a0 : float
        Natural log of the trial acceleration scale.
    g_bar  : ndarray
        Baryonic acceleration data [m s⁻²].
    g_obs  : ndarray
        Observed acceleration data [m s⁻²].

    Returns
    -------
    float  Negative log-likelihood (to be minimised).
    """
    a0 = np.exp(log_a0)
    g_pred = g_pred_rar(g_bar, a0=a0)
    # log-residuals
    delta = np.log10(g_obs) - np.log10(g_pred)
    return float(np.sum(delta ** 2))


def fit_g0(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    bounds: tuple[float, float] = G0_BOUNDS,
) -> dict:
    """Fit the Motor-de-Velos acceleration scale g0 from data.

    Uses scalar minimisation of the sum of squared log-residuals over
    log10(g0) in the interval ``bounds``.

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
        touches_lower – bool, True if g0_hat is within G0_BORDER_TOL of lower bound
        touches_upper – bool, True if g0_hat is within G0_BORDER_TOL of upper bound
        success  – bool, True if optimisation converged
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)

    log_lo = np.log(bounds[0])
    log_hi = np.log(bounds[1])

    result = minimize_scalar(
        _neg_log_likelihood,
        bounds=(log_lo, log_hi),
        method="bounded",
        args=(g_bar, g_obs),
    )

    g0_hat = float(np.exp(result.x))

    # Approximate CI from the curvature (Δχ²=1 ↔ 1σ for single parameter)
    delta_log = 0.5 / np.sqrt(max(result.fun, 1e-30))
    log_hat = result.x
    g0_lo = float(np.exp(np.clip(log_hat - delta_log, np.log(bounds[0]), np.log(bounds[1]))))
    g0_hi = float(np.exp(np.clip(log_hat + delta_log, np.log(bounds[0]), np.log(bounds[1]))))

    span_lo = bounds[0] * (1.0 + G0_BORDER_TOL)
    span_hi = bounds[1] * (1.0 - G0_BORDER_TOL)

    return {
        "g0_hat": g0_hat,
        "g0_lo": g0_lo,
        "g0_hi": g0_hi,
        "touches_lower": g0_hat <= span_lo,
        "touches_upper": g0_hat >= span_hi,
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
      [WARN] g0_hat tocó el límite <lower|upper>: revisa los bordes

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
        print(
            "[WARN] g0_hat tocó el límite lower: revisa los bordes "
            f"(g0_hat={fit['g0_hat']:.4e} ≈ lower bound {G0_BOUNDS[0]:.2e})"
        )
    if fit["touches_upper"]:
        print(
            "[WARN] g0_hat tocó el límite upper: revisa los bordes "
            f"(g0_hat={fit['g0_hat']:.4e} ≈ upper bound {G0_BOUNDS[1]:.2e})"
        )


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
