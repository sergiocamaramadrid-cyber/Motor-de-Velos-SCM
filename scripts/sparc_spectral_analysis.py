"""
scripts/sparc_spectral_analysis.py — Spatial spectral analysis of SPARC rotation curves.

Performs FFT-based spatial spectral analysis of SPARC rotmod rotation curve
files.  For each galaxy the script:

  1. Reads the rotmod file (columns: Rad, Vobs, errV, Vgas, Vdisk, Vbul,
     SBdisk, SBbul).
  2. Interpolates the observed rotation curve onto a uniform radial grid.
  3. Computes the FFT of the gridded velocity profile.
  4. Extracts spectral features: dominant wavelength, peak frequency, peak
     power, and number of significant peaks.

Usage
-----
Basic run::

    python scripts/sparc_spectral_analysis.py \\
        --sparc-dir data/SPARC \\
        --out results/spectral/sparc_spectral_catalog.csv

With optional plot output::

    python scripts/sparc_spectral_analysis.py \\
        --sparc-dir data/SPARC \\
        --out results/spectral/sparc_spectral_catalog.csv \\
        --plot-dir results/spectral/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Galaxy name extraction
# ---------------------------------------------------------------------------

def galaxy_name_from_path(path: str | Path) -> str:
    """Extract the galaxy name from a SPARC rotmod file path.

    Parameters
    ----------
    path : str or Path
        Path to a rotmod file, e.g. ``/data/SPARC/DDO064_rotmod.dat``.

    Returns
    -------
    str
        Galaxy name, e.g. ``DDO064``.
    """
    stem = Path(path).stem  # e.g. "DDO064_rotmod"
    if stem.endswith("_rotmod"):
        return stem[: -len("_rotmod")]
    return stem


# ---------------------------------------------------------------------------
# Rotmod file parsing
# ---------------------------------------------------------------------------

def parse_rotmod(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a SPARC rotmod file and return radial positions and velocities.

    The rotmod format has columns::

        Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul

    Comment lines starting with ``#`` are skipped.

    Parameters
    ----------
    path : str or Path
        Path to the ``*_rotmod.dat`` file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(r_kpc, v_obs_kms)`` — radial positions in kpc and observed
        circular velocities in km/s.
    """
    path = Path(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    r_kpc = data[:, 0]
    v_obs_kms = data[:, 1]
    return r_kpc, v_obs_kms


# ---------------------------------------------------------------------------
# Spectral feature computation
# ---------------------------------------------------------------------------

def compute_spectral_features(
    r: np.ndarray,
    v: np.ndarray,
    n_grid: int = 256,
    **kwargs,
) -> dict:
    """Compute FFT-based spectral features of a rotation curve.

    The velocity profile is interpolated onto a uniform radial grid before
    the FFT is applied.  The linear trend of *v(r)* is removed to obtain
    the residual RMS.

    Parameters
    ----------
    r : np.ndarray
        Radial positions in kpc (not necessarily uniform).
    v : np.ndarray
        Observed circular velocities in km/s.
    n_grid : int, optional
        Number of points in the uniform interpolation grid (default: 256).
    **kwargs
        Accepted for forward-compatibility; currently unused.

    Returns
    -------
    dict with keys:
        n_points_raw      — number of input data points
        rmin_kpc          — minimum radius (kpc)
        rmax_kpc          — maximum radius (kpc)
        n_grid            — grid size used for FFT
        residual_rms_kms  — RMS of v minus a linear fit to v(r)
        lambda_dom_kpc    — dominant wavelength (kpc), 1/peak_freq
        peak_freq_1perkpc — spatial frequency of dominant peak (1/kpc)
        peak_power        — power at dominant peak
        n_peaks           — number of peaks with power > 10 % of peak_power
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    n_points_raw = int(len(r))
    rmin_kpc = float(r.min())
    rmax_kpc = float(r.max())

    # Linear fit to compute residual RMS
    if n_points_raw >= 2:
        slope, intercept, *_ = linregress(r, v)
        v_linear = slope * r + intercept
        residual_rms_kms = float(np.sqrt(np.mean((v - v_linear) ** 2)))
    else:
        residual_rms_kms = 0.0

    # Uniform grid spacing; fallback to 1.0 if radial span is zero
    if rmax_kpc > rmin_kpc:
        dr = (rmax_kpc - rmin_kpc) / max(n_grid - 1, 1)
    else:
        dr = 1.0

    r_grid = np.linspace(rmin_kpc, rmax_kpc, n_grid)

    # Interpolate onto uniform grid (extrapolate at boundaries)
    if n_points_raw >= 2 and rmax_kpc > rmin_kpc:
        interp_fn = interp1d(
            r, v, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        v_grid = interp_fn(r_grid)
    else:
        v_grid = np.full(n_grid, v[0] if n_points_raw >= 1 else 0.0)

    # FFT (real input → positive frequencies only)
    fft_vals = np.fft.rfft(v_grid)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n_grid, d=dr)  # units: 1/kpc

    # Ignore DC component (index 0) to focus on oscillatory content
    if len(power) > 1:
        power_ac = power[1:]
        freqs_ac = freqs[1:]
    else:
        power_ac = power
        freqs_ac = freqs

    if len(power_ac) == 0 or power_ac.max() == 0:
        peak_freq = 0.0
        peak_power = 0.0
        lambda_dom = float("inf")
        n_peaks = 0
    else:
        peak_idx = int(np.argmax(power_ac))
        peak_freq = float(freqs_ac[peak_idx])
        peak_power = float(power_ac[peak_idx])
        lambda_dom = float(1.0 / peak_freq) if peak_freq > 0 else float("inf")
        n_peaks = int(np.sum(power_ac > 0.10 * peak_power))

    return {
        "n_points_raw": n_points_raw,
        "rmin_kpc": rmin_kpc,
        "rmax_kpc": rmax_kpc,
        "n_grid": n_grid,
        "residual_rms_kms": residual_rms_kms,
        "lambda_dom_kpc": lambda_dom,
        "peak_freq_1perkpc": peak_freq,
        "peak_power": peak_power,
        "n_peaks": n_peaks,
    }


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

_CATALOG_COLS = [
    "galaxy", "n_points_raw", "rmin_kpc", "rmax_kpc", "n_grid",
    "residual_rms_kms", "lambda_dom_kpc", "peak_freq_1perkpc",
    "peak_power", "n_peaks",
]


def build_spectral_catalog(
    sparc_dir: str | Path,
    out: str | Path,
    plot_dir: str | Path | None = None,
    min_points: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """Scan a directory for SPARC rotmod files and build a spectral catalog.

    Parameters
    ----------
    sparc_dir : str or Path
        Directory containing ``*_rotmod.dat`` files.
    out : str or Path
        Output CSV path for the catalog.
    plot_dir : str, Path, or None
        If given, reserved for per-galaxy power-spectrum plots (future use).
    min_points : int
        Minimum number of data points required to process a galaxy
        (default: 5).
    verbose : bool
        Print progress if True.

    Returns
    -------
    pd.DataFrame
        Spectral catalog with columns:
        galaxy, n_points_raw, rmin_kpc, rmax_kpc, n_grid,
        residual_rms_kms, lambda_dom_kpc, peak_freq_1perkpc,
        peak_power, n_peaks
    """
    sparc_dir = Path(sparc_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rotmod_files = sorted(sparc_dir.glob("*_rotmod.dat"))
    if verbose:
        print(f"Found {len(rotmod_files)} rotmod files in {sparc_dir}")

    records = []
    for fpath in rotmod_files:
        name = galaxy_name_from_path(fpath)
        try:
            r, v = parse_rotmod(fpath)
        except Exception as exc:
            if verbose:
                print(f"  [skip] {name}: parse error — {exc}")
            continue

        if len(r) < min_points:
            if verbose:
                print(f"  [skip] {name}: only {len(r)} points (< {min_points})")
            continue

        features = compute_spectral_features(r, v)
        features["galaxy"] = name
        records.append(features)

        if verbose:
            print(
                f"  {name}: λ_dom={features['lambda_dom_kpc']:.2f} kpc, "
                f"n_peaks={features['n_peaks']}"
            )

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=_CATALOG_COLS)
    else:
        df = df[_CATALOG_COLS]

    df = df.sort_values("galaxy").reset_index(drop=True)
    df.to_csv(out, index=False)

    if verbose:
        print(f"\nSpectral catalog written to {out}  ({len(df)} galaxies)")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FFT-based spatial spectral analysis of SPARC rotmod rotation curves."
        )
    )
    parser.add_argument(
        "--sparc-dir", required=True, dest="sparc_dir",
        help="Directory containing SPARC *_rotmod.dat files.",
    )
    parser.add_argument(
        "--out", default="results/spectral/sparc_spectral_catalog.csv",
        help="Output CSV path (default: results/spectral/sparc_spectral_catalog.csv).",
    )
    parser.add_argument(
        "--plot-dir", default=None, dest="plot_dir",
        help="Optional directory for per-galaxy power-spectrum plots.",
    )
    parser.add_argument(
        "--min-points", type=int, default=5, dest="min_points",
        help="Minimum data points per galaxy (default: 5).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Entry point: parse arguments and run spectral catalog generation.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments (``sys.argv[1:]`` if None).

    Returns
    -------
    dict with keys:
        catalog  — resulting pd.DataFrame
        n        — number of galaxies processed
        out_path — path to the written CSV (str)
    """
    args = _parse_args(argv)
    catalog = build_spectral_catalog(
        sparc_dir=args.sparc_dir,
        out=args.out,
        plot_dir=args.plot_dir,
        min_points=args.min_points,
        verbose=not args.quiet,
    )
    return {
        "catalog": catalog,
        "n": len(catalog),
        "out_path": str(args.out),
    }


if __name__ == "__main__":
    main()
