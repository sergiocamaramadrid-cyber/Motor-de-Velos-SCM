"""
scripts/sparc_spectral_analysis.py — Spatial spectral analysis of SPARC rotation curves.

For each SPARC galaxy rotation curve (*_rotmod.dat), this script:

1. Reads radius (kpc) and observed circular velocity (km/s).
2. Interpolates to a uniform radial grid.
3. Fits a smooth background with an adaptive Savitzky-Golay filter.
4. Computes the spatial power spectrum of the residuals via real FFT.
5. Detects the dominant spatial frequency and derives:
   - ``lambda_dom_kpc``: dominant spatial wavelength (kpc)
   - ``n_peaks``: number of significant spectral peaks
   - ``peak_power``: power at the dominant peak
   - ``residual_rms_kms``: RMS of the velocity residuals (km/s)

Output columns
--------------
galaxy            — galaxy name
n_points_raw      — number of raw data points after quality cuts
rmin_kpc          — minimum observed radius (kpc)
rmax_kpc          — maximum observed radius (kpc)
n_grid            — number of uniform-grid points used for FFT
residual_rms_kms  — RMS of velocity residuals after SG smoothing (km/s)
lambda_dom_kpc    — dominant spatial wavelength 1/f_dom (kpc); NaN if not found
peak_freq_1perkpc — dominant spatial frequency (1/kpc)
peak_power        — power spectral density at dominant peak
n_peaks           — number of significant peaks detected

Usage
-----
::

    python scripts/sparc_spectral_analysis.py \\
        --sparc-dir data/SPARC \\
        --out data/sparc_spectral_catalog.csv

With per-galaxy panel plots::

    python scripts/sparc_spectral_analysis.py \\
        --sparc-dir data/SPARC \\
        --out data/sparc_spectral_catalog.csv \\
        --plot-dir results/sparc_spectral/

Notes
-----
* The rotmod files are expected to follow the SPARC naming convention
  ``<galaxy>_rotmod.dat`` inside *sparc_dir* or *sparc_dir/rotmod/*.
* Only the first two numeric columns (radius, velocity) are used; the file
  format is the standard SPARC whitespace-separated text with ``#`` comments.
* Galaxies with fewer than ``--min-points`` valid data points are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import irfft, rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, savgol_filter

# ---------------------------------------------------------------------------
# Constants (physical / algorithmic)
# ---------------------------------------------------------------------------

#: Fraction of ``n_grid`` used as Savitzky-Golay window (before rounding).
SMOOTH_WINDOW_FRAC: float = 0.12

#: Minimum allowed Savitzky-Golay window length (must be ≥ polyorder + 1).
SMOOTH_WINDOW_MIN: int = 5

#: Savitzky-Golay polynomial order.
SG_POLYORDER: int = 3

#: Minimum number of valid raw data points to process a galaxy.
MIN_POINTS_DEFAULT: int = 20

#: Minimum number of grid points (lower bound for ngrid).
NGRID_MIN: int = 128

#: Maximum number of grid points (upper bound for ngrid).
NGRID_MAX: int = 512

#: Interpolation kind passed to :class:`scipy.interpolate.interp1d`.
INTERP_KIND: str = "linear"

#: Spectral peak-detection: height threshold = ``PEAK_HEIGHT_FACTOR × median``.
PEAK_HEIGHT_FACTOR: float = 3.0

#: Spectral peak-detection: percentile of power used for secondary threshold.
PEAK_HEIGHT_PERCENTILE: float = 75.0

#: Minimum spatial frequency (1/kpc) expressed as 1 / (fraction × r_range).
LAMBDA_MAX_FACTOR: float = 0.8

#: Minimum detectable wavelength (kpc) → maximum spatial frequency.
LAMBDA_MIN_KPC: float = 0.5

OUTPUT_COLUMNS = [
    "galaxy",
    "n_points_raw",
    "rmin_kpc",
    "rmax_kpc",
    "n_grid",
    "residual_rms_kms",
    "lambda_dom_kpc",
    "peak_freq_1perkpc",
    "peak_power",
    "n_peaks",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def galaxy_name_from_path(file_path: str | Path) -> str:
    """Extract a galaxy name from a SPARC rotmod file path.

    The function strips common SPARC filename decorators (``_rotmod``,
    ``rotmod_``) from the stem and returns the cleaned name.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    str
    """
    stem = Path(file_path).stem
    stem = stem.replace("_rotmod", "").replace("rotmod_", "")
    return stem.strip()


def _find_rotmod_files(sparc_dir: str | Path) -> list[Path]:
    """Return all ``*rotmod*.dat`` files found under *sparc_dir*."""
    sparc_dir = Path(sparc_dir)
    files: list[Path] = []
    for pattern in ("*_rotmod.dat", "rotmod_*.dat", "*rotmod*.dat"):
        files.extend(sparc_dir.rglob(pattern))
    # Deduplicate by absolute path
    return sorted({f.resolve(): f for f in files}.values())


def parse_rotmod(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a SPARC rotmod file and return (radius, velocity) arrays.

    Only the first two numeric columns are used (radius in kpc, velocity in
    km/s).  Lines starting with ``#`` and blank lines are skipped.  Rows
    where either column is non-finite, non-positive, or cannot be parsed are
    dropped silently.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    r : np.ndarray
        Radii in kpc (finite, > 0).
    v : np.ndarray
        Circular velocities in km/s (finite, > 0).
    """
    r_vals: list[float] = []
    v_vals: list[float] = []
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                r_vals.append(float(parts[0]))
                v_vals.append(float(parts[1]))
            except ValueError:
                continue

    r = np.array(r_vals, dtype=float)
    v = np.array(v_vals, dtype=float)
    mask = np.isfinite(r) & np.isfinite(v) & (r > 0.0) & (v > 0.0)
    return r[mask], v[mask]


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------


def compute_spectral_features(
    r: np.ndarray,
    v: np.ndarray,
    *,
    smooth_window_frac: float = SMOOTH_WINDOW_FRAC,
    smooth_window_min: int = SMOOTH_WINDOW_MIN,
    sg_polyorder: int = SG_POLYORDER,
    ngrid_min: int = NGRID_MIN,
    ngrid_max: int = NGRID_MAX,
    interp_kind: str = INTERP_KIND,
    peak_height_factor: float = PEAK_HEIGHT_FACTOR,
    peak_height_percentile: float = PEAK_HEIGHT_PERCENTILE,
    lambda_max_factor: float = LAMBDA_MAX_FACTOR,
    lambda_min_kpc: float = LAMBDA_MIN_KPC,
) -> dict:
    """Compute spatial spectral features from a rotation curve.

    Parameters
    ----------
    r, v : np.ndarray
        Sorted (ascending r) radius (kpc) and velocity (km/s) arrays.
        Both must be finite and positive; the caller is responsible for
        sorting and basic quality cuts.
    smooth_window_frac : float
        Fraction of ``n_grid`` used as the Savitzky-Golay window length.
    smooth_window_min : int
        Minimum SG window length.
    sg_polyorder : int
        SG polynomial order.
    ngrid_min, ngrid_max : int
        Bounds on the uniform-grid size.
    interp_kind : str
        Interpolation kind (see :class:`scipy.interpolate.interp1d`).
    peak_height_factor : float
        Significant peaks must exceed ``peak_height_factor × median(power)``.
    peak_height_percentile : float
        Secondary height threshold: ``np.percentile(power, peak_height_percentile)``.
    lambda_max_factor : float
        Maximum wavelength = ``lambda_max_factor × r_range``.
    lambda_min_kpc : float
        Minimum detectable wavelength in kpc.

    Returns
    -------
    dict with keys:
        ``n_grid``, ``residual_rms_kms``, ``lambda_dom_kpc``,
        ``peak_freq_1perkpc``, ``peak_power``, ``n_peaks``.
        ``lambda_dom_kpc`` is ``np.nan`` when no valid frequency range exists.

    Raises
    ------
    ValueError
        If ``r`` has fewer than ``smooth_window_min + 2`` elements or the SG
        window cannot be constructed.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    if len(r) < smooth_window_min + 2:
        raise ValueError(
            f"Too few points ({len(r)}) for spectral analysis "
            f"(need ≥ {smooth_window_min + 2})."
        )

    # Sort ascending
    order = np.argsort(r)
    r, v = r[order], v[order]

    # Uniform radial grid
    ngrid = int(np.clip(len(r) * 4, ngrid_min, ngrid_max))
    ru = np.linspace(r.min(), r.max(), ngrid)
    f_interp = interp1d(r, v, kind=interp_kind, fill_value="extrapolate")
    vu = f_interp(ru)

    # Adaptive Savitzky-Golay window (must be odd, ≥ smooth_window_min,
    # ≤ ngrid − 1)
    win = max(smooth_window_min, int(ngrid * smooth_window_frac))
    if win % 2 == 0:
        win += 1
    win = min(win, ngrid - 1)
    if win < smooth_window_min:
        raise ValueError(
            f"SG window too small ({win}); ngrid={ngrid}, "
            f"smooth_window_frac={smooth_window_frac}."
        )

    smooth = savgol_filter(vu, window_length=win, polyorder=sg_polyorder,
                           mode="interp")
    resid = vu - smooth
    residual_rms = float(np.sqrt(np.mean(resid ** 2)))

    # Real FFT of mean-subtracted residuals
    dr = float(np.median(np.diff(ru)))
    yf = rfft(resid - np.mean(resid))
    freq = rfftfreq(ngrid, d=dr)  # 1/kpc

    # Physical frequency range
    r_range = float(r.max() - r.min())
    f_min = (1.0 / (lambda_max_factor * r_range)) if r_range > 0.0 else 0.0
    f_max = 1.0 / lambda_min_kpc if lambda_min_kpc > 0.0 else np.inf

    keep = (freq > 0.0) & (freq >= f_min) & (freq <= f_max)
    if not np.any(keep):
        return {
            "n_grid": ngrid,
            "residual_rms_kms": residual_rms,
            "lambda_dom_kpc": np.nan,
            "peak_freq_1perkpc": np.nan,
            "peak_power": np.nan,
            "n_peaks": 0,
        }

    freq_filt = freq[keep]
    power = np.abs(yf[keep]) ** 2

    # Peak detection
    floor = float(np.median(power))
    height_threshold = max(
        peak_height_factor * floor,
        float(np.percentile(power, peak_height_percentile)),
    )
    peaks, _ = find_peaks(power, height=height_threshold)
    n_peaks = int(len(peaks))

    if n_peaks == 0:
        dom_idx = int(np.argmax(power))
    else:
        dom_idx = int(peaks[int(np.argmax(power[peaks]))])

    f_dom = float(freq_filt[dom_idx])
    peak_power = float(power[dom_idx])
    lambda_dom = float(1.0 / f_dom) if f_dom > 0.0 else np.nan

    return {
        "n_grid": ngrid,
        "residual_rms_kms": residual_rms,
        "lambda_dom_kpc": lambda_dom,
        "peak_freq_1perkpc": f_dom,
        "peak_power": peak_power,
        "n_peaks": n_peaks,
    }


# ---------------------------------------------------------------------------
# Per-galaxy panel plot
# ---------------------------------------------------------------------------


def _plot_galaxy_panel(
    galaxy: str,
    ru: np.ndarray,
    vu: np.ndarray,
    smooth: np.ndarray,
    resid: np.ndarray,
    freq_filt: np.ndarray,
    power: np.ndarray,
    features: dict,
    out_path: Path,
) -> None:
    """Save a 2×2 diagnostic panel for one galaxy."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    lambda_dom = features["lambda_dom_kpc"]
    f_dom = features["peak_freq_1perkpc"]

    # Top-left: rotation curve
    ax = axes[0, 0]
    ax.plot(ru, vu, "k-", lw=1.5, label="Interpolated")
    ax.plot(ru, smooth, "r--", lw=1.2, label="SG smooth")
    ax.plot(ru, resid, color="gray", alpha=0.6, label="Residual")
    if np.isfinite(lambda_dom):
        ax.axvline(lambda_dom, color="b", ls=":", lw=1,
                   label=f"λ_dom = {lambda_dom:.2f} kpc")
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel("V (km/s)")
    ax.set_title(f"{galaxy} — rotation curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: power spectrum
    ax = axes[0, 1]
    if len(freq_filt) > 0 and np.any(power > 0):
        ax.loglog(freq_filt, power, "b-", lw=1)
        if np.isfinite(f_dom):
            ax.axvline(f_dom, color="r", ls="--",
                       label=f"f_dom = {f_dom:.3f} 1/kpc")
    ax.set_xlabel("Spatial frequency (1/kpc)")
    ax.set_ylabel("Power")
    ax.set_title("Power spectrum (residuals)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: residual histogram
    ax = axes[1, 0]
    ax.hist(resid, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("ΔV (km/s)")
    ax.set_ylabel("Count")
    rms = features["residual_rms_kms"]
    ax.set_title(f"Residuals — RMS = {rms:.2f} km/s")
    ax.grid(True, alpha=0.3)

    # Bottom-right: text summary
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"galaxy:          {galaxy}\n"
        f"n_peaks:         {features['n_peaks']}\n"
        f"lambda_dom_kpc:  {lambda_dom:.2f}\n" if np.isfinite(lambda_dom)
        else f"lambda_dom_kpc:  NaN\n"
    )
    summary = (
        f"galaxy:          {galaxy}\n"
        f"n_peaks:         {features['n_peaks']}\n"
    )
    if np.isfinite(lambda_dom):
        summary += f"lambda_dom_kpc:  {lambda_dom:.2f}\n"
    else:
        summary += "lambda_dom_kpc:  NaN\n"
    summary += f"residual_rms:    {rms:.2f} km/s\n"
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            va="top", fontsize=10, family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------


def build_spectral_catalog(
    sparc_dir: str | Path,
    out: str | Path,
    *,
    plot_dir: str | Path | None = None,
    min_points: int = MIN_POINTS_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the SPARC spatial spectral catalog.

    Processes all ``*_rotmod.dat`` files found under *sparc_dir*, computes
    spectral features for each galaxy, and writes a CSV catalog.

    Parameters
    ----------
    sparc_dir : str or Path
        Root SPARC data directory.  Rotmod files are searched recursively.
    out : str or Path
        Output CSV path.
    plot_dir : str or Path or None
        If given, per-galaxy 2×2 diagnostic panels are saved there as PNG.
    min_points : int
        Minimum valid data points required to process a galaxy.
    verbose : bool
        Print progress if True.

    Returns
    -------
    pd.DataFrame
        Catalog with ``OUTPUT_COLUMNS``.
    """
    sparc_dir = Path(sparc_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    files = _find_rotmod_files(sparc_dir)

    # Deduplicate: keep first occurrence per galaxy name
    seen: dict[str, Path] = {}
    for f in files:
        name = galaxy_name_from_path(f)
        if name not in seen:
            seen[name] = f
    files = [seen[name] for name in sorted(seen)]

    if verbose:
        print(f"SPARC rotmod files found: {len(files)}")

    rows: list[dict] = []
    n_skip = 0
    n_error = 0

    for idx, file_path in enumerate(files):
        galaxy = galaxy_name_from_path(file_path)
        try:
            r_raw, v_raw = parse_rotmod(file_path)
            n_raw = int(len(r_raw))

            if n_raw < min_points:
                n_skip += 1
                if verbose:
                    print(
                        f"  [{idx + 1}/{len(files)}] {galaxy}: "
                        f"too few points ({n_raw}) — skipped"
                    )
                continue

            features = compute_spectral_features(r_raw, v_raw)

            row = {
                "galaxy": galaxy,
                "n_points_raw": n_raw,
                "rmin_kpc": float(r_raw.min()),
                "rmax_kpc": float(r_raw.max()),
                **features,
            }
            rows.append(row)

            if verbose:
                lam = features["lambda_dom_kpc"]
                lam_str = f"{lam:.2f}" if np.isfinite(lam) else "NaN"
                print(
                    f"  [{idx + 1}/{len(files)}] {galaxy}: "
                    f"λ_dom={lam_str} kpc | "
                    f"rms={features['residual_rms_kms']:.2f} km/s | "
                    f"n_peaks={features['n_peaks']}"
                )

            # Optional panel plot
            if plot_dir is not None:
                _save_panel(galaxy, r_raw, v_raw, features, plot_dir)

        except Exception as exc:  # noqa: BLE001
            n_error += 1
            if verbose:
                print(f"  [!] Error processing {galaxy}: {exc}")

    if rows:
        catalog = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    else:
        catalog = pd.DataFrame(columns=OUTPUT_COLUMNS)

    catalog = catalog.sort_values("galaxy").reset_index(drop=True)
    catalog.to_csv(out, index=False)

    if verbose:
        good = catalog["lambda_dom_kpc"].notna().sum()
        print(
            f"\nCatalog written to {out}  "
            f"({len(catalog)} galaxies processed, "
            f"{good} with valid λ_dom, "
            f"{n_skip} skipped, {n_error} errors)"
        )

    return catalog


def _save_panel(
    galaxy: str,
    r_raw: np.ndarray,
    v_raw: np.ndarray,
    features: dict,
    plot_dir: Path,
) -> None:
    """Recompute intermediate arrays and save the diagnostic panel."""
    # Recompute uniform grid and smooth for plotting
    ngrid = features["n_grid"]
    order = np.argsort(r_raw)
    r_s, v_s = r_raw[order], v_raw[order]
    ru = np.linspace(r_s.min(), r_s.max(), ngrid)
    f_interp = interp1d(r_s, v_s, kind=INTERP_KIND, fill_value="extrapolate")
    vu = f_interp(ru)

    win = max(SMOOTH_WINDOW_MIN, int(ngrid * SMOOTH_WINDOW_FRAC))
    if win % 2 == 0:
        win += 1
    win = min(win, ngrid - 1)
    smooth = savgol_filter(vu, window_length=win, polyorder=SG_POLYORDER,
                           mode="interp")
    resid = vu - smooth

    dr = float(np.median(np.diff(ru)))
    yf = rfft(resid - np.mean(resid))
    freq = rfftfreq(ngrid, d=dr)
    r_range = float(r_s.max() - r_s.min())
    f_min = (1.0 / (LAMBDA_MAX_FACTOR * r_range)) if r_range > 0.0 else 0.0
    f_max = 1.0 / LAMBDA_MIN_KPC
    keep = (freq > 0.0) & (freq >= f_min) & (freq <= f_max)
    freq_filt = freq[keep]
    power = np.abs(yf[keep]) ** 2

    out_path = plot_dir / f"SCM_panel_{galaxy}.png"
    _plot_galaxy_panel(galaxy, ru, vu, smooth, resid,
                       freq_filt, power, features, out_path)


# ---------------------------------------------------------------------------
# Summary statistics and figures
# ---------------------------------------------------------------------------


def print_summary(catalog: pd.DataFrame) -> None:
    """Print summary statistics to stdout."""
    good = catalog.dropna(subset=["lambda_dom_kpc"])
    print("\n--- SCM SPECTRAL STATISTICS ---")
    print(f"Total galaxies:         {len(catalog)}")
    print(f"With valid λ_dom:       {len(good)}")
    if len(good) > 0:
        print(f"median λ_dom_kpc:       {good['lambda_dom_kpc'].median():.2f}")
        print(f"median n_peaks:         {good['n_peaks'].median():.1f}")
        print(f"median residual_rms:    {good['residual_rms_kms'].median():.2f} km/s")


def generate_summary_figures(catalog: pd.DataFrame, out_dir: Path) -> None:
    """Save summary histogram and scatter plot."""
    good = catalog.dropna(subset=["lambda_dom_kpc"])
    if len(good) == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Histogram of dominant wavelength
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(good["lambda_dom_kpc"], bins=20, color="steelblue",
            edgecolor="black", alpha=0.8)
    ax.set_xlabel("λ_dom (kpc)")
    ax.set_ylabel("Number of galaxies")
    ax.set_title("SPARC dominant spatial wavelength distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_dom_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scatter: n_points vs lambda_dom, coloured by residual RMS
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        good["n_points_raw"], good["lambda_dom_kpc"],
        c=good["residual_rms_kms"], cmap="viridis", alpha=0.8,
    )
    fig.colorbar(sc, ax=ax, label="Residual RMS (km/s)")
    ax.set_xlabel("N raw points")
    ax.set_ylabel("λ_dom (kpc)")
    ax.set_title("SPARC: dominant wavelength vs data density")
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_dom_vs_npoints.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial spectral analysis of SPARC rotation curves."
    )
    parser.add_argument(
        "--sparc-dir", default="data/SPARC", metavar="DIR",
        help="Root SPARC directory with *_rotmod.dat files (default: data/SPARC).",
    )
    parser.add_argument(
        "--out", default="data/sparc_spectral_catalog.csv",
        help="Output CSV path (default: data/sparc_spectral_catalog.csv).",
    )
    parser.add_argument(
        "--plot-dir", default=None, metavar="DIR",
        help="Directory for per-galaxy panel PNGs (omit to skip plots).",
    )
    parser.add_argument(
        "--min-points", type=int, default=MIN_POINTS_DEFAULT, metavar="N",
        help=f"Minimum raw data points per galaxy (default: {MIN_POINTS_DEFAULT}).",
    )
    parser.add_argument(
        "--no-summary-figures", action="store_true",
        help="Skip writing summary histogram and scatter plot.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    """Parse arguments, run the spectral catalog builder, return summary dict.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    dict with keys:
        ``n_galaxies``, ``n_valid``, ``median_lambda_dom_kpc``,
        ``median_n_peaks``, ``median_residual_rms_kms``, ``out_path``,
        ``catalog`` (DataFrame).
    """
    args = _parse_args(argv)

    catalog = build_spectral_catalog(
        sparc_dir=args.sparc_dir,
        out=args.out,
        plot_dir=args.plot_dir,
        min_points=args.min_points,
        verbose=not args.quiet,
    )

    good = catalog.dropna(subset=["lambda_dom_kpc"])

    if not args.quiet:
        print_summary(catalog)

    if not args.no_summary_figures:
        out_dir = Path(args.out).parent
        generate_summary_figures(catalog, out_dir)

    return {
        "n_galaxies": int(len(catalog)),
        "n_valid": int(len(good)),
        "median_lambda_dom_kpc": (float(good["lambda_dom_kpc"].median())
                                  if len(good) > 0 else float("nan")),
        "median_n_peaks": (float(good["n_peaks"].median())
                           if len(good) > 0 else float("nan")),
        "median_residual_rms_kms": (float(good["residual_rms_kms"].median())
                                    if len(good) > 0 else float("nan")),
        "out_path": str(args.out),
        "catalog": catalog,
    }


if __name__ == "__main__":
    result = main()
    sys.exit(0)
