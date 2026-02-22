"""
sensitivity.py — Sensitivity analysis for the Motor de Velos SCM framework.

Tests how the pipeline results change as the characteristic acceleration *a0*
is varied around its fiducial value.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .scm_models import v_total, chi2_reduced
from .scm_analysis import fit_galaxy, load_rotation_curve, load_galaxy_table


def a0_sensitivity(rc, a0_values):
    """Compute reduced chi-squared as a function of *a0* for one galaxy.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data (output of :func:`~scm_analysis.load_rotation_curve`).
    a0_values : array_like
        Sequence of *a0* values (m/s²) to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: ``['a0', 'chi2_reduced', 'upsilon_disk']``.
    """
    records = []
    for a0 in a0_values:
        fit = fit_galaxy(rc, a0=a0)
        records.append({
            "a0": float(a0),
            "chi2_reduced": fit["chi2"],
            "upsilon_disk": fit["upsilon_disk"],
        })
    return pd.DataFrame(records)


def run_sensitivity(data_dir, out_dir,
                    a0_min=0.5e-10, a0_max=3.0e-10, n_steps=11,
                    max_galaxies=None, verbose=True):
    """Run a sensitivity analysis varying *a0* over a grid.

    For each galaxy in the SPARC sample (up to *max_galaxies*) the reduced
    chi-squared is computed at each grid point.  The median chi-squared across
    galaxies is reported for every *a0* value.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SPARC data.
    out_dir : str or Path
        Output directory for sensitivity results.
    a0_min, a0_max : float
        Range of *a0* values to explore (m/s²).
    n_steps : int
        Number of logarithmically-spaced *a0* grid points.
    max_galaxies : int or None
        Limit the number of galaxies processed (useful for quick runs).
    verbose : bool
        Print progress if True.

    Returns
    -------
    pd.DataFrame
        Sensitivity summary with columns ``['a0', 'chi2_median', 'chi2_mean']``.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a0_grid = np.logspace(
        np.log10(a0_min), np.log10(a0_max), n_steps
    )

    galaxy_table = load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()
    if max_galaxies is not None:
        galaxy_names = galaxy_names[:max_galaxies]

    # chi2[i_galaxy][i_a0]
    all_chi2 = []
    for name in galaxy_names:
        try:
            rc = load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            continue
        sens = a0_sensitivity(rc, a0_grid)
        all_chi2.append(sens["chi2_reduced"].values)
        if verbose:
            print(f"  sensitivity: {name}")

    if not all_chi2:
        return pd.DataFrame(columns=["a0", "chi2_median", "chi2_mean"])

    chi2_matrix = np.array(all_chi2)  # shape (n_galaxies, n_steps)
    summary = pd.DataFrame({
        "a0": a0_grid,
        "chi2_median": np.median(chi2_matrix, axis=0),
        "chi2_mean": np.mean(chi2_matrix, axis=0),
    })

    out_path = out_dir / "sensitivity_a0.csv"
    summary.to_csv(out_path, index=False)
    if verbose:
        print(f"Sensitivity results written to {out_path}")

    return summary


def bootstrap_chi2(rc, a0=1.2e-10, n_boot=200, seed=42):
    """Estimate chi-squared uncertainty via bootstrap resampling.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data for a single galaxy.
    a0 : float
        Characteristic velos acceleration.
    n_boot : int
        Number of bootstrap samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``chi2_mean``, ``chi2_std``, ``chi2_samples``.
    """
    rng = np.random.default_rng(seed)
    n = len(rc)
    chi2_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rc_boot = rc.iloc[idx].reset_index(drop=True)
        fit = fit_galaxy(rc_boot, a0=a0)
        chi2_samples.append(fit["chi2"])
    chi2_samples = np.array(chi2_samples)
    return {
        "chi2_mean": float(np.mean(chi2_samples)),
        "chi2_std": float(np.std(chi2_samples)),
        "chi2_samples": chi2_samples,
    }
