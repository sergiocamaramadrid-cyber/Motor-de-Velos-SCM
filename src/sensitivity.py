"""
sensitivity.py — Sensitivity analysis for the g0 fit.

Varies g0_init and data subsets to assess robustness of the fitted g0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scm_models import fit_g0, G0_DEFAULT


def g0_sensitivity_g0_init(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    g0_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Fit g0 starting from a range of initial guesses.

    Parameters
    ----------
    g_bar, g_obs : array-like
        Acceleration data.
    g0_grid : array-like, optional
        Initial g0 values to try (m/s²).  Defaults to a log-spaced grid
        from 1e-11 to 1e-9.

    Returns
    -------
    pd.DataFrame with columns ``g0_init``, ``g0_fit``, ``g0_err``, ``rms``.
    """
    if g0_grid is None:
        g0_grid = np.logspace(-11, -9, 20)

    records = []
    for g0_init in g0_grid:
        try:
            res = fit_g0(g_bar, g_obs, g0_init=float(g0_init))
            records.append({"g0_init": g0_init, **res})
        except Exception:
            records.append({
                "g0_init": g0_init, "g0": np.nan, "g0_err": np.nan,
                "rms": np.nan, "n": 0,
            })
    return pd.DataFrame(records)


def g0_sensitivity_bootstrap(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Bootstrap uncertainty on g0.

    Parameters
    ----------
    g_bar, g_obs : array-like
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        ``g0_median``, ``g0_std``, ``g0_p16``, ``g0_p84``,
        ``samples`` (array of length n_boot).
    """
    rng = np.random.default_rng(seed)
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)
    n = len(g_bar)

    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            res = fit_g0(g_bar[idx], g_obs[idx])
            samples.append(res["g0"])
        except Exception:
            pass

    samples = np.array(samples)
    return {
        "g0_median": float(np.median(samples)),
        "g0_std": float(np.std(samples)),
        "g0_p16": float(np.percentile(samples, 16)),
        "g0_p84": float(np.percentile(samples, 84)),
        "samples": samples,
    }


def run_sensitivity(
    csv_path: str,
    n_boot: int = 200,
    out_dir: str | None = None,
) -> dict:
    """Full sensitivity analysis over the SPARC RAR CSV.

    Parameters
    ----------
    csv_path : str
    n_boot : int
    out_dir : str, optional

    Returns
    -------
    dict with keys ``init_sweep`` and ``bootstrap``.
    """
    from src.scm_analysis import load_sparc_csv
    from pathlib import Path

    df = load_sparc_csv(csv_path)
    gb = df["g_bar"].values
    go = df["g_obs"].values

    init_sweep = g0_sensitivity_g0_init(gb, go)
    boot = g0_sensitivity_bootstrap(gb, go, n_boot=n_boot)

    print("--- Sensitivity: g0_init sweep ---")
    print(f"  g0 range across inits: "
          f"{init_sweep['g0'].min():.4e} – {init_sweep['g0'].max():.4e} m/s²")

    print("\n--- Sensitivity: bootstrap (n={}) ---".format(n_boot))
    print(f"  g0 median : {boot['g0_median']:.4e} m/s²")
    print(f"  g0 ±1σ    : {boot['g0_std']:.4e} m/s²")
    print(f"  g0 [16,84]%: [{boot['g0_p16']:.4e}, {boot['g0_p84']:.4e}] m/s²")

    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        init_sweep.to_csv(p / "sensitivity_init_sweep.csv", index=False)
        pd.DataFrame({"g0_sample": boot["samples"]}).to_csv(
            p / "sensitivity_bootstrap.csv", index=False
        )

    return {"init_sweep": init_sweep, "bootstrap": boot}
