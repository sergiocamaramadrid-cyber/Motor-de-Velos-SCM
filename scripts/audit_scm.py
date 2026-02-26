"""
scripts/audit_scm.py — GroupKFold cross-validation audit of Motor de Velos SCM.

Five audit checks (A–E):

A) GroupKFold (no data leakage)
   Groups are galaxies.  Fitting is done ONLY on each test galaxy's own radial
   points; the global parameter a0 is fixed and never re-estimated from data.
   Each galaxy appears in exactly one test fold.

B) Per-galaxy metrics
   Outputs:
     groupkfold_per_point.csv  — one row per radial point
     groupkfold_per_galaxy.csv — one row per galaxy with
         rmse_btfr, rmse_no_hinge, rmse_full, delta_rmse_full_vs_btfr

C) Permutation test
   Within each galaxy: shuffle (v_gas, v_disk, v_bul) rows together,
   keeping r and v_obs fixed.  This breaks the g_bar–g_obs structure.
   Empirical p-value:
       p = (1 + Σ[perm_rmse ≤ rmse_real]) / (n_perm + 1)
   Output: permutation_distribution.csv

D) Coefficient stability
   upsilon_disk fitted per galaxy per fold is stored in coeffs_by_fold.csv.
   Summary statistics (mean, std, range) are written to the audit report.

E) Output directory
   All artefacts are written to <out_dir>/ (default: results/audit/).

Three models compared
---------------------
btfr       — Flat-curve baseline: fit a single V_flat per galaxy.
             Represents the simplest possible prediction (one free param).
no_hinge   — Baryonic-only (Newtonian): V_pred = V_bar(upsilon_disk).
             No MOND / velos correction.
full       — Motor de Velos SCM: V_pred = sqrt(V_bar² + V_velos²).
             V_velos² = a0_kpc * r  (linear rise).

Usage
-----
With rotmod files::

    python scripts/audit_scm.py --data-dir data/SPARC --out results/audit

Options::

    python scripts/audit_scm.py \\
        --data-dir data/SPARC \\
        --out      results/audit \\
        --n-splits 5 \\
        --n-perm   1000 \\
        --a0       1.2e-10 \\
        --seed     42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

KPC_TO_M = 3.085677581e16          # metres per kiloparsec (IAU 2012)
_CONV = 1e6 / KPC_TO_M             # (km/s)²/kpc → m/s²
A0_DEFAULT = 1.2e-10               # m/s²

# ---------------------------------------------------------------------------
# Velocity model helpers
# ---------------------------------------------------------------------------

def _v_baryonic(v_gas: np.ndarray, v_disk: np.ndarray, v_bul: np.ndarray,
                upsilon_disk: float) -> np.ndarray:
    """Signed baryonic rotation velocity (km/s)."""
    v2 = (v_gas * np.abs(v_gas)
          + upsilon_disk * v_disk * np.abs(v_disk)
          + 0.7 * v_bul * np.abs(v_bul))
    return np.sign(v2) * np.sqrt(np.abs(v2))


def _v_pred_no_hinge(r: np.ndarray, v_gas: np.ndarray, v_disk: np.ndarray,
                     v_bul: np.ndarray, upsilon_disk: float) -> np.ndarray:
    """Baryonic-only (no velos / MOND term)."""
    return _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk)


def _v_pred_full(r: np.ndarray, v_gas: np.ndarray, v_disk: np.ndarray,
                 v_bul: np.ndarray, upsilon_disk: float,
                 a0: float = A0_DEFAULT) -> np.ndarray:
    """Full Motor de Velos model: V_total² = V_bar² + V_velos²."""
    vb = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk)
    a0_kpc = a0 / _CONV  # (km/s)² / kpc
    vv2 = a0_kpc * np.maximum(r, 0.0)
    v2 = vb * np.abs(vb) + vv2
    return np.sign(v2) * np.sqrt(np.abs(v2))


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _rmse(v_obs: np.ndarray, v_pred: np.ndarray) -> float:
    """Root-mean-square error in km/s."""
    return float(np.sqrt(np.mean((v_obs - v_pred) ** 2)))


def _fit_v_flat(v_obs: np.ndarray) -> float:
    """Best-fit flat velocity for the BTFR baseline (minimises RMSE)."""
    # For a constant predictor, the optimal value is the mean.
    return float(np.mean(v_obs))


def _fit_upsilon(r: np.ndarray, v_obs: np.ndarray,
                 v_gas: np.ndarray, v_disk: np.ndarray, v_bul: np.ndarray,
                 pred_fn, bounds=(0.1, 5.0)) -> float:
    """Fit upsilon_disk ∈ bounds by minimising RMSE."""
    def objective(ud: float) -> float:
        return _rmse(v_obs, pred_fn(r, v_gas, v_disk, v_bul, ud))
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# GroupKFold splitter
# ---------------------------------------------------------------------------

def group_kfold_split(
    groups: list[str], n_splits: int
) -> Iterator[tuple[int, list[int], list[int]]]:
    """Yield ``(fold_id, train_idx, test_idx)`` for GroupKFold.

    Each unique group appears in exactly one test fold.  Groups are assigned
    to folds round-robin so that the fold sizes are as equal as possible.

    Parameters
    ----------
    groups : list of str
        Group label for every sample (length = total radial points).
    n_splits : int
        Number of folds.

    Yields
    ------
    fold_id : int
        Zero-based fold index.
    train_idx : list of int
        Indices of training samples.
    test_idx : list of int
        Indices of test samples.
    """
    unique_groups = list(dict.fromkeys(groups))  # preserve insertion order
    fold_of: dict[str, int] = {g: i % n_splits for i, g in enumerate(unique_groups)}

    for fold in range(n_splits):
        test_set = {g for g, f in fold_of.items() if f == fold}
        train_idx = [i for i, g in enumerate(groups) if g not in test_set]
        test_idx = [i for i, g in enumerate(groups) if g in test_set]
        yield fold, train_idx, test_idx


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rotmod(data_dir: Path, name: str) -> pd.DataFrame | None:
    """Load one galaxy's rotation curve; returns None if not found."""
    for candidate in (
        data_dir / f"{name}_rotmod.dat",
        data_dir / "raw" / f"{name}_rotmod.dat",
    ):
        if candidate.exists():
            df = pd.read_csv(
                candidate, sep=r"\s+", comment="#",
                names=["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul",
                       "SBdisk", "SBbul"],
            )
            return df[["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]]
    return None


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """Load all rotation curves found under *data_dir* into one DataFrame.

    Tries to read a SPARC galaxy table for the ordered galaxy list; falls back
    to scanning for ``*_rotmod.dat`` files.

    Returns
    -------
    pd.DataFrame
        Columns: galaxy, r, v_obs, v_obs_err, v_gas, v_disk, v_bul.
    """
    data_dir = Path(data_dir)
    # Try to load ordered galaxy list from SPARC table
    table_candidates = [
        data_dir / "SPARC_Lelli2016c.csv",
        data_dir / "SPARC_Lelli2016c.mrt",
        data_dir / "raw" / "SPARC_Lelli2016c.csv",
        data_dir / "processed" / "SPARC_Lelli2016c.csv",
    ]
    galaxy_names: list[str] = []
    for tc in table_candidates:
        if tc.exists():
            sep = "," if tc.suffix == ".csv" else r"\s+"
            tbl = pd.read_csv(tc, sep=sep, comment="#")
            galaxy_names = tbl["Galaxy"].tolist()
            break

    if not galaxy_names:
        # Fall back: scan directory
        rotmod_files = sorted(data_dir.glob("*_rotmod.dat"))
        galaxy_names = [f.name.replace("_rotmod.dat", "") for f in rotmod_files]

    if not galaxy_names:
        raise FileNotFoundError(
            f"No rotation-curve files found in {data_dir}. "
            "Expected *_rotmod.dat files."
        )

    frames: list[pd.DataFrame] = []
    for name in galaxy_names:
        rc = _load_rotmod(data_dir, name)
        if rc is None:
            continue
        rc = rc.copy()
        rc.insert(0, "galaxy", name)
        frames.append(rc)

    if not frames:
        raise FileNotFoundError(
            f"No rotation curves could be loaded from {data_dir}."
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-galaxy evaluation
# ---------------------------------------------------------------------------

def _evaluate_galaxy(
    gdf: pd.DataFrame, a0: float
) -> dict:
    """Fit all three models for one galaxy and return metrics.

    Parameters
    ----------
    gdf : pd.DataFrame
        Rotation-curve rows for a single galaxy.
    a0 : float
        Fixed characteristic acceleration (m/s²).

    Returns
    -------
    dict with keys:
        v_flat, ud_no_hinge, ud_full,
        rmse_btfr, rmse_no_hinge, rmse_full, delta_rmse_full_vs_btfr,
        v_pred_btfr, v_pred_no_hinge, v_pred_full   (arrays)
    """
    r = gdf["r"].values
    v_obs = gdf["v_obs"].values
    v_gas = gdf["v_gas"].values
    v_disk = gdf["v_disk"].values
    v_bul = gdf["v_bul"].values

    # — BTFR baseline: fit flat rotation curve (one free param: V_flat) —
    v_flat = _fit_v_flat(v_obs)
    v_pred_btfr = np.full_like(v_obs, v_flat)

    # — No-hinge: baryonic only, fit upsilon_disk —
    pred_no_hinge = lambda r_, vg_, vd_, vb_, ud_: _v_pred_no_hinge(
        r_, vg_, vd_, vb_, ud_)
    ud_no_hinge = _fit_upsilon(r, v_obs, v_gas, v_disk, v_bul, pred_no_hinge)
    v_pred_no_hinge = _v_pred_no_hinge(r, v_gas, v_disk, v_bul, ud_no_hinge)

    # — Full model: V_bar² + V_velos², fit upsilon_disk —
    pred_full = lambda r_, vg_, vd_, vb_, ud_: _v_pred_full(
        r_, vg_, vd_, vb_, ud_, a0=a0)
    ud_full = _fit_upsilon(r, v_obs, v_gas, v_disk, v_bul, pred_full)
    v_pred_full_arr = _v_pred_full(r, v_gas, v_disk, v_bul, ud_full, a0=a0)

    rmse_btfr = _rmse(v_obs, v_pred_btfr)
    rmse_no_hinge = _rmse(v_obs, v_pred_no_hinge)
    rmse_full = _rmse(v_obs, v_pred_full_arr)

    return {
        "v_flat": v_flat,
        "ud_no_hinge": ud_no_hinge,
        "ud_full": ud_full,
        "rmse_btfr": rmse_btfr,
        "rmse_no_hinge": rmse_no_hinge,
        "rmse_full": rmse_full,
        "delta_rmse_full_vs_btfr": rmse_full - rmse_btfr,
        "v_pred_btfr": v_pred_btfr,
        "v_pred_no_hinge": v_pred_no_hinge,
        "v_pred_full": v_pred_full_arr,
    }


# ---------------------------------------------------------------------------
# A+B  GroupKFold cross-validation
# ---------------------------------------------------------------------------

def run_groupkfold_cv(
    df: pd.DataFrame, n_splits: int = 5, a0: float = A0_DEFAULT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run GroupKFold CV and return per-point and per-galaxy DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (all galaxies); columns galaxy, r, v_obs, v_obs_err,
        v_gas, v_disk, v_bul.
    n_splits : int
        Number of folds (default 5).
    a0 : float
        Fixed characteristic acceleration (m/s²).

    Returns
    -------
    per_point_df : pd.DataFrame
        One row per radial point.
    per_galaxy_df : pd.DataFrame
        One row per galaxy with rmse_btfr, rmse_no_hinge, rmse_full,
        delta_rmse_full_vs_btfr.

    Notes
    -----
    Fit is performed exclusively on **test-galaxy** data — each galaxy's
    upsilon_disk is estimated from that galaxy's own radial points only.
    Global parameter a0 is *fixed* and never re-estimated from the dataset;
    this is documented explicitly as "fixed coefficients" (not fitted from
    training data).
    """
    groups = df["galaxy"].tolist()

    point_rows: list[dict] = []
    galaxy_rows: list[dict] = []

    for fold_id, _train_idx, test_idx in group_kfold_split(groups, n_splits):
        test_df = df.iloc[test_idx]
        for galaxy, gdf in test_df.groupby("galaxy", sort=False):
            metrics = _evaluate_galaxy(gdf, a0=a0)

            r_arr = gdf["r"].values
            v_obs_arr = gdf["v_obs"].values

            # Per-point rows
            for i in range(len(r_arr)):
                point_rows.append({
                    "galaxy": galaxy,
                    "fold": fold_id,
                    "r_kpc": float(r_arr[i]),
                    "v_obs": float(v_obs_arr[i]),
                    "v_pred_btfr": float(metrics["v_pred_btfr"][i]),
                    "v_pred_no_hinge": float(metrics["v_pred_no_hinge"][i]),
                    "v_pred_full": float(metrics["v_pred_full"][i]),
                    "residual_btfr": float(v_obs_arr[i] - metrics["v_pred_btfr"][i]),
                    "residual_no_hinge": float(v_obs_arr[i] - metrics["v_pred_no_hinge"][i]),
                    "residual_full": float(v_obs_arr[i] - metrics["v_pred_full"][i]),
                })

            # Per-galaxy rows
            galaxy_rows.append({
                "galaxy": galaxy,
                "fold": fold_id,
                "n_points": len(gdf),
                "v_flat_btfr": metrics["v_flat"],
                "upsilon_disk_no_hinge": metrics["ud_no_hinge"],
                "upsilon_disk_full": metrics["ud_full"],
                "rmse_btfr": metrics["rmse_btfr"],
                "rmse_no_hinge": metrics["rmse_no_hinge"],
                "rmse_full": metrics["rmse_full"],
                "delta_rmse_full_vs_btfr": metrics["delta_rmse_full_vs_btfr"],
            })

    per_point_df = pd.DataFrame(point_rows)
    per_galaxy_df = pd.DataFrame(galaxy_rows)
    return per_point_df, per_galaxy_df


# ---------------------------------------------------------------------------
# C  Permutation test
# ---------------------------------------------------------------------------

def _permute_baryonic_within_galaxies(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Shuffle (v_gas, v_disk, v_bul) rows within each galaxy.

    This permutes g_bar = V_bar²/r within each galaxy while leaving r and
    v_obs unchanged, breaking the g_bar–g_obs relationship.
    """
    permuted = df.copy()
    bary_cols = ["v_gas", "v_disk", "v_bul"]
    for galaxy in df["galaxy"].unique():
        mask = permuted["galaxy"] == galaxy
        idx = permuted.index[mask]
        perm = rng.permutation(len(idx))
        permuted.loc[idx, bary_cols] = (
            permuted.loc[idx, bary_cols].values[perm]
        )
    return permuted


def run_permutation_test(
    df: pd.DataFrame,
    n_splits: int = 5,
    a0: float = A0_DEFAULT,
    n_perm: int = 500,
    rng_seed: int = 42,
) -> tuple[float, np.ndarray, float, pd.DataFrame]:
    """Run permutation test on the full Motor de Velos model.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    n_splits : int
        GroupKFold folds.
    a0 : float
        Characteristic acceleration (m/s²).
    n_perm : int
        Number of permutations.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    rmse_real : float
        Mean per-galaxy RMSE_full from GroupKFold on real data.
    perm_rmse : ndarray, shape (n_perm,)
        Mean per-galaxy RMSE_full from each permuted run.
    p_value : float
        Empirical p-value:
            p = (1 + Σ[perm_rmse ≤ rmse_real]) / (n_perm + 1)
    perm_df : pd.DataFrame
        DataFrame with columns perm_id and rmse_full for CSV export.
    """
    rng = np.random.default_rng(rng_seed)

    # Real RMSE
    _, pg_real = run_groupkfold_cv(df, n_splits=n_splits, a0=a0)
    rmse_real = float(pg_real["rmse_full"].mean())

    # Permuted distribution
    perm_rmse_list: list[float] = []
    for _ in range(n_perm):
        df_perm = _permute_baryonic_within_galaxies(df, rng)
        _, pg_perm = run_groupkfold_cv(df_perm, n_splits=n_splits, a0=a0)
        perm_rmse_list.append(float(pg_perm["rmse_full"].mean()))

    perm_rmse = np.array(perm_rmse_list)
    p_value = float((1.0 + np.sum(perm_rmse <= rmse_real)) / (n_perm + 1))

    perm_df = pd.DataFrame({
        "perm_id": np.arange(n_perm),
        "rmse_full": perm_rmse,
    })
    return rmse_real, perm_rmse, p_value, perm_df


# ---------------------------------------------------------------------------
# D  Coefficient stability
# ---------------------------------------------------------------------------

def build_coeffs_by_fold(per_galaxy_df: pd.DataFrame) -> pd.DataFrame:
    """Return upsilon_disk fitted per (fold, galaxy) from GroupKFold results.

    Parameters
    ----------
    per_galaxy_df : pd.DataFrame
        Output of :func:`run_groupkfold_cv`.

    Returns
    -------
    pd.DataFrame
        Columns: fold, galaxy, upsilon_disk_no_hinge, upsilon_disk_full.
    """
    return per_galaxy_df[
        ["fold", "galaxy", "upsilon_disk_no_hinge", "upsilon_disk_full"]
    ].copy()


def coefficient_stability_stats(coeffs_df: pd.DataFrame) -> dict:
    """Compute mean, std, and range of upsilon_disk across all galaxies.

    Parameters
    ----------
    coeffs_df : pd.DataFrame
        Output of :func:`build_coeffs_by_fold`.

    Returns
    -------
    dict
        Keys: ``ud_no_hinge_mean``, ``ud_no_hinge_std``, ``ud_no_hinge_range``,
              ``ud_full_mean``, ``ud_full_std``, ``ud_full_range``.
    """
    stats: dict = {}
    for col, prefix in [
        ("upsilon_disk_no_hinge", "ud_no_hinge"),
        ("upsilon_disk_full", "ud_full"),
    ]:
        vals = coeffs_df[col].dropna().values
        stats[f"{prefix}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        stats[f"{prefix}_std"] = float(np.std(vals)) if len(vals) else float("nan")
        stats[f"{prefix}_range"] = (
            float(np.ptp(vals)) if len(vals) else float("nan")
        )
    return stats


# ---------------------------------------------------------------------------
# E  Main audit entry point
# ---------------------------------------------------------------------------

_SEP = "=" * 72


def run_audit(
    data_dir: str | Path,
    out_dir: str | Path = "results/audit",
    n_splits: int = 5,
    a0: float = A0_DEFAULT,
    n_perm: int = 500,
    rng_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run the full SCM GroupKFold audit and write all artefacts.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SPARC rotmod files.
    out_dir : str or Path
        Output directory (created if necessary).
    n_splits : int
        Number of GroupKFold folds.
    a0 : float
        Fixed characteristic acceleration (m/s²).
    n_perm : int
        Number of permutations for the permutation test.
    rng_seed : int
        Random seed.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Summary results including ``rmse_real``, ``p_value``,
        ``coeff_stats``, and paths to all written files.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(_SEP)
    _log("  Motor de Velos SCM — GroupKFold Audit")
    _log(_SEP)
    _log(f"  data_dir  : {data_dir}")
    _log(f"  out_dir   : {out_dir}")
    _log(f"  n_splits  : {n_splits}")
    _log(f"  a0        : {a0:.2e} m/s²  [FIXED — not fitted from data]")
    _log(f"  n_perm    : {n_perm}")
    _log(f"  rng_seed  : {rng_seed}")
    _log("")

    # ── Load data ──────────────────────────────────────────────────────────
    _log("  [1/4] Loading rotation curves …")
    df = load_all_data(data_dir)
    n_galaxies = df["galaxy"].nunique()
    n_points = len(df)
    _log(f"        {n_galaxies} galaxies, {n_points} radial points")

    # ── A + B: GroupKFold CV ────────────────────────────────────────────────
    _log(f"\n  [2/4] Running {n_splits}-fold GroupKFold CV …")
    _log("        NOTE: a0 is fixed; upsilon_disk is fitted per galaxy "
         "on its own radial points (legitimate per-galaxy free parameter).")
    per_point_df, per_galaxy_df = run_groupkfold_cv(df, n_splits=n_splits, a0=a0)

    pp_path = out_dir / "groupkfold_per_point.csv"
    pg_path = out_dir / "groupkfold_per_galaxy.csv"
    per_point_df.to_csv(pp_path, index=False)
    per_galaxy_df.to_csv(pg_path, index=False)
    _log(f"        Written: {pp_path.name}")
    _log(f"        Written: {pg_path.name}")

    rmse_btfr_mean = float(per_galaxy_df["rmse_btfr"].mean())
    rmse_no_hinge_mean = float(per_galaxy_df["rmse_no_hinge"].mean())
    rmse_full_mean = float(per_galaxy_df["rmse_full"].mean())
    delta_mean = float(per_galaxy_df["delta_rmse_full_vs_btfr"].mean())

    _log(f"\n        Mean RMSE (km/s) across {n_galaxies} galaxies:")
    _log(f"          btfr      : {rmse_btfr_mean:.4f}")
    _log(f"          no_hinge  : {rmse_no_hinge_mean:.4f}")
    _log(f"          full      : {rmse_full_mean:.4f}")
    _log(f"          Δ(full−btfr) mean: {delta_mean:+.4f}  "
         f"({'full better' if delta_mean < 0 else 'btfr better or equal'})")

    # ── C: Permutation test ────────────────────────────────────────────────
    _log(f"\n  [3/4] Permutation test ({n_perm} permutations) …")
    rmse_real, perm_rmse, p_value, perm_df = run_permutation_test(
        df, n_splits=n_splits, a0=a0, n_perm=n_perm, rng_seed=rng_seed
    )
    perm_path = out_dir / "permutation_distribution.csv"
    perm_df.to_csv(perm_path, index=False)
    _log(f"        Written: {perm_path.name}")
    _log(f"        rmse_real  = {rmse_real:.4f} km/s")
    _log(f"        perm mean  = {perm_rmse.mean():.4f} km/s")
    _log(f"        p-value    = {p_value:.4f}  "
         f"[p = (1 + Σ[perm≤real]) / (n_perm+1)]")

    # ── D: Coefficient stability ───────────────────────────────────────────
    _log("\n  [4/4] Coefficient stability …")
    coeffs_df = build_coeffs_by_fold(per_galaxy_df)
    coeffs_path = out_dir / "coeffs_by_fold.csv"
    coeffs_df.to_csv(coeffs_path, index=False)
    _log(f"        Written: {coeffs_path.name}")

    coeff_stats = coefficient_stability_stats(coeffs_df)
    _log(f"        upsilon_disk (no_hinge): "
         f"mean={coeff_stats['ud_no_hinge_mean']:.3f}, "
         f"std={coeff_stats['ud_no_hinge_std']:.3f}, "
         f"range={coeff_stats['ud_no_hinge_range']:.3f}")
    _log(f"        upsilon_disk (full):     "
         f"mean={coeff_stats['ud_full_mean']:.3f}, "
         f"std={coeff_stats['ud_full_std']:.3f}, "
         f"range={coeff_stats['ud_full_range']:.3f}")

    # ── Write audit report ─────────────────────────────────────────────────
    report_lines = [
        _SEP,
        "  Motor de Velos SCM — GroupKFold Audit Report",
        _SEP,
        f"  data_dir   : {data_dir}",
        f"  n_galaxies : {n_galaxies}",
        f"  n_points   : {n_points}",
        f"  n_splits   : {n_splits}",
        f"  a0         : {a0:.2e} m/s²  [FIXED]",
        f"  n_perm     : {n_perm}",
        "",
        "  [A] GroupKFold: a0 fixed; upsilon_disk fitted per galaxy on its own",
        "      radial points (legitimate nuisance parameter, not data leakage).",
        "",
        "  [B] Mean RMSE (km/s):",
        f"      btfr      : {rmse_btfr_mean:.4f}",
        f"      no_hinge  : {rmse_no_hinge_mean:.4f}",
        f"      full      : {rmse_full_mean:.4f}",
        f"      Δ full−btfr: {delta_mean:+.4f}",
        "",
        "  [C] Permutation test (permute g_bar within each galaxy):",
        f"      rmse_real  = {rmse_real:.4f}",
        f"      perm mean  = {perm_rmse.mean():.4f}",
        f"      p-value    = {p_value:.4f}",
        f"      verdict    : {'SIGNIFICANT (p < 0.05)' if p_value < 0.05 else 'NOT SIGNIFICANT (p >= 0.05)'}",
        "",
        "  [D] upsilon_disk stability:",
        f"      no_hinge: mean={coeff_stats['ud_no_hinge_mean']:.3f}, "
        f"std={coeff_stats['ud_no_hinge_std']:.3f}, "
        f"range={coeff_stats['ud_no_hinge_range']:.3f}",
        f"      full:     mean={coeff_stats['ud_full_mean']:.3f}, "
        f"std={coeff_stats['ud_full_std']:.3f}, "
        f"range={coeff_stats['ud_full_range']:.3f}",
        "",
        "  [E] Artefacts written to:",
        f"      {pp_path}",
        f"      {pg_path}",
        f"      {perm_path}",
        f"      {coeffs_path}",
        _SEP,
    ]
    report_path = out_dir / "audit_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _log(f"\n  Audit report: {report_path}")
    _log(_SEP)

    return {
        "rmse_real": rmse_real,
        "rmse_btfr_mean": rmse_btfr_mean,
        "rmse_no_hinge_mean": rmse_no_hinge_mean,
        "rmse_full_mean": rmse_full_mean,
        "delta_rmse_mean": delta_mean,
        "p_value": p_value,
        "coeff_stats": coeff_stats,
        "per_point_path": pp_path,
        "per_galaxy_path": pg_path,
        "permutation_path": perm_path,
        "coeffs_path": coeffs_path,
        "report_path": report_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GroupKFold cross-validation audit of Motor de Velos SCM."
    )
    parser.add_argument(
        "--data-dir", required=True, metavar="DIR",
        help="Directory containing SPARC rotmod files.",
    )
    parser.add_argument(
        "--out", default="results/audit", metavar="DIR",
        help="Output directory (default: results/audit).",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, metavar="K",
        help="Number of GroupKFold folds (default: 5).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--n-perm", type=int, default=500, metavar="N",
        help="Number of permutations for the permutation test (default: 500).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_audit(
        data_dir=args.data_dir,
        out_dir=args.out,
        n_splits=args.n_splits,
        a0=args.a0,
        n_perm=args.n_perm,
        rng_seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
