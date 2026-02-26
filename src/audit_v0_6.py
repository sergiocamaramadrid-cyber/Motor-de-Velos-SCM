"""
audit_v0_6.py — Structural audit for the Motor de Velos SCM v0.6.

Checks the multicollinearity and numerical stability of the design matrix
used in the Motor de Velos rotation-curve regression.

Design matrix predictors
------------------------
  log_g_bar  — log10 of baryonic centripetal acceleration (m/s²)
  log_r      — log10 of galactocentric radius (kpc)
  v_bar      — standardised baryonic rotation velocity (unit-variance, km/s)
  v_velos    — standardised velos pressure velocity (unit-variance, km/s)
  hinge      — piecewise transition above the median g_bar:
               max(0, log10(g_bar) − median(log10(g_bar)))

Pass criteria
-------------
  hinge VIF           :  2.0 ≤ VIF ≤ 5.0   (structural)
  condition_number_κ  :  κ < 30             (numerical stability)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .scm_models import KPC_TO_M

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Convert (km/s)²/kpc → m/s²
_CONV = 1e6 / KPC_TO_M

_A0_DEFAULT = 1.2e-10   # m/s²  (characteristic velos acceleration)

# ---------------------------------------------------------------------------
# Audit thresholds
# ---------------------------------------------------------------------------

VIF_HINGE_MIN = 2.0
VIF_HINGE_MAX = 5.0
VIF_HINGE_WARN = 10.0

KAPPA_PASS = 30.0
KAPPA_WARN = 100.0


# ---------------------------------------------------------------------------
# Synthetic-data generator
# ---------------------------------------------------------------------------

def _generate_synthetic_data(n_galaxies: int = 175, n_pts: int = 20,
                              seed: int = 42) -> tuple:
    """Generate a SPARC-like synthetic dataset.

    Parameters
    ----------
    n_galaxies : int
        Number of synthetic galaxies (default 175, matching SPARC).
    n_pts : int
        Radial points per galaxy.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (r_all, v_bar_all, v_velos_all)
        All arrays are 1-D with length n_galaxies × n_pts.
    """
    a0_kpc = _A0_DEFAULT / _CONV  # (km/s)² / kpc

    r_lists, vb_lists, vv_lists = [], [], []
    for i in range(n_galaxies):
        v_flat = 80.0 + (320.0 - 80.0) * i / max(n_galaxies - 1, 1)
        r = np.linspace(0.5, 15.0, n_pts)
        v_bar = v_flat * np.tanh(r / 2.0)
        v_velos = np.sqrt(a0_kpc * r)
        r_lists.append(r)
        vb_lists.append(v_bar)
        vv_lists.append(v_velos)

    return (
        np.concatenate(r_lists),
        np.concatenate(vb_lists),
        np.concatenate(vv_lists),
    )


# ---------------------------------------------------------------------------
# Design-matrix builder
# ---------------------------------------------------------------------------

def build_design_matrix(r: np.ndarray, v_bar: np.ndarray,
                         v_velos: np.ndarray,
                         a0: float = _A0_DEFAULT) -> tuple:
    """Build the regression design matrix from per-radial-point kinematics.

    Parameters
    ----------
    r : ndarray
        Galactocentric radii (kpc).
    v_bar : ndarray
        Baryonic rotation velocity (km/s).
    v_velos : ndarray
        Velos pressure velocity (km/s).
    a0 : float
        Characteristic velos acceleration (m/s²).

    Returns
    -------
    X : ndarray, shape (N, 5)
        Design matrix with columns [log_g_bar, log_r, v_bar, v_velos, hinge].
    feature_names : list[str]
        Column labels.
    """
    g_bar = v_bar ** 2 / np.maximum(r, 1e-10) * _CONV  # m/s²
    valid = g_bar > 0
    r_v = r[valid]
    g_bar_v = g_bar[valid]
    v_bar_v = v_bar[valid]
    v_velos_v = v_velos[valid]

    log_g_bar = np.log10(g_bar_v)
    log_r = np.log10(np.maximum(r_v, 1e-10))

    # Normalise velocities to unit variance so they are commensurate with
    # the log-space predictors.
    std_vb = v_bar_v.std() if v_bar_v.std() > 0 else 1.0
    std_vv = v_velos_v.std() if v_velos_v.std() > 0 else 1.0
    v_bar_std = v_bar_v / std_vb
    v_velos_std = v_velos_v / std_vv

    # Hinge: piecewise linear transition above the median g_bar
    log_g_mid = float(np.median(log_g_bar))
    hinge = np.maximum(0.0, log_g_bar - log_g_mid)

    X = np.column_stack([log_g_bar, log_r, v_bar_std, v_velos_std, hinge])
    feature_names = ["log_g_bar", "log_r", "v_bar", "v_velos", "hinge"]
    return X, feature_names


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_vif(X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each column of *X*.

    For predictor j, VIF_j = 1 / (1 − R²_j) where R²_j is the coefficient
    of determination from regressing column j on all other columns (plus an
    intercept).

    Parameters
    ----------
    X : ndarray, shape (N, p)
        Design matrix (no intercept column).
    feature_names : list[str]
        Column labels.

    Returns
    -------
    pd.DataFrame
        Columns: ``['feature', 'vif']``.
    """
    n, p = X.shape
    records = []
    for j in range(p):
        y = X[:, j]
        other_cols = [i for i in range(p) if i != j]
        X_oth = np.column_stack([np.ones(n), X[:, other_cols]])
        coefs, _, _, _ = np.linalg.lstsq(X_oth, y, rcond=None)
        y_hat = X_oth @ coefs
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        vif = 1.0 / max(1.0 - r2, 1e-10)
        records.append({"feature": feature_names[j], "vif": vif})
    return pd.DataFrame(records)


def compute_condition_number(X: np.ndarray) -> float:
    """Compute the condition number κ of the column-standardised design matrix.

    Each column is mean-centred and then divided by its standard deviation
    before computing the singular value ratio.  Mean-centring removes the
    constant offset that would otherwise inflate κ.

    Parameters
    ----------
    X : ndarray, shape (N, p)
        Design matrix.

    Returns
    -------
    float
        κ = σ_max / σ_min (ratio of largest to smallest singular value of the
        mean-centred, unit-variance matrix).
    """
    col_std = X.std(axis=0)
    col_std = np.where(col_std > 0, col_std, 1.0)
    X_std = (X - X.mean(axis=0)) / col_std
    sv = np.linalg.svd(X_std, compute_uv=False)
    return float(sv[0] / sv[-1])


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _hinge_status(vif: float) -> str:
    if VIF_HINGE_MIN <= vif <= VIF_HINGE_MAX:
        return "PASS"
    if vif <= VIF_HINGE_WARN:
        return "WARNING"
    return "FAIL"


def _kappa_status(kappa: float) -> str:
    if kappa < KAPPA_PASS:
        return "stable"
    if kappa < KAPPA_WARN:
        return "warning"
    return "unstable"


# ---------------------------------------------------------------------------
# Main audit entry point
# ---------------------------------------------------------------------------

def run_audit(out_dir=None, a0: float = _A0_DEFAULT,
              n_galaxies: int = 175, n_pts: int = 20,
              seed: int = 42) -> dict:
    """Run the Motor de Velos SCM structural audit v0.6.

    Generates a SPARC-sized synthetic kinematic dataset, builds the regression
    design matrix, and evaluates multicollinearity (VIF) and numerical
    stability (condition number κ).

    Parameters
    ----------
    out_dir : str, Path, or None
        Directory for audit output files.  Defaults to
        ``results/audit_v0.6/audit``.
    a0 : float
        Characteristic velos acceleration (m/s²).
    n_galaxies : int
        Number of synthetic galaxies.
    n_pts : int
        Radial points per galaxy.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``vif_table`` (DataFrame), ``condition_number_kappa`` (float),
        ``overall_status`` (str).
    """
    if out_dir is None:
        out_dir = Path("results/audit_v0.6/audit")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate kinematic data
    r_all, v_bar_all, v_velos_all = _generate_synthetic_data(
        n_galaxies=n_galaxies, n_pts=n_pts, seed=seed
    )

    # 2. Build design matrix
    X, feature_names = build_design_matrix(r_all, v_bar_all, v_velos_all, a0=a0)

    # 3. Compute VIF table
    vif_df = compute_vif(X, feature_names)
    vif_df.to_csv(out_dir / "vif_table.csv", index=False)

    # 4. Compute condition number
    kappa = compute_condition_number(X)
    kap_stat = _kappa_status(kappa)
    stability_df = pd.DataFrame([{
        "metric": "condition_number_kappa",
        "value": round(kappa, 1),
        "status": kap_stat,
    }])
    stability_df.to_csv(out_dir / "stability_metrics.csv", index=False)

    # 5. Quality status
    hinge_vif = float(
        vif_df.loc[vif_df["feature"] == "hinge", "vif"].iloc[0]
    )
    hinge_pass = VIF_HINGE_MIN <= hinge_vif <= VIF_HINGE_MAX
    kappa_pass = kappa < KAPPA_PASS
    overall = "PASS" if (hinge_pass and kappa_pass) else "FAIL"

    status_lines = [
        f"hinge VIF: {hinge_vif:.2f}",
        f"  PASS criteria: {VIF_HINGE_MIN} <= VIF <= {VIF_HINGE_MAX}",
        f"  STATUS: {_hinge_status(hinge_vif)}",
        "",
        f"condition_number_kappa: {kappa:.1f}",
        f"  PASS criteria: condition_number_kappa < {KAPPA_PASS}",
        f"  STATUS: {'PASS' if kappa_pass else 'FAIL'}",
        "",
        f"OVERALL STATUS: {overall}",
    ]
    (out_dir / "quality_status.txt").write_text(
        "\n".join(status_lines) + "\n", encoding="utf-8"
    )

    return {
        "vif_table": vif_df,
        "condition_number_kappa": kappa,
        "overall_status": overall,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Motor de Velos SCM structural audit v0.6"
    )
    parser.add_argument(
        "--out", default="results/audit_v0.6/audit",
        help="Output directory (default: results/audit_v0.6/audit)"
    )
    parser.add_argument(
        "--a0", type=float, default=_A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {_A0_DEFAULT:.2e})"
    )
    parser.add_argument(
        "--n-galaxies", type=int, default=175,
        help="Number of synthetic galaxies (default: 175)"
    )
    args = parser.parse_args(argv)
    result = run_audit(out_dir=args.out, a0=args.a0, n_galaxies=args.n_galaxies)
    vif_df = result["vif_table"]
    hinge_vif = float(vif_df.loc[vif_df["feature"] == "hinge", "vif"].iloc[0])
    print(f"hinge VIF     : {hinge_vif:.3f}")
    print(f"kappa         : {result['condition_number_kappa']:.1f}")
    print(f"OVERALL STATUS: {result['overall_status']}")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
