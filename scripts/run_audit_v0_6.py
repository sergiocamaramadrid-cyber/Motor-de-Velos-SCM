"""
scripts/run_audit_v0_6.py — Model-diagnostic audit for Motor de Velos SCM v0.6.

Computes:
  - VIF (Variance Inflation Factor) for each design-matrix column.
  - Condition number kappa of the normalised design matrix.
  - PASS/WARNING/FAIL verdicts based on pre-defined thresholds.

Design matrix
-------------
Rows: one per (galaxy, radial-point) pair from a 175-galaxy synthetic dataset.
Columns:
  - intercept  : constant 1
  - baryonic   : V_bar² / V_flat²  (dimensionless baryonic contribution)
  - hinge      : V_velos² / V_velos²_max  (dimensionless veil-pressure term)

The rising-disk profile used here (v_disk ∝ r/sqrt(1+(r/r_d)²)) is the
standard Freeman-disk approximation with scale radius r_d = 3 kpc.

Writes three files into <out_dir>:
  vif_table.csv          — per-predictor VIF and PASS/WARNING/FAIL status
  stability_metrics.csv  — condition number and stability status
  quality_status.txt     — single-line OVERALL STATUS verdict

Usage
-----
    python scripts/run_audit_v0_6.py --out results/audit_v0.6/audit
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

# ── make ``src`` importable when running as a plain script ──────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.scm_models import KPC_TO_M, v_baryonic  # noqa: E402

# ── Thresholds ───────────────────────────────────────────────────────────────
_VIF_PASS_LO = 2.0
_VIF_PASS_HI = 5.0
_VIF_WARN_HI = 10.0

_KAPPA_PASS = 30.0
_KAPPA_WARN = 100.0

# ── Design-matrix parameters ─────────────────────────────────────────────────
_N_GALAXIES = 175
_N_PTS = 20
_VFLAT_MIN = 80.0   # km/s
_VFLAT_MAX = 320.0  # km/s
_RDISK_KPC = 3.0    # Freeman disk scale radius (kpc)
_RMAX_KPC = 15.0    # maximum observed radius (kpc)


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------

def _build_design_matrix(a0: float = 1.2e-10) -> np.ndarray:
    """Build the normalised design matrix from a 175-galaxy synthetic dataset.

    Parameters
    ----------
    a0 : float
        Characteristic velos acceleration (m/s²).

    Returns
    -------
    np.ndarray, shape (N_GALAXIES * N_PTS, 3)
        Columns: [intercept, baryonic, hinge] — each scaled to unit variance.
    """
    a0_kpc = a0 * 1e-3 / KPC_TO_M  # km²/s² per kpc

    v_flats = np.linspace(_VFLAT_MIN, _VFLAT_MAX, _N_GALAXIES)
    rows = []

    for vf in v_flats:
        r = np.linspace(0.5, _RMAX_KPC, _N_PTS)

        # Freeman-disk profile: rises steeply inside r_d, flattens beyond
        v_disk = 0.75 * vf * (r / _RDISK_KPC) / np.sqrt(1.0 + (r / _RDISK_KPC) ** 2)
        v_gas = 0.3 * vf * np.ones(_N_PTS)
        v_bul = np.zeros(_N_PTS)

        vb = v_baryonic(r, v_gas, v_disk, v_bul, upsilon_disk=1.0, upsilon_bul=0.7)

        # Normalise each term to make predictors dimensionless and
        # comparable across galaxies of very different masses.
        vb2_norm = vb ** 2 / vf ** 2                        # baryonic column
        vv2_norm = a0_kpc * r / (a0_kpc * _RMAX_KPC)       # hinge column ∈ [0,1]

        for i in range(_N_PTS):
            rows.append([1.0, vb2_norm[i], vv2_norm[i]])

    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# VIF computation
# ---------------------------------------------------------------------------

def _compute_vif(X_norm: np.ndarray) -> list:
    """Return VIF for every column of *X_norm*.

    For column j, VIF_j = 1/(1 − R²_j) where R²_j is the coefficient of
    determination from the OLS regression of column j on all other columns.

    Parameters
    ----------
    X_norm : np.ndarray, shape (N, p)
        Design matrix with columns already normalised to unit variance.

    Returns
    -------
    list of float
        VIF for each column.
    """
    _, p = X_norm.shape
    vifs = []
    for j in range(p):
        y = X_norm[:, j]
        X_others = np.delete(X_norm, j, axis=1)
        coef, _, _, _ = lstsq(X_others, y, rcond=None)
        y_pred = X_others @ coef
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0.0:
            vifs.append(1.0)
        else:
            r2 = 1.0 - ss_res / ss_tot
            vifs.append(1.0 / (1.0 - r2) if r2 < 1.0 else float("inf"))
    return vifs


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _vif_status(vif: float) -> str:
    if vif <= _VIF_PASS_HI:
        return "PASS"
    if vif <= _VIF_WARN_HI:
        return "WARNING"
    return "FAIL"


def _kappa_status(kappa: float) -> str:
    if kappa < _KAPPA_PASS:
        return "stable"
    if kappa < _KAPPA_WARN:
        return "WARNING"
    return "FAIL"


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit(out_dir: str | Path = "results/audit_v0.6/audit",
              a0: float = 1.2e-10) -> tuple:
    """Execute the v0.6 model audit and write three diagnostic files.

    Parameters
    ----------
    out_dir : str or Path
        Directory where the three audit files are written (created if needed).
    a0 : float
        Characteristic velos acceleration (m/s²).

    Returns
    -------
    tuple
        (vif_df, stability_df, overall_status)
        - vif_df : pd.DataFrame  — per-predictor VIF table
        - stability_df : pd.DataFrame  — condition-number table
        - overall_status : str  — 'PASS', 'WARNING', or 'FAIL'
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build and normalise design matrix ────────────────────────────────────
    X = _build_design_matrix(a0=a0)
    col_std = X.std(axis=0)
    col_std[col_std == 0.0] = 1.0
    X_norm = X / col_std

    predictor_names = ["intercept", "baryonic", "hinge"]

    # ── VIF ──────────────────────────────────────────────────────────────────
    vifs = _compute_vif(X_norm)
    vif_rows = [
        {"predictor": name, "vif": round(vif, 4), "status": _vif_status(vif)}
        for name, vif in zip(predictor_names, vifs)
    ]
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(out_dir / "vif_table.csv", index=False)

    # ── Condition number ─────────────────────────────────────────────────────
    kappa = float(np.linalg.cond(X_norm))
    kappa_status = _kappa_status(kappa)
    stability_df = pd.DataFrame([{
        "metric": "condition_number_kappa",
        "value": round(kappa, 2),
        "status": kappa_status,
    }])
    stability_df.to_csv(out_dir / "stability_metrics.csv", index=False)

    # ── Overall quality status ────────────────────────────────────────────────
    hinge_row = vif_df[vif_df["predictor"] == "hinge"].iloc[0]
    hinge_vif = float(hinge_row["vif"])
    hinge_pass = _VIF_PASS_LO <= hinge_vif <= _VIF_PASS_HI

    if hinge_pass and kappa < _KAPPA_PASS:
        overall = "PASS"
    elif hinge_vif > _VIF_WARN_HI or kappa >= _KAPPA_WARN:
        overall = "FAIL"
    else:
        overall = "WARNING"

    status_lines = [
        f"OVERALL STATUS: {overall}",
        "",
        f"hinge VIF:          {hinge_vif:.4f}  ({hinge_row['status']})",
        f"condition_number:   {kappa:.2f}  ({kappa_status})",
    ]
    (out_dir / "quality_status.txt").write_text(
        "\n".join(status_lines) + "\n", encoding="utf-8"
    )

    print(f"Audit complete → {out_dir}")
    print(f"  hinge VIF = {hinge_vif:.4f}")
    print(f"  kappa     = {kappa:.2f}")
    print(f"  OVERALL   = {overall}")

    return vif_df, stability_df, overall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Motor de Velos v0.6 model-diagnostic audit"
    )
    p.add_argument(
        "--out", default="results/audit_v0.6/audit",
        help="Output directory (default: results/audit_v0.6/audit)",
    )
    p.add_argument(
        "--a0", type=float, default=1.2e-10,
        help="Characteristic velos acceleration in m/s² (default: 1.2e-10)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_audit(out_dir=args.out, a0=args.a0)


if __name__ == "__main__":
    main()
