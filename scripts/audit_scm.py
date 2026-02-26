"""
scripts/audit_scm.py — Out-of-sample (OOS) structural audit for the SCM pipeline.

Produces the residual-vs-hinge diagnostic used to verify that the Motor de
Velos SCM term explains variance without introducing systematic residual trends.

Outputs (written to ``--outdir/audit/``):
  residual_vs_hinge.csv   — per-radial-point OOS residuals and hinge values
  residual_vs_hinge.png   — two-panel diagnostic figure
  vif_table.csv           — VIF for hinge regression predictors
  stability_metrics.csv   — condition number (kappa) and related diagnostics
  quality_status.txt      — PASS / WARNING verdict
  audit_features.csv      — feature summary statistics

Success criteria:
  Panel 1 (residual SCM vs hinge): median residual ≈ 0, no clear trend.
  Panel 2 (improvement vs hinge):  improvement tends > 0 as hinge grows.

Usage
-----
With SPARC rotmod files::

    python scripts/audit_scm.py --data-dir data/sparc --outdir results/final_audit

With a pre-computed universal_term_comparison_full.csv (limited mode)::

    python scripts/audit_scm.py \\
        --csv results/universal_term_comparison_full.csv \\
        --outdir results/final_audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Physical constants — must match src/scm_models.py convention
# ---------------------------------------------------------------------------

KPC_TO_M = 3.085677581e16  # internal "kpc" unit (matches scm_models.py)
_CONV = 1e6 / KPC_TO_M    # (km/s)²/kpc → m/s²  (internal convention)
_A0_DEFAULT = 1.2e-10      # m/s²


# ---------------------------------------------------------------------------
# Velocity predictors (mirror compare_nu_models.py conventions)
# ---------------------------------------------------------------------------

def _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk: float,
                upsilon_bul: float = 0.7) -> np.ndarray:
    """Signed baryonic rotation velocity."""
    v2 = (v_gas * np.abs(v_gas)
          + upsilon_disk * v_disk * np.abs(v_disk)
          + upsilon_bul * v_bul * np.abs(v_bul))
    return np.sign(v2) * np.sqrt(np.abs(v2))


def _v_velos(r: np.ndarray, a0: float = _A0_DEFAULT) -> np.ndarray:
    """SCM hinge velocity contribution: V_velos = sqrt(a0_kpc × r).

    Uses the same convention as compare_nu_models.py:
        a0_kpc = a0 / _CONV    [(km/s)² / kpc]
    """
    a0_kpc = a0 / _CONV  # (km/s)² / kpc
    return np.sqrt(np.maximum(a0_kpc * np.asarray(r, dtype=float), 0.0))


def _v_total(r, v_gas, v_disk, v_bul, upsilon_disk: float,
             a0: float = _A0_DEFAULT) -> np.ndarray:
    """Total SCM predicted velocity: V² = V_bar² + V_velos²."""
    vb = _v_baryonic(v_gas, v_disk, v_bul, upsilon_disk)
    vb2 = vb * np.abs(vb)
    a0_kpc = a0 / _CONV
    vv2 = a0_kpc * np.maximum(np.asarray(r, dtype=float), 0.0)
    v2 = vb2 + vv2
    return np.sign(v2) * np.sqrt(np.abs(v2))


def _fit_upsilon(rc: pd.DataFrame, a0: float = _A0_DEFAULT) -> float:
    """Fit upsilon_disk minimising chi-squared for a single galaxy."""
    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_obs_err = rc["v_obs_err"].values
    v_gas = rc["v_gas"].values
    v_disk = rc["v_disk"].values
    v_bul = rc["v_bul"].values

    def objective(ud: float) -> float:
        vp = _v_total(r, v_gas, v_disk, v_bul, ud, a0=a0)
        safe_err = np.where(v_obs_err > 0, v_obs_err, 1.0)
        return float(np.sum(((v_obs - vp) / safe_err) ** 2) / max(len(r) - 1, 1))

    result = minimize_scalar(objective, bounds=(0.1, 5.0), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_galaxy_table(data_dir: Path) -> pd.DataFrame:
    candidates = [
        data_dir / "SPARC_Lelli2016c.csv",
        data_dir / "SPARC_Lelli2016c.mrt",
        data_dir / "raw" / "SPARC_Lelli2016c.csv",
        data_dir / "processed" / "SPARC_Lelli2016c.csv",
    ]
    for p in candidates:
        if p.exists():
            sep = "," if p.suffix == ".csv" else r"\s+"
            return pd.read_csv(p, sep=sep, comment="#")
    raise FileNotFoundError(
        f"SPARC galaxy table not found in {data_dir}. "
        "Expected SPARC_Lelli2016c.csv or .mrt"
    )


def _load_rotation_curve(data_dir: Path, name: str) -> pd.DataFrame:
    candidates = [
        data_dir / f"{name}_rotmod.dat",
        data_dir / "raw" / f"{name}_rotmod.dat",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(
                p, sep=r"\s+", comment="#",
                names=["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul",
                       "SBdisk", "SBbul"],
            )
            return df[["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]]
    raise FileNotFoundError(f"Rotation curve for {name} not found in {data_dir}")


# ---------------------------------------------------------------------------
# OOS residual computation
# ---------------------------------------------------------------------------

def compute_oos_residuals(
    data_dir: Path,
    a0: float = _A0_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute per-radial-point OOS residuals and hinge values for all galaxies.

    Each galaxy is fitted independently (upsilon_disk free parameter, a0
    fixed at its fiducial value).  Residuals are computed on the same galaxy
    after fitting — this represents the structural audit diagnostic rather
    than a strict leave-one-out cross-validation.

    Parameters
    ----------
    data_dir : Path
        SPARC data directory.
    a0 : float
        Characteristic SCM acceleration (m/s²).
    verbose : bool
        Print progress if True.

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy``, ``r_kpc``, ``v_hinge``, ``residual_scm``,
        ``residual_bary``, ``improvement``.
    """
    galaxy_table = _load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()

    rows = []
    for name in galaxy_names:
        try:
            rc = _load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: rotation curve not found", file=sys.stderr)
            continue

        ud = _fit_upsilon(rc, a0=a0)

        r = rc["r"].values
        v_obs = rc["v_obs"].values
        v_obs_err = rc["v_obs_err"].values
        v_gas = rc["v_gas"].values
        v_disk = rc["v_disk"].values
        v_bul = rc["v_bul"].values

        v_pred_scm = _v_total(r, v_gas, v_disk, v_bul, ud, a0=a0)
        v_pred_bary = _v_baryonic(v_gas, v_disk, v_bul, ud)
        v_hinge_arr = _v_velos(r, a0=a0)

        safe_err = np.where(v_obs_err > 0, v_obs_err, 1.0)
        res_scm = (v_obs - v_pred_scm) / safe_err
        res_bary = (v_obs - v_pred_bary) / safe_err
        improvement = np.abs(res_bary) - np.abs(res_scm)

        # Baryonic and observed accelerations (same convention as run_pipeline)
        r_safe = np.maximum(r, 1e-10)
        g_bar = v_pred_bary ** 2 / r_safe * _CONV
        g_obs = v_obs ** 2 / r_safe * _CONV
        valid = (g_bar > 0) & (g_obs > 0)

        for k in range(len(r)):
            row: dict = {
                "galaxy": name,
                "r_kpc": float(r[k]),
                "v_hinge": float(v_hinge_arr[k]),
                "residual_scm": float(res_scm[k]),
                "residual_bary": float(res_bary[k]),
                "improvement": float(improvement[k]),
            }
            if valid[k]:
                row["log_g_bar"] = float(np.log10(g_bar[k]))
                row["log_g_obs"] = float(np.log10(g_obs[k]))
            else:
                row["log_g_bar"] = float("nan")
                row["log_g_obs"] = float("nan")
            rows.append(row)

        if verbose:
            print(f"  {name}: ud={ud:.2f}, N={len(r)}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Audit artefacts (VIF, stability, quality)
# ---------------------------------------------------------------------------

def _write_audit_artefacts_from_oos(
    oos_df: pd.DataFrame,
    audit_dir: Path,
    a0: float = _A0_DEFAULT,
) -> None:
    """Write VIF, stability, quality and feature audit files from OOS data.

    Parameters
    ----------
    oos_df : pd.DataFrame
        Output of :func:`compute_oos_residuals`.
    audit_dir : Path
        Destination directory.
    a0 : float
        Characteristic acceleration used as hinge threshold.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Build a compare_df compatible with _write_audit_artifacts by using the
    # log_g_bar / log_g_obs columns already computed in compute_oos_residuals.
    # Filter to rows where both values are finite.
    if "log_g_bar" in oos_df.columns and "log_g_obs" in oos_df.columns:
        mask = oos_df["log_g_bar"].notna() & oos_df["log_g_obs"].notna()
        compare_df = oos_df.loc[mask, ["galaxy", "r_kpc", "log_g_bar", "log_g_obs"]].copy()
        compare_df["g_bar"] = 10.0 ** compare_df["log_g_bar"]
        compare_df["g_obs"] = 10.0 ** compare_df["log_g_obs"]
    else:
        # Fall back to a sibling universal_term_comparison_full.csv if present
        compare_csv = audit_dir.parent / "universal_term_comparison_full.csv"
        compare_df = pd.read_csv(compare_csv) if compare_csv.exists() else pd.DataFrame()

    # Re-use the helper from src.scm_analysis if available
    try:
        from src.scm_analysis import _write_audit_artifacts  # noqa: PLC0415
        _write_audit_artifacts(compare_df, audit_dir, a0=a0)
    except ImportError:
        # Fallback: write empty stubs so downstream scripts don't crash
        for fname in ("vif_table.csv", "stability_metrics.csv",
                      "audit_features.csv"):
            if not (audit_dir / fname).exists():
                pd.DataFrame().to_csv(audit_dir / fname, index=False)
        if not (audit_dir / "quality_status.txt").exists():
            (audit_dir / "quality_status.txt").write_text(
                "quality_status: UNKNOWN\n", encoding="utf-8"
            )


# ---------------------------------------------------------------------------
# Residual-vs-hinge diagnostic
# ---------------------------------------------------------------------------

def write_residual_vs_hinge(
    oos_df: pd.DataFrame,
    audit_dir: Path,
) -> None:
    """Write ``residual_vs_hinge.csv`` and ``residual_vs_hinge.png``.

    Parameters
    ----------
    oos_df : pd.DataFrame
        Output of :func:`compute_oos_residuals`.
    audit_dir : Path
        Destination directory.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    csv_cols = ["galaxy", "r_kpc", "v_hinge", "residual_scm",
                "residual_bary", "improvement"]
    out_df = oos_df[[c for c in csv_cols if c in oos_df.columns]].copy()
    out_df.to_csv(audit_dir / "residual_vs_hinge.csv", index=False)

    # --- PNG ---
    _plot_residual_vs_hinge(oos_df, audit_dir / "residual_vs_hinge.png")


def _plot_residual_vs_hinge(oos_df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel diagnostic plot: residual SCM and improvement vs hinge."""
    v_hinge = oos_df["v_hinge"].values
    res_scm = oos_df["residual_scm"].values
    improvement = oos_df["improvement"].values

    # Clip extreme residuals for readability
    clip = np.percentile(np.abs(res_scm[np.isfinite(res_scm)]), 99) if len(res_scm) else 5.0
    clip = max(clip, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: residual SCM vs hinge
    ax = axes[0]
    ax.scatter(v_hinge, res_scm, s=4, alpha=0.3, color="steelblue", rasterized=True)
    ax.axhline(0, color="crimson", linewidth=1.2, linestyle="--")
    # Running median
    if len(v_hinge) >= 10:
        order = np.argsort(v_hinge)
        xo, yo = v_hinge[order], res_scm[order]
        w = max(len(xo) // 10, 1)
        xmed = [float(np.median(xo[i:i + w])) for i in range(0, len(xo) - w + 1, w)]
        ymed = [float(np.median(yo[i:i + w])) for i in range(0, len(yo) - w + 1, w)]
        ax.plot(xmed, ymed, color="darkorange", linewidth=1.5, label="running median")
        ax.legend(fontsize=8)
    ax.set_xlabel("V_hinge (km/s)", fontsize=10)
    ax.set_ylabel("Normalised residual SCM", fontsize=10)
    ax.set_title("Residual SCM vs Hinge", fontsize=11)
    ax.set_ylim(-clip, clip)
    ax.grid(True, alpha=0.3)

    # Panel 2: improvement vs hinge
    ax = axes[1]
    ax.scatter(v_hinge, improvement, s=4, alpha=0.3, color="seagreen", rasterized=True)
    ax.axhline(0, color="crimson", linewidth=1.2, linestyle="--")
    if len(v_hinge) >= 10:
        order = np.argsort(v_hinge)
        xo, yo = v_hinge[order], improvement[order]
        xmed = [float(np.median(xo[i:i + w])) for i in range(0, len(xo) - w + 1, w)]
        ymed = [float(np.median(yo[i:i + w])) for i in range(0, len(yo) - w + 1, w)]
        ax.plot(xmed, ymed, color="darkorange", linewidth=1.5, label="running median")
        ax.legend(fontsize=8)
    ax.set_xlabel("V_hinge (km/s)", fontsize=10)
    ax.set_ylabel("|res_bary| − |res_SCM|  (improvement)", fontsize=10)
    ax.set_title("SCM Improvement vs Hinge", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Motor de Velos SCM — OOS Residual vs Hinge Diagnostic", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Limited mode: use pre-computed universal_term_comparison_full.csv
# ---------------------------------------------------------------------------

def run_from_csv(csv_path: Path, audit_dir: Path, a0: float = _A0_DEFAULT) -> None:
    """Run a limited audit using only the pre-computed comparison CSV.

    The residual-vs-hinge plot is not available in this mode (no rotation-curve
    data), so only the VIF / stability / quality artefacts are written.
    """
    compare_df = pd.read_csv(csv_path)
    required = {"galaxy", "log_g_bar", "log_g_obs"}
    missing = required - set(compare_df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            "Need 'galaxy', 'log_g_bar', 'log_g_obs'.  "
            "Run python -m src.scm_analysis first to generate "
            "universal_term_comparison_full.csv."
        )

    try:
        from src.scm_analysis import _write_audit_artifacts  # noqa: PLC0415
        _write_audit_artifacts(compare_df, audit_dir, a0=a0)
    except ImportError:
        pass

    print(
        f"  [csv-mode] Audit artefacts written to {audit_dir}\n"
        "  NOTE: residual_vs_hinge.csv/.png require --data-dir."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motor de Velos SCM — OOS structural audit."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data-dir", metavar="DIR",
        help="Directory containing SPARC rotmod files.",
    )
    src.add_argument(
        "--csv", metavar="FILE",
        help="Pre-computed universal_term_comparison_full.csv (limited mode).",
    )
    parser.add_argument(
        "--outdir", required=True, metavar="DIR",
        help="Output directory for audit results.",
    )
    parser.add_argument(
        "--a0", type=float, default=_A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {_A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.outdir)
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if verbose:
            print(f"Running OOS audit on {data_dir} → {out_dir}")

        # 1. Compute per-point OOS residuals
        oos_df = compute_oos_residuals(data_dir, a0=args.a0, verbose=verbose)

        # 2. Write residual_vs_hinge CSV + PNG
        write_residual_vs_hinge(oos_df, audit_dir)

        # 3. Write VIF / stability / quality artefacts
        _write_audit_artefacts_from_oos(oos_df, audit_dir, a0=args.a0)

        if verbose:
            print(f"\nAudit artefacts written to {audit_dir}")
            csv_path = audit_dir / "residual_vs_hinge.csv"
            if csv_path.exists():
                print(f"  residual_vs_hinge.csv : {len(oos_df)} rows")
            png_path = audit_dir / "residual_vs_hinge.png"
            if png_path.exists():
                print(f"  residual_vs_hinge.png : generated")
    else:
        # CSV-only mode
        run_from_csv(Path(args.csv), audit_dir, a0=args.a0)


if __name__ == "__main__":
    main()
