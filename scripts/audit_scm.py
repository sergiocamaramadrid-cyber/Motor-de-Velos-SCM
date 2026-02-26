"""
scripts/audit_scm.py — OOS (out-of-sample) audit for the Motor de Velos SCM.

Splits the SPARC sample into train / test galaxies, fits *upsilon_disk* per
galaxy, then evaluates each galaxy's rotation curve using:

  * SCM model  (baryonic + V_velos)
  * Baryonic-only model (no V_velos)

For every out-of-sample radial point the script records:

  * ``residual_scm``        — (v_obs − v_scm) / v_obs_err
  * ``residual_baryonic``   — (v_obs − v_bary) / v_obs_err
  * ``improvement``         — |residual_baryonic| − |residual_scm|
                              (positive = SCM closer to data)
  * ``hinge``               — max(log10(g_bar / a0), 0)

Results are written to::

    <outdir>/audit/residual_vs_hinge.csv
    <outdir>/audit/residual_vs_hinge.png
    <outdir>/audit/oos_summary.json

Usage
-----
With defaults (data in ``data/sparc``)::

    python scripts/audit_scm.py --outdir results/final_audit

Explicit data directory::

    python scripts/audit_scm.py --data-dir data/sparc --outdir results/final_audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Ensure the repo root is on the path when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.scm_analysis import (  # noqa: E402
    load_galaxy_table,
    load_rotation_curve,
    fit_galaxy,
    _A0_DEFAULT,
    _CONV,
    _MIN_RADIUS_KPC,
)
from src.scm_models import v_total, v_baryonic  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_FRACTION = 0.8   # fraction of galaxies used for "training"
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _residuals_for_galaxy(rc: pd.DataFrame, a0: float) -> tuple[dict, list[dict]]:
    """Fit *upsilon_disk* for one galaxy and return per-point OOS diagnostics.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data from :func:`~src.scm_analysis.load_rotation_curve`.
    a0 : float
        Characteristic velos acceleration (m/s²).

    Returns
    -------
    fit : dict
        Output of :func:`~src.scm_analysis.fit_galaxy`.
    rows : list of dict
        One dict per radial point with keys:
        ``r_kpc``, ``g_bar``, ``log_g_bar``, ``hinge``,
        ``residual_scm``, ``residual_baryonic``, ``improvement``.
    """
    fit = fit_galaxy(rc, a0=a0)
    ud = fit["upsilon_disk"]

    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_obs_err = rc["v_obs_err"].values
    v_gas = rc["v_gas"].values
    v_disk = rc["v_disk"].values
    v_bul = rc["v_bul"].values

    v_scm = v_total(r, v_gas, v_disk, v_bul, upsilon_disk=ud, upsilon_bul=0.7,
                    a0=a0, include_velos=True)
    v_bary = v_baryonic(r, v_gas, v_disk, v_bul, upsilon_disk=ud, upsilon_bul=0.7)

    safe_err = np.where(v_obs_err > 0, v_obs_err, 1.0)
    resid_scm = (v_obs - v_scm) / safe_err
    resid_bary = (v_obs - np.abs(v_bary)) / safe_err

    r_safe = np.maximum(r, _MIN_RADIUS_KPC)
    g_bar = v_bary ** 2 / r_safe * _CONV
    valid = g_bar > 0
    log_a0 = np.log10(a0)
    log_g_bar = np.where(valid, np.log10(np.where(valid, g_bar, 1.0)), np.nan)
    hinge = np.where(valid, np.maximum(log_g_bar - log_a0, 0.0), np.nan)

    rows = []
    for k in range(len(r)):
        if not valid[k]:
            continue
        improvement = float(abs(resid_bary[k])) - float(abs(resid_scm[k]))
        rows.append({
            "r_kpc": float(r[k]),
            "g_bar": float(g_bar[k]),
            "log_g_bar": float(log_g_bar[k]),
            "hinge": float(hinge[k]),
            "residual_scm": float(resid_scm[k]),
            "residual_baryonic": float(resid_bary[k]),
            "improvement": improvement,
        })
    return fit, rows


def run_oos_audit(data_dir: str | Path, out_dir: str | Path,
                  a0: float = _A0_DEFAULT,
                  train_fraction: float = TRAIN_FRACTION,
                  seed: int = RANDOM_SEED,
                  verbose: bool = True) -> pd.DataFrame:
    """Run the OOS audit and write all artifacts to *out_dir*/audit/.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SPARC rotation-curve files.
    out_dir : str or Path
        Root output directory; artefacts go into ``<out_dir>/audit/``.
    a0 : float
        Characteristic velos acceleration.
    train_fraction : float
        Fraction of galaxies used as the "train" split (informational only;
        upsilon_disk is still fitted per-galaxy on all its radial points).
    seed : int
        Random seed for the train/test split.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Per-radial-point table with OOS diagnostics.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    galaxy_table = load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()

    rng = np.random.default_rng(seed)
    n_total = len(galaxy_names)
    perm = rng.permutation(n_total)
    n_train = max(1, int(np.round(train_fraction * n_total)))
    train_idx = set(int(i) for i in perm[:n_train])
    test_idx = set(int(i) for i in perm[n_train:])
    if not test_idx:          # edge-case: only 1 galaxy
        test_idx = train_idx

    test_galaxies = [galaxy_names[i] for i in sorted(test_idx)]

    if verbose:
        print(f"OOS audit: {len(train_idx)} train / {len(test_idx)} test galaxies")

    all_rows: list[dict] = []
    fit_records: list[dict] = []

    for name in test_galaxies:
        try:
            rc = load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f"  [skip] {name}: rotation curve not found", file=sys.stderr)
            continue

        fit, rows = _residuals_for_galaxy(rc, a0=a0)
        for row in rows:
            row["galaxy"] = name
        all_rows.extend(rows)
        fit_records.append({"galaxy": name, **fit})

        if verbose:
            print(f"  {name}: n_pts={fit['n_points']} chi2={fit['chi2_reduced']:.2f}")

    if not all_rows:
        if verbose:
            print("No OOS data produced — check data_dir.", file=sys.stderr)
        return pd.DataFrame()

    df_oos = pd.DataFrame(all_rows)
    # Ensure column order
    col_order = [
        "galaxy", "r_kpc", "g_bar", "log_g_bar", "hinge",
        "residual_scm", "residual_baryonic", "improvement",
    ]
    df_oos = df_oos[[c for c in col_order if c in df_oos.columns]]

    csv_path = audit_dir / "residual_vs_hinge.csv"
    df_oos.to_csv(csv_path, index=False)
    if verbose:
        print(f"  Written: {csv_path}")

    # --- Summary JSON --------------------------------------------------------
    summary = {
        "n_train_galaxies": len(train_idx),
        "n_test_galaxies": len(test_galaxies),
        "n_oos_points": len(df_oos),
        "residual_scm_median": float(df_oos["residual_scm"].median()),
        "residual_scm_std": float(df_oos["residual_scm"].std()),
        "improvement_median": float(df_oos["improvement"].median()),
        "a0": a0,
        "seed": seed,
    }
    json_path = audit_dir / "oos_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    if verbose:
        print(f"  Written: {json_path}")

    # --- Plot ----------------------------------------------------------------
    png_path = audit_dir / "residual_vs_hinge.png"
    _plot_residual_vs_hinge(df_oos, png_path, a0=a0)
    if verbose:
        print(f"  Written: {png_path}")

    return df_oos


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_residual_vs_hinge(df: pd.DataFrame, path: Path,
                             a0: float = _A0_DEFAULT) -> None:
    """Two-panel diagnostic plot saved to *path*.

    Panel 1 — SCM residual vs hinge (should be centred at 0, no slope).
    Panel 2 — Improvement vs hinge  (should trend positive for large hinge).
    """
    hinge = df["hinge"].values
    resid = df["residual_scm"].values
    imprv = df["improvement"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ----- Panel 1: residual_scm vs hinge -----------------------------------
    ax = axes[0]
    ax.scatter(hinge, resid, s=4, alpha=0.3, color="#2c7bb6", rasterized=True)
    ax.axhline(0, color="k", linewidth=1.0)

    # Running median
    order = np.argsort(hinge)
    h_s = hinge[order]
    r_s = resid[order]
    if len(h_s) >= 20:
        window = max(len(h_s) // 20, 10)
        med_x = np.array([np.median(h_s[i:i + window])
                          for i in range(0, len(h_s) - window + 1, window // 2)])
        med_y = np.array([np.median(r_s[i:i + window])
                          for i in range(0, len(r_s) - window + 1, window // 2)])
        ax.plot(med_x, med_y, color="#d7191c", linewidth=1.8, label="running median")
        ax.legend(fontsize=8)

    ax.set_xlabel(r"Hinge  $\max(\log_{10}(g_\mathrm{bar}/a_0),\,0)$", fontsize=10)
    ax.set_ylabel(r"SCM residual  $(v_\mathrm{obs}-v_\mathrm{SCM})/\sigma$", fontsize=10)
    ax.set_title("OOS SCM residual vs hinge", fontsize=11)

    # ----- Panel 2: improvement vs hinge ------------------------------------
    ax2 = axes[1]
    ax2.scatter(hinge, imprv, s=4, alpha=0.3, color="#1a9641", rasterized=True)
    ax2.axhline(0, color="k", linewidth=1.0)

    if len(h_s) >= 20:
        i_s = imprv[order]
        med_iy = np.array([np.median(i_s[i:i + window])
                           for i in range(0, len(i_s) - window + 1, window // 2)])
        ax2.plot(med_x, med_iy, color="#d7191c", linewidth=1.8, label="running median")
        ax2.legend(fontsize=8)

    ax2.set_xlabel(r"Hinge  $\max(\log_{10}(g_\mathrm{bar}/a_0),\,0)$", fontsize=10)
    ax2.set_ylabel(
        r"Improvement  $|\mathrm{res}_\mathrm{bary}|-|\mathrm{res}_\mathrm{SCM}|$",
        fontsize=10,
    )
    ax2.set_title("OOS improvement (SCM vs baryonic)", fontsize=11)

    fig.suptitle(
        f"Motor de Velos SCM — OOS Audit  ($a_0={a0:.2e}$ m/s²)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Motor de Velos SCM — OOS audit: residual vs hinge diagnostic."
        )
    )
    parser.add_argument(
        "--data-dir", default="data/sparc",
        help="Directory containing SPARC data (default: data/sparc).",
    )
    parser.add_argument(
        "--outdir", default="results/final_audit",
        help="Root output directory (default: results/final_audit).",
    )
    parser.add_argument(
        "--a0", type=float, default=_A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {_A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=TRAIN_FRACTION, dest="train_fraction",
        help=f"Fraction of galaxies used as train split (default: {TRAIN_FRACTION}).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for train/test split (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Entry point for the OOS audit script."""
    args = _parse_args(argv)
    return run_oos_audit(
        data_dir=args.data_dir,
        out_dir=args.outdir,
        a0=args.a0,
        train_fraction=args.train_fraction,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
