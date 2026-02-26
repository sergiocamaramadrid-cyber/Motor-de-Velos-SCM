"""
scripts/scm_oos_validation.py — Out-of-sample (OOS) validation for the Motor de Velos SCM.

Methodology
-----------
For each galaxy in the SPARC sample the rotation curve is sorted by radius
and split at the median: the inner half is the **training set** and the outer
half is the **OOS test set**.  The disk mass-to-light ratio (upsilon_disk) is
fitted on the training set independently for two models:

  scm        — Motor de Velos baseline: V²_total = V²_bar + V²_velos
  baryonic   — Baryonic-only baseline: V²_total = V²_bar  (no velos term)

The fit quality on the held-out OOS test set is measured as:

  rms_dex = RMS of log10(|V_obs| / |V_pred|)    (standard RAR/SPARC metric)

The per-galaxy OOS advantage of the SCM model is:

  ΔRMSE_out = rms_dex(baryonic) − rms_dex(scm)

Positive ΔRMSE_out means the SCM model predicts the *outer* rotation curve
better than the baryonic-only model.

Aggregate statistics written to --out:
  oos_per_galaxy.csv   — per-galaxy OOS statistics
  oos_summary.csv      — N_valid, success_pct, median_delta_rmse_out,
                         wilcoxon_statistic, wilcoxon_pvalue
  oos_validation.log   — full run log

Usage
-----
::

    python scripts/scm_oos_validation.py \\
        --data-dir data/SPARC \\
        --out results/oos_validation

    python scripts/scm_oos_validation.py \\
        --data-dir data/SPARC \\
        --out results/oos_validation \\
        --a0 1.2e-10 \\
        --train-frac 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A0_DEFAULT = 1.2e-10        # characteristic acceleration (m/s²)
KPC_TO_M = 3.085677581e16   # metres per kiloparsec (IAU 2012)
_CONV = 1e6 / KPC_TO_M      # (km/s)²/kpc → m/s²
_VEL_FLOOR = 1e-10          # km/s — prevents log(0) in rms_dex

TRAIN_FRAC = 0.5            # fraction of radial points used for training (inner)
MIN_TRAIN_POINTS = 4        # minimum training points to fit upsilon_disk
MIN_TEST_POINTS = 2         # minimum OOS test points for evaluation

_SEP = "=" * 64

# ---------------------------------------------------------------------------
# Data loading helpers (mirror scm_analysis.py)
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
# Model helpers
# ---------------------------------------------------------------------------


def _v_baryonic(r: np.ndarray, v_gas: np.ndarray, v_disk: np.ndarray,
                v_bul: np.ndarray, upsilon_disk: float) -> np.ndarray:
    """Signed baryonic velocity with upsilon_disk and fixed upsilon_bul=0.7."""
    v2 = (v_gas * np.abs(v_gas)
          + upsilon_disk * v_disk * np.abs(v_disk)
          + 0.7 * v_bul * np.abs(v_bul))
    return np.sign(v2) * np.sqrt(np.abs(v2))


def _v_total(r: np.ndarray, v_gas: np.ndarray, v_disk: np.ndarray,
             v_bul: np.ndarray, upsilon_disk: float,
             a0: float, include_velos: bool) -> np.ndarray:
    """Total predicted velocity for SCM (include_velos=True) or baryonic-only."""
    vb = _v_baryonic(r, v_gas, v_disk, v_bul, upsilon_disk)
    vb2 = vb * np.abs(vb)
    if include_velos:
        a0_kpc = a0 / _CONV          # convert m/s² → (km/s)²/kpc
        vv2 = a0_kpc * np.maximum(r, 0.0)
    else:
        vv2 = 0.0
    v2 = vb2 + vv2
    return np.sign(v2) * np.sqrt(np.abs(v2))


def _chi2_reduced(v_obs: np.ndarray, v_obs_err: np.ndarray,
                  v_pred: np.ndarray) -> float:
    """Reduced chi-squared with 1 free parameter (upsilon_disk)."""
    safe_err = np.where(v_obs_err > 0, v_obs_err, 1.0)
    res = (v_obs - v_pred) / safe_err
    dof = max(len(v_obs) - 1, 1)
    return float(np.sum(res ** 2) / dof)


def _rms_dex(v_obs: np.ndarray, v_pred: np.ndarray) -> float:
    """RMS of log10(|V_obs| / |V_pred|) — standard galactic dynamics metric."""
    safe_vp = np.where(np.abs(v_pred) > 0, np.abs(v_pred), _VEL_FLOOR)
    log_ratio = np.log10(np.maximum(np.abs(v_obs), _VEL_FLOOR) / safe_vp)
    return float(np.sqrt(np.mean(log_ratio ** 2)))


def _fit_upsilon_disk(rc: pd.DataFrame, a0: float, include_velos: bool) -> float:
    """Fit upsilon_disk on *rc* minimising chi2_reduced."""
    r = rc["r"].values
    v_obs = rc["v_obs"].values
    v_obs_err = rc["v_obs_err"].values
    v_gas = rc["v_gas"].values
    v_disk = rc["v_disk"].values
    v_bul = rc["v_bul"].values

    def objective(ud: float) -> float:
        vp = _v_total(r, v_gas, v_disk, v_bul, ud, a0, include_velos)
        return _chi2_reduced(v_obs, v_obs_err, vp)

    result = minimize_scalar(objective, bounds=(0.1, 5.0), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Per-galaxy OOS validation
# ---------------------------------------------------------------------------


def oos_validate_galaxy(rc: pd.DataFrame, a0: float = A0_DEFAULT,
                        train_frac: float = TRAIN_FRAC) -> dict | None:
    """Perform radial-split OOS validation for a single galaxy.

    The rotation-curve rows are sorted by radius; the inner ``train_frac``
    fraction is used for training and the remaining outer fraction for OOS
    evaluation.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation-curve data with columns
        ``['r', 'v_obs', 'v_obs_err', 'v_gas', 'v_disk', 'v_bul']``.
    a0 : float
        Characteristic acceleration (m/s²).
    train_frac : float
        Fraction of radial points (sorted by r) used for training.

    Returns
    -------
    dict or None
        Returns ``None`` when the galaxy has too few points for a valid OOS
        split.  Otherwise returns a dict with keys:
        ``n_train``, ``n_test``, ``rmse_scm_out``, ``rmse_bar_out``,
        ``delta_rmse_out``, ``scm_wins``.
    """
    rc = rc.sort_values("r").reset_index(drop=True)
    n = len(rc)
    n_train = max(int(np.ceil(n * train_frac)), 1)
    n_test = n - n_train

    if n_train < MIN_TRAIN_POINTS or n_test < MIN_TEST_POINTS:
        return None

    rc_train = rc.iloc[:n_train]
    rc_test = rc.iloc[n_train:]

    # Fit each model on training data
    ud_scm = _fit_upsilon_disk(rc_train, a0, include_velos=True)
    ud_bar = _fit_upsilon_disk(rc_train, a0, include_velos=False)

    # Evaluate rms_dex on OOS test data
    r_t = rc_test["r"].values
    v_obs_t = rc_test["v_obs"].values
    v_gas_t = rc_test["v_gas"].values
    v_disk_t = rc_test["v_disk"].values
    v_bul_t = rc_test["v_bul"].values

    vp_scm = _v_total(r_t, v_gas_t, v_disk_t, v_bul_t, ud_scm, a0, True)
    vp_bar = _v_total(r_t, v_gas_t, v_disk_t, v_bul_t, ud_bar, a0, False)

    rmse_scm = _rms_dex(v_obs_t, vp_scm)
    rmse_bar = _rms_dex(v_obs_t, vp_bar)
    delta = rmse_bar - rmse_scm  # positive → SCM wins

    return {
        "n_train": n_train,
        "n_test": n_test,
        "rmse_scm_out": rmse_scm,
        "rmse_bar_out": rmse_bar,
        "delta_rmse_out": delta,
        "scm_wins": bool(delta > 0),
    }


# ---------------------------------------------------------------------------
# Full dataset run
# ---------------------------------------------------------------------------


def run_oos_validation(data_dir, out_dir, a0: float = A0_DEFAULT,
                       train_frac: float = TRAIN_FRAC,
                       verbose: bool = True) -> dict:
    """Run OOS validation across all galaxies in *data_dir*.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing SPARC rotmod files and the galaxy table.
    out_dir : str or Path
        Output directory; created if it does not exist.
    a0 : float
        Characteristic acceleration (m/s²).
    train_frac : float
        Fraction of radial points (inner) used for training.
    verbose : bool
        Print per-galaxy progress to stdout.

    Returns
    -------
    dict
        Aggregate summary with keys:
        ``N_valid``, ``success_pct``, ``median_delta_rmse_out``,
        ``wilcoxon_statistic``, ``wilcoxon_pvalue``.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    galaxy_table = _load_galaxy_table(data_dir)
    galaxy_names = galaxy_table["Galaxy"].tolist()

    per_galaxy: list[dict] = []
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        if verbose:
            print(msg)
        log_lines.append(msg)

    _log(f"  OOS validation — {len(galaxy_names)} galaxies in table")
    _log(f"  a0 = {a0:.2e} m/s²  |  train_frac = {train_frac:.2f}")
    _log(f"  min_train = {MIN_TRAIN_POINTS}  |  min_test = {MIN_TEST_POINTS}")

    for name in galaxy_names:
        try:
            rc = _load_rotation_curve(data_dir, name)
        except FileNotFoundError:
            _log(f"  [skip] {name}: rotation curve not found")
            continue

        result = oos_validate_galaxy(rc, a0=a0, train_frac=train_frac)
        if result is None:
            _log(f"  [skip] {name}: too few points for OOS split")
            continue

        per_galaxy.append({"galaxy": name, **result})
        if verbose:
            sign = "✓" if result["scm_wins"] else "✗"
            _log(f"  {sign} {name}: ΔRMSE={result['delta_rmse_out']:+.4f} "
                 f"(scm={result['rmse_scm_out']:.4f}, bar={result['rmse_bar_out']:.4f})")

    per_df = pd.DataFrame(per_galaxy)

    if per_df.empty:
        summary: dict = {
            "N_valid": 0,
            "success_pct": float("nan"),
            "median_delta_rmse_out": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_pvalue": float("nan"),
        }
        _log("\n  No valid galaxies for OOS evaluation.")
    else:
        N_valid = len(per_df)
        success_pct = float(per_df["scm_wins"].mean() * 100.0)
        median_delta = float(per_df["delta_rmse_out"].median())
        delta_arr = per_df["delta_rmse_out"].values

        # Wilcoxon signed-rank test (one-sided: H1 = SCM wins, i.e. delta > 0)
        # Requires at least 1 non-zero difference; falls back to NaN otherwise.
        if len(delta_arr) >= 1 and np.any(delta_arr != 0):
            wstat, wpval = wilcoxon(delta_arr, alternative="greater")
            wstat = float(wstat)
            wpval = float(wpval)
        else:
            wstat = float("nan")
            wpval = float("nan")

        summary = {
            "N_valid": N_valid,
            "success_pct": success_pct,
            "median_delta_rmse_out": median_delta,
            "wilcoxon_statistic": wstat,
            "wilcoxon_pvalue": wpval,
        }

        _log(f"\n  N_valid            : {N_valid}")
        _log(f"  Success %          : {success_pct:.1f}%")
        _log(f"  Median ΔRMSE_out   : {median_delta:+.4f} dex")
        if not np.isnan(wstat):
            _log(f"  Wilcoxon statistic : {wstat:.1f}")
            _log(f"  Wilcoxon p-value   : {wpval:.4e}")

    # Write outputs
    per_df.to_csv(out_dir / "oos_per_galaxy.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "oos_summary.csv", index=False)
    (out_dir / "oos_validation.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    if verbose:
        print(f"\n  Results written to {out_dir}")

    return summary


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(summary: dict, a0: float, data_dir: str) -> list[str]:
    """Format the OOS validation summary as a list of printable lines."""
    lines = [
        _SEP,
        "  Motor de Velos SCM — Out-of-Sample Validation Report",
        _SEP,
        f"  Data dir     : {data_dir}",
        f"  a0           : {a0:.2e} m/s²",
        f"  Split        : inner {TRAIN_FRAC:.0%} train / outer {1 - TRAIN_FRAC:.0%} test",
        f"  Metric       : rms_dex = RMS of log10(V_obs / V_pred)",
        "",
        f"  N_valid                : {summary['N_valid']}",
    ]
    if not np.isnan(summary.get("success_pct", float("nan"))):
        lines += [
            f"  Success (SCM wins)     : {summary['success_pct']:.1f}%",
            f"  Median ΔRMSE_out       : {summary['median_delta_rmse_out']:+.4f} dex",
        ]
        if not np.isnan(summary.get("wilcoxon_pvalue", float("nan"))):
            lines += [
                f"  Wilcoxon statistic     : {summary['wilcoxon_statistic']:.1f}",
                f"  Wilcoxon p-value       : {summary['wilcoxon_pvalue']:.4e}",
            ]
    lines += ["", _SEP]
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Out-of-sample validation for the Motor de Velos SCM."
    )
    parser.add_argument(
        "--data-dir", required=True, metavar="DIR",
        help="Directory containing SPARC rotmod files and galaxy table.",
    )
    parser.add_argument(
        "--out", default="results/oos_validation", metavar="DIR",
        help="Output directory (default: results/oos_validation).",
    )
    parser.add_argument(
        "--a0", type=float, default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--train-frac", type=float, default=TRAIN_FRAC, dest="train_frac",
        help=(f"Inner fraction of radial points used for training "
              f"(default: {TRAIN_FRAC})."),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-galaxy progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run OOS validation and print the summary report.

    Returns the summary dict so callers can inspect it programmatically.
    """
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    summary = run_oos_validation(
        data_dir=data_dir,
        out_dir=args.out,
        a0=args.a0,
        train_frac=args.train_frac,
        verbose=not args.quiet,
    )

    report_lines = format_report(summary, args.a0, str(data_dir))
    for line in report_lines:
        print(line)

    return summary


if __name__ == "__main__":
    main()
