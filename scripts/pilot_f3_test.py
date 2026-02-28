"""
scripts/pilot_f3_test.py — Phase-1 pilot test: F3 for 4 dwarf galaxies.

Theory
------
F3 is defined as the residual of the observed flat rotation velocity from
the deep-MOND interpolation-model prediction:

    F3 = log10(V_obs) − log10(V_model)

where V_model is derived from the self-consistent deep-MOND BTFR:

    V_flat^6 = g_bar · a0 · j²

i.e.  log10(V_model) = (log10(g_bar) + 2·log10(j) + C) / 6

with C = log10(a0) + 2·log10(kpc_to_m · km_to_m) − 18.

Interpretation:
  F3 ≈ 0   → rotation curve is flat (observed matches the deep-MOND prediction)
  F3 > 0   → still rising (V_obs > V_model: galaxy above the RAR in velocity space)
  F3 < 0   → declining (V_obs < V_model: galaxy below the deep-MOND prediction)

Scientific context
------------------
Typical F3 values for gas-dominated dwarf galaxies: −0.05 to +0.15.
With only N=4 galaxies the goal is *not* a definitive p-value but:
  - checking the sign of the effect (positive → physically coherent)
  - estimating the order of magnitude
  - identifying obvious outliers before expanding the sample

Usage
-----
::

    python scripts/pilot_f3_test.py

    python scripts/pilot_f3_test.py \\
        --csv  data/little_things_global.csv \\
        --out  results \\
        --a0   1.2e-10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.blind_test_little_things import (
    A0_DEFAULT,
    load_dataset,
    predict_logv_interp,
)

# ---------------------------------------------------------------------------
# Pilot galaxy list (Phase 1 — mandatory first execution)
# ---------------------------------------------------------------------------

PILOT_GALAXIES: list[str] = ["DDO69", "DDO70", "DDO75", "DDO210"]

# Output CSV column order
F3_COLS: list[str] = ["galaxy_id", "log_gbar", "log_j", "logVobs", "logV_model", "F3"]

_SEP = "=" * 65


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_f3(df: pd.DataFrame, a0: float = A0_DEFAULT) -> pd.DataFrame:
    """Compute F3 for each row of the dataset.

    F3 = log10(V_obs) − log10(V_model)

    where V_model is the deep-MOND interpolation-model prediction:

        log10(V_model) = (log_gbar + 2·log_j + C) / 6

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ``galaxy_id``, ``log_gbar``, ``log_j``,
        ``logVobs``.
    a0 : float
        Characteristic acceleration in m/s².

    Returns
    -------
    pd.DataFrame
        Columns: ``galaxy_id``, ``log_gbar``, ``log_j``, ``logVobs``,
        ``logV_model``, ``F3``.
    """
    logV_model = predict_logv_interp(
        df["log_gbar"].values, df["log_j"].values, a0=a0
    )
    f3 = df["logVobs"].values - logV_model

    return pd.DataFrame(
        {
            "galaxy_id": df["galaxy_id"].values,
            "log_gbar": df["log_gbar"].values,
            "log_j": df["log_j"].values,
            "logVobs": df["logVobs"].values,
            "logV_model": np.round(logV_model, 6),
            "F3": np.round(f3, 6),
        }
    )


def run_pilot_f3(
    csv_path: Path,
    out_dir: Path,
    a0: float = A0_DEFAULT,
    galaxies: list[str] | None = None,
) -> tuple[pd.DataFrame, object]:
    """Run the F3 pilot test for the specified galaxies.

    Parameters
    ----------
    csv_path : Path
        Input dataset CSV (must contain LITTLE THINGS required columns).
    out_dir : Path
        Directory where ``F3_values.csv`` is written.
    a0 : float
        Characteristic acceleration in m/s².
    galaxies : list[str] or None
        Galaxy IDs to include.  Defaults to :data:`PILOT_GALAXIES`.

    Returns
    -------
    f3_df : pd.DataFrame
        Per-galaxy F3 values and model predictions.
    ols_result : statsmodels RegressionResults
        OLS fit of F3 ~ log_gbar (with a constant).

    Raises
    ------
    ValueError
        If none of the requested galaxies are found in the dataset.
    """
    if galaxies is None:
        galaxies = PILOT_GALAXIES

    full_df = load_dataset(csv_path)
    mask = full_df["galaxy_id"].isin(galaxies)
    df = full_df[mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"None of {galaxies} found in {csv_path}. "
            "Check galaxy_id spelling (e.g. 'DDO69', not 'DDO 69')."
        )

    f3_df = compute_f3(df, a0=a0)

    # OLS regression: F3 ~ 1 + log_gbar
    # Tests whether F3 correlates systematically with baryonic-acceleration depth.
    X = sm.add_constant(f3_df["log_gbar"].values)
    y = f3_df["F3"].values
    ols_model = sm.OLS(y, X)
    ols_result = ols_model.fit()

    out_dir.mkdir(parents=True, exist_ok=True)
    f3_df[F3_COLS].to_csv(out_dir / "F3_values.csv", index=False)

    return f3_df, ols_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1 pilot test: compute F3 residuals for 4 dwarf galaxies "
            "and run an OLS regression."
        )
    )
    parser.add_argument(
        "--csv",
        default=str(
            Path(__file__).parent.parent / "data" / "little_things_global.csv"
        ),
        metavar="FILE",
        help="Input dataset CSV (default: data/little_things_global.csv).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "results"),
        metavar="DIR",
        help="Output directory for F3_values.csv (default: results/).",
    )
    parser.add_argument(
        "--a0",
        type=float,
        default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    return parser.parse_args(argv)


def _print_f3_table(f3_df: pd.DataFrame) -> None:
    """Print a compact F3 table to stdout."""
    print(_SEP)
    print("  Motor de Velos SCM — F3 Pilot Test (N=4)")
    print(_SEP)
    print(f"  {'galaxy_id':<10} {'log_gbar':>10} {'logVobs':>9} "
          f"{'logV_model':>11} {'F3':>8}")
    print("  " + "-" * 51)
    for _, row in f3_df.iterrows():
        print(
            f"  {row['galaxy_id']:<10} {row['log_gbar']:>10.4f} "
            f"{row['logVobs']:>9.4f} {row['logV_model']:>11.4f} "
            f"{row['F3']:>8.4f}"
        )
    print(_SEP)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    f3_df, ols_result = run_pilot_f3(csv_path, out_dir, a0=args.a0)

    _print_f3_table(f3_df)
    print("\n  OLS: F3 ~ 1 + log_gbar")
    print(ols_result.summary())
    print(f"\n  F3_values.csv written to: {out_dir / 'F3_values.csv'}")


if __name__ == "__main__":
    main()
