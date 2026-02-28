"""
scripts/blind_test_little_things.py — Blind-test CLI for LITTLE THINGS dataset.

Applies two baseline models to the LITTLE THINGS dwarf-galaxy sample and
reports predictions vs observed flat rotation velocities:

  btfr        — Baryonic Tully–Fisher Relation (deep-MOND limit):
                    Vflat = (G · Mbar · a0)^(1/4)
                with Mbar = M_star × (1 + gas_fraction).

  interp      — Interpolation-based model (deep-MOND BTFR via g_bar and j):
                    log Vflat = (log_gbar + 2·log_j + C) / 6
                where C captures SI/observational unit conversions and a0.

Input
-----
data/little_things_global.csv  (or --csv PATH)
  Required columns: galaxy_id, logM, logVobs, log_gbar, log_j

Outputs written to --out DIR (default: results/blind_test_lt):
  predictions.csv  — per-galaxy observed and predicted log Vflat
  summary.csv      — per-model RMSE, MAE, bias, and N

Usage
-----
::

    python scripts/blind_test_little_things.py

    python scripts/blind_test_little_things.py \\
        --csv data/little_things_global.csv \\
        --out results/blind_test_lt \\
        --gas-fraction 5.0 \\
        --a0 1.2e-10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

KPC_TO_M: float = 3.085677581e19   # metres per kiloparsec (IAU 2012)
KMS_TO_MS: float = 1.0e3           # m/s per km/s
G_SI: float = 6.674e-11            # m³ kg⁻¹ s⁻²
MSUN_KG: float = 1.989e30          # kg per solar mass

A0_DEFAULT: float = 1.2e-10        # characteristic acceleration (m/s²)
GAS_FRACTION_DEFAULT: float = 5.0  # M_gas / M_star for gas-dominated dwarfs

# Required columns in the input CSV
REQUIRED_COLS: list[str] = ["galaxy_id", "logM", "logVobs", "log_gbar", "log_j"]

# Output column order for predictions.csv
PRED_COLS: list[str] = [
    "galaxy_id",
    "logVobs",
    "logV_btfr",
    "logV_interp",
    "residual_btfr",
    "residual_interp",
]


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def _interp_constant(a0: float = A0_DEFAULT) -> float:
    """Unit-conversion constant for the interpolation model.

    Derivation (SI throughout, then convert to km/s and kpc·km/s):

        Vflat⁶ = g_bar · a0 · j²         [SI: m⁶/s⁶ = m/s² · m/s² · m⁴/s²]

    Converting to observational units (Vflat in km/s, j in kpc·km/s):

        6·log(Vflat_kms) = log(g_bar) + log(a0)
                         + 2·log(j_kpc·kms) + 2·log(KPC_TO_M·KMS_TO_MS) − 18

    Returns
    -------
    float
        C = log10(a0) + 2·log10(KPC_TO_M · KMS_TO_MS) − 18
    """
    return float(np.log10(a0) + 2.0 * np.log10(KPC_TO_M * KMS_TO_MS) - 18.0)


def predict_logv_btfr(
    logM: np.ndarray,
    gas_fraction: float = GAS_FRACTION_DEFAULT,
    a0: float = A0_DEFAULT,
) -> np.ndarray:
    """Predict log10(Vflat / km/s) from stellar mass via the deep-MOND BTFR.

    Model: Vflat = (G · Mbar · a0)^(1/4)  with Mbar = M_star · (1 + gas_fraction).

    Parameters
    ----------
    logM : array_like
        log10(M_star / M_sun).
    gas_fraction : float
        Ratio M_gas / M_star (default 5.0).
    a0 : float
        Characteristic acceleration in m/s².

    Returns
    -------
    ndarray
        log10(Vflat / km/s).
    """
    logM = np.asarray(logM, dtype=float)
    logM_bar = logM + np.log10(1.0 + gas_fraction)
    # log10(Vflat_kms) = 0.25 * log10(G · Mbar · a0) − 3
    log_GaM = np.log10(G_SI * MSUN_KG * a0)  # log10(G * 1 Msun * a0) [SI]
    return 0.25 * (logM_bar + log_GaM) - 3.0


def predict_logv_interp(
    log_gbar: np.ndarray,
    log_j: np.ndarray,
    a0: float = A0_DEFAULT,
) -> np.ndarray:
    """Predict log10(Vflat / km/s) from g_bar and specific angular momentum.

    Model (deep-MOND self-consistent BTFR):

        Vflat⁶ = g_bar · a0 · j²

    with j = R · Vflat (specific angular momentum) substituted self-consistently.

    Parameters
    ----------
    log_gbar : array_like
        log10(g_bar / m·s⁻²).
    log_j : array_like
        log10(j / kpc·km·s⁻¹).
    a0 : float
        Characteristic acceleration in m/s².

    Returns
    -------
    ndarray
        log10(Vflat / km/s).
    """
    log_gbar = np.asarray(log_gbar, dtype=float)
    log_j = np.asarray(log_j, dtype=float)
    C = _interp_constant(a0)
    return (log_gbar + 2.0 * log_j + C) / 6.0


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the LITTLE THINGS global dataset.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Validated dataset.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_blind_test(
    csv_path: Path,
    out_dir: Path,
    gas_fraction: float = GAS_FRACTION_DEFAULT,
    a0: float = A0_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the blind-test pipeline.

    Parameters
    ----------
    csv_path : Path
        Input dataset CSV.
    out_dir : Path
        Directory where predictions.csv and summary.csv are written.
    gas_fraction : float
        M_gas / M_star ratio for the BTFR model.
    a0 : float
        Characteristic acceleration (m/s²).

    Returns
    -------
    predictions : pd.DataFrame
        Per-galaxy predictions and residuals.
    summary : pd.DataFrame
        Per-model RMSE, MAE, bias, and N statistics.
    """
    df = load_dataset(csv_path)

    logV_btfr = predict_logv_btfr(df["logM"].values, gas_fraction=gas_fraction, a0=a0)
    logV_interp = predict_logv_interp(
        df["log_gbar"].values, df["log_j"].values, a0=a0
    )

    res_btfr = logV_btfr - df["logVobs"].values
    res_interp = logV_interp - df["logVobs"].values

    predictions = pd.DataFrame(
        {
            "galaxy_id": df["galaxy_id"],
            "logVobs": df["logVobs"],
            "logV_btfr": np.round(logV_btfr, 4),
            "logV_interp": np.round(logV_interp, 4),
            "residual_btfr": np.round(res_btfr, 4),
            "residual_interp": np.round(res_interp, 4),
        }
    )

    n = len(df)
    summary_rows = []
    for model_name, residuals in [("btfr", res_btfr), ("interp", res_interp)]:
        summary_rows.append(
            {
                "model": model_name,
                "N": n,
                "RMSE_dex": float(np.sqrt(np.mean(residuals**2))),
                "MAE_dex": float(np.mean(np.abs(residuals))),
                "bias_dex": float(np.mean(residuals)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    return predictions, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blind-test LITTLE THINGS dataset against BTFR and "
            "interpolation-based models."
        )
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent.parent / "data" / "little_things_global.csv"),
        metavar="FILE",
        help="Input dataset CSV (default: data/little_things_global.csv).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "results" / "blind_test_lt"),
        metavar="DIR",
        help="Output directory (default: results/blind_test_lt).",
    )
    parser.add_argument(
        "--gas-fraction",
        type=float,
        default=GAS_FRACTION_DEFAULT,
        dest="gas_fraction",
        help=f"M_gas/M_star ratio for BTFR model (default: {GAS_FRACTION_DEFAULT}).",
    )
    parser.add_argument(
        "--a0",
        type=float,
        default=A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {A0_DEFAULT:.2e}).",
    )
    return parser.parse_args(argv)


def _print_summary(predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    sep = "=" * 60
    print(sep)
    print("  Motor de Velos SCM — LITTLE THINGS Blind Test")
    print(sep)
    print(f"  Galaxies: {len(predictions)}")
    print()
    print(f"  {'Model':<10} {'N':>4} {'RMSE (dex)':>12} {'MAE (dex)':>11} {'Bias (dex)':>12}")
    print("  " + "-" * 53)
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:<10} {int(row['N']):>4} "
            f"{row['RMSE_dex']:>12.4f} {row['MAE_dex']:>11.4f} "
            f"{row['bias_dex']:>12.4f}"
        )
    print(sep)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    predictions, summary = run_blind_test(
        csv_path,
        out_dir,
        gas_fraction=args.gas_fraction,
        a0=args.a0,
    )

    _print_summary(predictions, summary)
    print(f"\n  Results written to: {out_dir}")
    print(f"    predictions.csv ({len(predictions)} rows)")
    print(f"    summary.csv     ({len(summary)} rows)")


if __name__ == "__main__":
    main()
