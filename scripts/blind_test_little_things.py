"""
blind_test_little_things.py — Blind test of the Motor de Velos SCM model
against the standard Baryonic Tully-Fisher Relation (BTFR) on LITTLE THINGS.

Usage
-----
    python blind_test_little_things.py \\
        --input little_things_global.csv \\
        --outdir resultados_lt

Input CSV columns
-----------------
    galaxy_id  : galaxy name
    logM       : log10(M_bar / M_sun)
    logVobs    : log10(V_flat / km s^-1)
    log_gbar   : log10(g_bar / m s^-2)  — outer-regime baryonic acceleration
    log_j      : specific angular momentum proxy (optional; set to 0 if absent)

Output
------
    <outdir>/predictions.csv  — per-galaxy predictions and residuals
    <outdir>/summary.csv      — RMSE and Wilcoxon statistics

Terminal output
---------------
    Galaxias: N
    RMSE SCM: X
    RMSE BTFR: Y
    Fracción mejora: Z%
    Wilcoxon p: P
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as scipy_wilcoxon

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_G_SI = 6.674e-11           # m^3 kg^-1 s^-2
_MSUN_KG = 1.989e30         # kg per solar mass
_A0_DEFAULT = 1.2e-10       # m s^-2  — characteristic MOND/SCM acceleration
_KMS_TO_MS = 1.0e3          # km/s → m/s
_EPSILON = 1e-30            # guard against division by zero in log/sqrt


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def _logV_btfr(logM: np.ndarray, a0: float = _A0_DEFAULT) -> np.ndarray:
    """Predict log10(V_flat / km s^-1) from the Baryonic Tully-Fisher Relation.

    M_bar = V_flat^4 / (G * a0)  →  V_flat = (G * M_bar * a0)^0.25

    Parameters
    ----------
    logM : array_like
        log10(M_bar / M_sun).
    a0 : float
        Characteristic acceleration in m/s^2.

    Returns
    -------
    ndarray
        log10(V_flat / km s^-1).
    """
    logM = np.asarray(logM, dtype=float)
    # log10( (G * M_sun * a0)^0.25 / 1000 )
    # = 0.25 * (log10(G) + log10(M_sun) + log10(a0)) - log10(1000)
    intercept = 0.25 * (
        np.log10(_G_SI) + np.log10(_MSUN_KG) + np.log10(a0)
    ) - np.log10(_KMS_TO_MS)
    return 0.25 * logM + intercept


def _logV_scm(
    logM: np.ndarray,
    log_gbar: np.ndarray,
    a0: float = _A0_DEFAULT,
) -> np.ndarray:
    """Predict log10(V_flat / km s^-1) using the Motor de Velos SCM model.

    The SCM applies the standard RAR interpolation function
    (McGaugh, Lelli & Schombert 2016):

        g_obs = g_bar / (1 - exp(-sqrt(g_bar / a0)))

    In the deep-MOND limit this reduces to g_obs = sqrt(g_bar * a0).

    The characteristic outer radius is estimated from
    R_eff = sqrt(G * M_bar / g_bar), which follows from assuming
    V_bar^2 = G * M_bar / R and g_bar = V_bar^2 / R simultaneously.

    V_SCM = sqrt(g_obs * R_eff)

    Parameters
    ----------
    logM : array_like
        log10(M_bar / M_sun).
    log_gbar : array_like
        log10(g_bar / m s^-2).
    a0 : float
        Characteristic acceleration in m/s^2.

    Returns
    -------
    ndarray
        log10(V_flat / km s^-1).
    """
    logM = np.asarray(logM, dtype=float)
    log_gbar = np.asarray(log_gbar, dtype=float)

    M_kg = 10.0 ** logM * _MSUN_KG
    g_bar = 10.0 ** log_gbar

    # RAR interpolation function (McGaugh+2016)
    x = g_bar / a0
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, _EPSILON))))
    g_obs = g_bar * nu

    # Effective outer radius (m)
    R_eff = np.sqrt(_G_SI * M_kg / np.maximum(g_bar, _EPSILON))

    V_scm_ms = np.sqrt(g_obs * R_eff)
    V_scm_kms = V_scm_ms / _KMS_TO_MS
    return np.log10(np.maximum(V_scm_kms, _EPSILON))


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run_blind_test(
    input_path: str | Path,
    out_dir: str | Path,
    a0: float = _A0_DEFAULT,
) -> pd.DataFrame:
    """Execute the blind test and write output files.

    Parameters
    ----------
    input_path : str or Path
        Path to the input CSV (little_things_global.csv format).
    out_dir : str or Path
        Directory where predictions.csv and summary.csv are written.
    a0 : float
        Characteristic SCM acceleration (m/s^2).

    Returns
    -------
    pd.DataFrame
        Summary statistics as a single-row DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {"galaxy_id", "logM", "logVobs", "log_gbar"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df.dropna(subset=["logM", "logVobs", "log_gbar"]).reset_index(drop=True)

    logV_btfr = _logV_btfr(df["logM"].values, a0=a0)
    logV_scm = _logV_scm(df["logM"].values, df["log_gbar"].values, a0=a0)

    res_scm = df["logVobs"].values - logV_scm
    res_btfr = df["logVobs"].values - logV_btfr

    rmse_scm = float(np.sqrt(np.mean(res_scm ** 2)))
    rmse_btfr = float(np.sqrt(np.mean(res_btfr ** 2)))
    improvement_pct = 100.0 * (rmse_btfr - rmse_scm) / rmse_btfr if rmse_btfr > 0 else 0.0

    # Wilcoxon signed-rank test on |error_BTFR| - |error_SCM|
    # Positive values indicate SCM has smaller absolute error
    delta = np.abs(res_btfr) - np.abs(res_scm)
    try:
        _, pval = scipy_wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        pval = float("nan")

    n = len(df)

    # --- Terminal output ---
    print(f"Galaxias: {n}")
    print(f"RMSE SCM: {rmse_scm:.4f}")
    print(f"RMSE BTFR: {rmse_btfr:.4f}")
    print(f"Fracción mejora: {improvement_pct:.1f}%")
    print(f"Wilcoxon p: {pval:.4f}" if np.isfinite(pval) else "Wilcoxon p: nan")

    # --- Write predictions.csv ---
    predictions = pd.DataFrame({
        "galaxy_id": df["galaxy_id"],
        "logM": df["logM"],
        "logVobs": df["logVobs"],
        "log_gbar": df["log_gbar"],
        "logV_SCM": logV_scm,
        "logV_BTFR": logV_btfr,
        "res_SCM": res_scm,
        "res_BTFR": res_btfr,
    })
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    # --- Write summary.csv ---
    summary = pd.DataFrame([{
        "N": n,
        "RMSE_SCM": rmse_scm,
        "RMSE_BTFR": rmse_btfr,
        "frac_mejora_pct": improvement_pct,
        "wilcoxon_p": pval,
    }])
    summary.to_csv(out_dir / "summary.csv", index=False)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Blind test: Motor de Velos SCM vs BTFR on LITTLE THINGS"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to little_things_global.csv",
    )
    parser.add_argument(
        "--outdir",
        default="resultados_lt",
        help="Output directory (default: resultados_lt)",
    )
    parser.add_argument(
        "--a0",
        type=float,
        default=_A0_DEFAULT,
        help=f"Characteristic SCM acceleration m/s^2 (default: {_A0_DEFAULT})",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_blind_test(
        input_path=args.input,
        out_dir=args.outdir,
        a0=args.a0,
    )


if __name__ == "__main__":
    main()
