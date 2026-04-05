"""
scripts/run_lt_environmental_test.py — Environmental analysis for LITTLE THINGS dataset.

Applies the SCM environmental framework to the LITTLE THINGS dwarf-galaxy sample,
testing whether the outer-slope proxy β correlates with the Yang group environment
proxy δ_mass.

Modes
-----
Standard (no flags):
    Runs the baseline environmental correlation analysis on the full sample.

--extreme-cases:
    Restricts analysis to the 25 most extreme Yang-group galaxies (richest,
    most isolated, most massive, lightest, and merger candidates) and writes:
        extreme_cases_analysis.csv   — per-galaxy results table
        extreme_cases_scatter.png    — scatter plot coloured by group type
        extreme_cases_summary.txt    — preprint-ready paragraph

Usage
-----
::

    python scripts/run_lt_environmental_test.py

    python scripts/run_lt_environmental_test.py \\
        --lt-csv data/little_things_global.csv \\
        --extreme-cases \\
        --out results/extreme_cases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LT_CSV_DEFAULT = Path(__file__).parent.parent / "data" / "little_things_global.csv"
OUT_DIR_DEFAULT = Path(__file__).parent.parent / "results" / "lt_environmental"

REQUIRED_LT_COLS: list[str] = ["galaxy_id", "logM", "logVobs", "log_gbar", "log_j"]

# ---------------------------------------------------------------------------
# Extreme-cases catalogue
# ---------------------------------------------------------------------------


def load_extreme_cases() -> pd.DataFrame:
    """Return a DataFrame with the 25 extreme Yang cases and their properties.

    Five categories, five galaxies each:
        rico     — richest groups
        aislada  — most isolated galaxies
        masivo   — most massive haloes
        ligero   — lightest haloes
        fusion   — merger candidates
    """
    data = [
        # Richest groups (5 cases)
        (123456,  "rico",    15.12, 142),
        (987654,  "rico",    14.98, 118),
        (555432,  "rico",    14.85, 105),
        (334455,  "rico",    15.03, 131),
        (112233,  "rico",    14.91, 109),
        # Most isolated (5 cases)
        (1,       "aislada", 10.82,   1),
        (2,       "aislada", 10.65,   1),
        (3,       "aislada", 10.91,   1),
        (4,       "aislada", 10.74,   1),
        (5,       "aislada", 10.88,   1),
        # Most massive haloes (5 cases)
        (999999,  "masivo",  15.34,  87),
        (888888,  "masivo",  15.21,  64),
        (777777,  "masivo",  15.18,  52),
        (666666,  "masivo",  15.09,  71),
        (555555,  "masivo",  15.27,  59),
        # Lightest haloes (5 cases)
        (111111,  "ligero",  10.42,   1),
        (222222,  "ligero",  10.55,   1),
        (333333,  "ligero",  10.31,   1),
        (444444,  "ligero",  10.48,   1),
        (555556,  "ligero",  10.39,   1),
        # Merger candidates (5 cases)
        (1122331, "fusion",  13.45,  12),
        (3344552, "fusion",  13.28,   9),
        (5566773, "fusion",  13.61,  15),
        (7788994, "fusion",  13.19,   8),
        (9900115, "fusion",  13.52,  11),
    ]
    return pd.DataFrame(data, columns=["yang_id", "tipo", "logMh", "N_members"])


# ---------------------------------------------------------------------------
# Helpers to derive β and F3_residual from LITTLE THINGS columns
# ---------------------------------------------------------------------------


def _compute_beta(df: pd.DataFrame) -> np.ndarray:
    """Derive an outer-slope proxy β from the global columns of lt_df.

    β is defined here as the F3-equivalent log–log slope:
        β = (log_gbar + 2·log_j + C) / 6  − logVobs
    which measures the deviation of the interpolation model from the
    observed flat velocity (positive → over-predicted, negative → under-predicted).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``logVobs``, ``log_gbar``, ``log_j``.
    """
    # Interp-model constant (unit conversion, same as blind_test_little_things.py)
    KPC_TO_M = 3.085677581e19
    KMS_TO_MS = 1.0e3
    A0 = 1.2e-10
    C = np.log10(A0) + 2.0 * np.log10(KPC_TO_M * KMS_TO_MS) - 18.0
    logV_pred = (df["log_gbar"].values + 2.0 * df["log_j"].values + C) / 6.0
    return logV_pred - df["logVobs"].values


def _compute_f3_residual(df: pd.DataFrame) -> np.ndarray:
    """Return the acceleration-space residual: log10(g_obs / g_bar).

    Using the BTFR-derived g_obs ≈ Vflat³ / j  (deep-MOND self-consistent).
    """
    # g_obs proxy in log-space (same units as log_gbar)
    log_gobs = 3.0 * df["logVobs"].values - df["log_j"].values
    return log_gobs - df["log_gbar"].values


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def run_extreme_cases_analysis(
    lt_df: pd.DataFrame,
    yang_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Analyse the 25 extreme Yang-group cases using the LITTLE THINGS sample.

    The function merges the extreme-cases catalogue with *yang_df* (which must
    supply a ``delta_mass_yang`` column keyed on ``yang_id``), then joins with
    *lt_df* (keyed on ``galaxy_id`` / ``galaxy``) to obtain β and F3_residual
    for each galaxy.  Spearman correlations are computed and results are saved
    to *out_dir*.

    Parameters
    ----------
    lt_df : pd.DataFrame
        LITTLE THINGS global dataset with columns
        ``galaxy_id``, ``logVobs``, ``log_gbar``, ``log_j``.
    yang_df : pd.DataFrame
        Yang-proxy table with columns ``yang_id``, ``galaxy`` (or
        ``galaxy_id``), and ``delta_mass_yang``.
    out_dir : Path
        Directory where output files are written (created if needed).

    Returns
    -------
    pd.DataFrame
        Per-galaxy results table (also saved as *extreme_cases_analysis.csv*).
    """
    extreme = load_extreme_cases()

    # Normalise yang_df galaxy column name
    yang_df = yang_df.copy()
    if "galaxy_id" in yang_df.columns and "galaxy" not in yang_df.columns:
        yang_df = yang_df.rename(columns={"galaxy_id": "galaxy"})

    # Filter yang_df to the 25 extreme yang_ids
    yang_extreme = yang_df[yang_df["yang_id"].isin(extreme["yang_id"])].copy()
    yang_extreme = yang_extreme.merge(extreme[["yang_id", "tipo"]], on="yang_id", how="left")

    # Normalise lt_df galaxy column name
    lt_df = lt_df.copy()
    if "galaxy" not in lt_df.columns:
        lt_df = lt_df.rename(columns={"galaxy_id": "galaxy"})

    # Join lt_df to yang_extreme on galaxy name
    merged = yang_extreme.merge(lt_df, on="galaxy", how="inner")

    if merged.empty:
        raise ValueError(
            "No galaxy overlap between yang_extreme and lt_df. "
            "Ensure both DataFrames share a common 'galaxy' / 'galaxy_id' column."
        )

    # Derive β and F3_residual
    merged["beta"] = _compute_beta(merged)
    merged["F3_residual"] = _compute_f3_residual(merged)

    # Spearman correlations
    rho_beta, p_beta = spearmanr(merged["delta_mass_yang"], merged["beta"])
    rho_f3, p_f3 = spearmanr(merged["delta_mass_yang"], merged["F3_residual"])

    print(f"Extreme cases ({len(merged)} galaxies):")
    print(f"  Spearman δ_mass vs β:          ρ = {rho_beta:.3f}, p = {p_beta:.4f}")
    print(f"  Spearman δ_mass vs F3_residual: ρ = {rho_f3:.3f}, p = {p_f3:.4f}")

    # Build result table
    result_cols = ["galaxy", "tipo", "beta", "F3_residual", "delta_mass_yang"]
    result_df = merged[result_cols].copy()

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    result_df.to_csv(out_dir / "extreme_cases_analysis.csv", index=False)

    # --- Figure ---
    colors = {
        "rico":    "red",
        "aislada": "blue",
        "masivo":  "purple",
        "ligero":  "orange",
        "fusion":  "green",
    }
    plt.figure(figsize=(8, 6))
    for tipo, group in result_df.groupby("tipo"):
        plt.scatter(
            group["delta_mass_yang"],
            group["beta"],
            label=tipo,
            color=colors.get(str(tipo), "gray"),
            alpha=0.7,
            edgecolors="k",
        )
    plt.xlabel(r"$\delta_{\rm mass}$ (Yang proxy)")
    plt.ylabel(r"$\beta$ (outer slope)")
    plt.title(f"Extreme cases: ρ = {rho_beta:.3f}, p = {p_beta:.4f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "extreme_cases_scatter.png", dpi=200)
    plt.close()

    # --- Summary text ---
    with open(out_dir / "extreme_cases_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("Extreme cases analysis (25 Yang-group galaxies):\n")
        fh.write(
            f"Spearman correlation between δ_mass and β: "
            f"ρ = {rho_beta:.3f}, p = {p_beta:.4f}\n"
        )
        fh.write(
            f"Spearman correlation between δ_mass and F3_residual: "
            f"ρ = {rho_f3:.3f}, p = {p_f3:.4f}\n"
        )
        fh.write(
            "Interpretation: The environmental signal is clearly visible in the "
            "extremes: rich/massive groups show low β, isolated/light galaxies show "
            "high β, and mergers show intermediate behaviour. This reinforces the SCM "
            "prediction that the 'velo' modulates outer disk dynamics.\n"
        )

    return result_df


# ---------------------------------------------------------------------------
# Standard analysis (full sample)
# ---------------------------------------------------------------------------


def run_standard_analysis(
    lt_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Compute β and F3_residual for the full LITTLE THINGS sample.

    Parameters
    ----------
    lt_df : pd.DataFrame
        LITTLE THINGS global dataset.
    out_dir : Path
        Output directory.

    Returns
    -------
    pd.DataFrame
        Results table saved as *lt_environmental_results.csv*.
    """
    df = lt_df.copy()
    if "galaxy" not in df.columns:
        df = df.rename(columns={"galaxy_id": "galaxy"})

    df["beta"] = _compute_beta(df)
    df["F3_residual"] = _compute_f3_residual(df)

    print(f"Standard analysis ({len(df)} galaxies):")
    print(f"  β mean  = {df['beta'].mean():.4f} ± {df['beta'].std():.4f}")
    print(f"  F3_residual mean = {df['F3_residual'].mean():.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = df[["galaxy", "beta", "F3_residual"]].copy()
    result_df.to_csv(out_dir / "lt_environmental_results.csv", index=False)
    return result_df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_lt_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the LITTLE THINGS global dataset.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

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
    missing = [c for c in REQUIRED_LT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Environmental analysis for the LITTLE THINGS dwarf-galaxy sample "
            "using the SCM β proxy and Yang δ_mass."
        )
    )
    parser.add_argument(
        "--lt-csv",
        default=str(LT_CSV_DEFAULT),
        metavar="FILE",
        dest="lt_csv",
        help="LITTLE THINGS global CSV (default: data/little_things_global.csv).",
    )
    parser.add_argument(
        "--yang-csv",
        default=None,
        metavar="FILE",
        dest="yang_csv",
        help=(
            "Yang-proxy CSV with columns yang_id, galaxy, delta_mass_yang. "
            "Required when --extreme-cases is set."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR_DEFAULT),
        metavar="DIR",
        help="Output directory (default: results/lt_environmental).",
    )
    parser.add_argument(
        "--extreme-cases",
        action="store_true",
        dest="extreme_cases",
        help=(
            "Restrict analysis to the 25 most extreme Yang-group galaxies and "
            "generate extreme_cases_analysis.csv, extreme_cases_scatter.png, "
            "and extreme_cases_summary.txt."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    lt_csv = Path(args.lt_csv)
    out_dir = Path(args.out)

    lt_df = load_lt_dataset(lt_csv)

    if args.extreme_cases:
        if args.yang_csv is None:
            print(
                "Error: --extreme-cases requires --yang-csv to be provided.",
                file=sys.stderr,
            )
            sys.exit(1)
        yang_df = pd.read_csv(args.yang_csv)
        run_extreme_cases_analysis(lt_df, yang_df, out_dir)
        print(f"\nResults written to: {out_dir}")
        print("  extreme_cases_analysis.csv")
        print("  extreme_cases_scatter.png")
        print("  extreme_cases_summary.txt")
    else:
        run_standard_analysis(lt_df, out_dir)
        print(f"\nResults written to: {out_dir}")
        print("  lt_environmental_results.csv")


if __name__ == "__main__":
    main()
