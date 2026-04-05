"""
scripts/analyze_sparc_chae.py — Improved SPARC × Chae environmental analysis.

Merges SPARC (sparc_table1.csv) with Chae environment data (chae_env.csv)
using normalised galaxy names, then tests whether the environment proxy
``e_env`` adds explanatory power beyond baryonic mass for the chosen
kinematics observable.

Analysis steps
--------------
1. Normalise galaxy names and merge the two tables.
2. Auto-detect the best dependent variable: F3 > beta > delta_f3 > DeltaF3 > Vflat.
3. Auto-detect the best mass proxy: logM > logMbar > … > L[3.6].
4. OLS base model:  y ~ log_mass_proxy              (HC3)
5. Spearman ρ: residuals vs e_env.
6. OLS residual model: residuals ~ e_env             (HC3)
7. OLS full model:  y ~ log_mass_proxy + e_env       (HC3)
8. ΔAIC, ΔBIC, ΔR², ΔR²_adj.
9. Permutation test (shuffle e_env).
10. Save merged CSV for inspection.

Usage
-----
::

    python scripts/analyze_sparc_chae.py \\
        --sparc sparc_table1.csv \\
        --chae  chae_env.csv \\
        --out   env_analysis_merged.csv \\
        --n-perms 2000 \\
        --seed 42

All arguments are optional; defaults match the CONFIG block in the original
Colab notebook.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

# =========================
# DEFAULT CONFIG
# =========================
_SPARC_PATH_DEFAULT = "sparc_table1.csv"
_CHAE_PATH_DEFAULT = "chae_env.csv"
_OUT_PATH_DEFAULT = "env_analysis_merged.csv"
_N_PERM_DEFAULT = 2000
_RANDOM_SEED_DEFAULT = 42
_MIN_SAMPLE = 15

# =========================
# HELPERS
# =========================

def clean_name(x: object) -> str | float:
    """Normalise a galaxy name for robust cross-table merging.

    Strips leading/trailing whitespace, converts to upper case, removes
    hyphens, and collapses all internal whitespace so that, e.g.,
    ``'DDO 154'``, ``'DDO154'``, ``'ddo154'`` all map to ``'DDO154'``.

    Parameters
    ----------
    x : str or NaN
        Raw galaxy name.

    Returns
    -------
    str
        Cleaned name, or ``np.nan`` if the input is NaN/None.
    """
    if pd.isna(x):
        return np.nan  # type: ignore[return-value]
    x = str(x).strip().upper()
    x = x.replace("-", "")
    x = re.sub(r"\s+", "", x)
    return x


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from *candidates* that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_mass_column(df: pd.DataFrame) -> str:
    """Add a ``log_mass_proxy`` column to *df* using the best available source.

    Priority order
    --------------
    1. Direct log-mass columns: ``logM``, ``logMbar``, ``logM_baryon``, …
    2. Linear mass columns (→ log10): ``Mbar``, ``Mb``, ``Mstar``, …
    3. 3.6 μm luminosity columns (→ log10): ``L[3.6]``, ``L36``, …

    Parameters
    ----------
    df : pd.DataFrame
        Merged galaxy table (modified in-place).

    Returns
    -------
    str
        Human-readable description of the column used.

    Raises
    ------
    ValueError
        If no usable mass or luminosity column is found.
    """
    log_candidates = [
        "logM", "logMbar", "logM_baryon", "logMb", "logMstar", "logM_star",
        "logM_b", "logMtot",
    ]
    lin_mass_candidates = ["Mbar", "Mb", "M_baryon", "Mstar", "M_star", "Mtot"]
    lum_candidates = ["L[3.6]", "L36", "L3.6", "L_3.6", "Lum36", "L36_Lsun"]

    c = find_first_existing(df, log_candidates)
    if c is not None:
        df["log_mass_proxy"] = pd.to_numeric(df[c], errors="coerce")
        return f"log mass directa: {c}"

    c = find_first_existing(df, lin_mass_candidates)
    if c is not None:
        vals = pd.to_numeric(df[c], errors="coerce")
        vals = vals.where(vals > 0)
        df["log_mass_proxy"] = np.log10(vals)
        return f"log10(masa lineal): {c}"

    c = find_first_existing(df, lum_candidates)
    if c is not None:
        vals = pd.to_numeric(df[c], errors="coerce")
        vals = vals.where(vals > 0)
        df["log_mass_proxy"] = np.log10(vals)
        return f"log10(luminosidad 3.6μm): {c}"

    raise ValueError(
        "No se encontró ninguna columna de masa/luminosidad utilizable. "
        "Añade una de estas: logM, logMbar, Mbar, Mstar, L[3.6], L36, …"
    )


def choose_target_column(df: pd.DataFrame) -> str:
    """Return the best available kinematics observable column.

    Priority: ``F3`` > ``beta`` > ``delta_f3`` > ``DeltaF3`` > ``Vflat``.

    Raises
    ------
    ValueError
        If none of the candidate columns exist in *df*.
    """
    candidates = ["F3", "beta", "delta_f3", "DeltaF3", "Vflat"]
    c = find_first_existing(df, candidates)
    if c is None:
        raise ValueError(
            "No se encontró variable dependiente. Añade una de estas columnas: "
            "F3, beta, delta_f3, DeltaF3, Vflat"
        )
    return c


def permutation_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 2000,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Two-sided permutation test for Spearman ρ.

    Parameters
    ----------
    x, y : array-like
        Paired observations (NaN-free).
    n_perm : int
        Number of permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    rho_obs : float
        Observed Spearman ρ.
    p_two_sided : float
        Permutation p-value (two-sided).
    perm_rhos : np.ndarray
        Array of permuted ρ values.
    """
    rng = np.random.default_rng(seed)
    rho_obs, _ = spearmanr(x, y, nan_policy="omit")
    y_arr = np.asarray(y, dtype=float)
    perm_rhos = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = rng.permutation(y_arr)
        perm_rhos[i], _ = spearmanr(x, y_perm, nan_policy="omit")
    p_two_sided = float(np.mean(np.abs(perm_rhos) >= np.abs(rho_obs)))
    return float(rho_obs), p_two_sided, perm_rhos


# =========================
# CORE ANALYSIS
# =========================

def run_analysis(
    sparc: pd.DataFrame,
    chae: pd.DataFrame,
    n_perms: int = _N_PERM_DEFAULT,
    seed: int = _RANDOM_SEED_DEFAULT,
    verbose: bool = True,
) -> dict:
    """Full SPARC × Chae environmental analysis.

    Parameters
    ----------
    sparc : pd.DataFrame
        SPARC global table.  Must contain column ``Galaxy``.
    chae : pd.DataFrame
        Chae environment table.  Must contain columns ``Galaxy`` and ``e_env``.
    n_perms : int
        Permutation iterations.
    seed : int
        Random seed.
    verbose : bool
        Print progress / diagnostic messages.

    Returns
    -------
    dict with keys:
        ``df``          — merged DataFrame used for the analysis
        ``target_col``  — name of the dependent variable column
        ``mass_desc``   — description of the mass proxy used
        ``model_mass``  — fitted OLS base model (statsmodels)
        ``model_resid`` — fitted OLS residuals-vs-env model
        ``model_full``  — fitted OLS full model
        ``rho``         — Spearman ρ (residuals vs e_env)
        ``p_spear``     — Spearman p-value
        ``p_perm``      — permutation p-value
        ``delta_aic``   — AIC(base) − AIC(full)
        ``delta_bic``   — BIC(base) − BIC(full)
        ``delta_r2``    — R²(full) − R²(base)
        ``delta_adj_r2``— adjusted-R²(full) − adjusted-R²(base)
        ``match_diag``  — dict with match diagnostics
    """
    # ── Validate required columns ────────────────────────────────────────────
    for col in ("Galaxy",):
        if col not in sparc.columns:
            raise ValueError(f"sparc table must contain column '{col}'")
        if col not in chae.columns:
            raise ValueError(f"chae table must contain column '{col}'")
    if "e_env" not in chae.columns:
        raise ValueError("chae table must contain column 'e_env'")

    # ── Name normalisation ───────────────────────────────────────────────────
    sparc = sparc.copy()
    chae = chae.copy()
    sparc["Galaxy_clean"] = sparc["Galaxy"].apply(clean_name)
    chae["Galaxy_clean"] = chae["Galaxy"].apply(clean_name)

    sparc_names = set(sparc["Galaxy_clean"].dropna())
    chae_names = set(chae["Galaxy_clean"].dropna())
    intersection = sparc_names & chae_names

    match_diag = {
        "n_sparc": len(sparc_names),
        "n_chae": len(chae_names),
        "n_intersection": len(intersection),
        "missing_in_chae": sorted(sparc_names - chae_names)[:20],
        "missing_in_sparc": sorted(chae_names - sparc_names)[:20],
    }

    if verbose:
        sep = "=" * 70
        print(sep)
        print("DIAGNÓSTICO DE MATCH")
        print(sep)
        print(f"SPARC galaxias únicas: {match_diag['n_sparc']}")
        print(f"CHAE galaxias únicas:  {match_diag['n_chae']}")
        print(f"Intersección:          {match_diag['n_intersection']}")
        print("\nEjemplos en SPARC que no aparecen en CHAE (hasta 20):")
        print(match_diag["missing_in_chae"])
        print("\nEjemplos en CHAE que no aparecen en SPARC (hasta 20):")
        print(match_diag["missing_in_sparc"])

    # ── Merge ────────────────────────────────────────────────────────────────
    chae_dedup = chae.drop_duplicates(subset=["Galaxy_clean"]).copy()
    chae_cols = ["Galaxy_clean", "Galaxy", "e_env"]
    if "e_env_err" in chae.columns:
        chae_cols.append("e_env_err")

    df = pd.merge(
        sparc,
        chae_dedup[chae_cols],
        on="Galaxy_clean",
        how="inner",
        suffixes=("_sparc", "_chae"),
    ).copy()

    if verbose:
        print(f"\nMuestra combinada tras merge: {len(df)}")

    # ── Auto-detect columns ──────────────────────────────────────────────────
    target_col = choose_target_column(df)
    mass_desc = build_mass_column(df)

    if verbose:
        print("\nVARIABLES ELEGIDAS")
        print(f"Objetivo: {target_col}")
        print(f"Masa:     {mass_desc}")

    # ── Coerce to numeric and drop NaN ───────────────────────────────────────
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df["e_env"] = pd.to_numeric(df["e_env"], errors="coerce")
    df["log_mass_proxy"] = pd.to_numeric(df["log_mass_proxy"], errors="coerce")

    df = df.dropna(subset=[target_col, "e_env", "log_mass_proxy"]).copy()

    if verbose:
        print(f"\nMuestra final usable: {len(df)} galaxias")

    if len(df) < _MIN_SAMPLE:
        raise ValueError(
            f"La muestra final tiene sólo {len(df)} galaxias "
            f"(mínimo requerido: {_MIN_SAMPLE})."
        )

    # ── MODEL 1: y ~ mass ────────────────────────────────────────────────────
    y = df[target_col].values
    X_base = sm.add_constant(df[["log_mass_proxy"]], has_constant="add")
    model_mass = sm.OLS(y, X_base).fit(cov_type="HC3")
    df["resid_mass"] = model_mass.resid

    # ── SPEARMAN: resid ~ e_env ──────────────────────────────────────────────
    rho, p_spear = spearmanr(df["resid_mass"], df["e_env"], nan_policy="omit")

    # ── MODEL 2: resid ~ e_env ───────────────────────────────────────────────
    X_env = sm.add_constant(df[["e_env"]], has_constant="add")
    model_resid = sm.OLS(df["resid_mass"].values, X_env).fit(cov_type="HC3")

    # ── MODEL 3: y ~ mass + e_env ────────────────────────────────────────────
    X_full = sm.add_constant(df[["log_mass_proxy", "e_env"]], has_constant="add")
    model_full = sm.OLS(y, X_full).fit(cov_type="HC3")

    # ── PERMUTATION TEST ─────────────────────────────────────────────────────
    rho_perm, p_perm, _ = permutation_spearman(
        df["resid_mass"].values,
        df["e_env"].values,
        n_perm=n_perms,
        seed=seed,
    )

    # ── ΔAIC / ΔBIC / ΔR² ───────────────────────────────────────────────────
    delta_aic = model_mass.aic - model_full.aic
    delta_bic = model_mass.bic - model_full.bic
    delta_r2 = model_full.rsquared - model_mass.rsquared
    delta_adj_r2 = model_full.rsquared_adj - model_mass.rsquared_adj

    if verbose:
        sep = "=" * 70
        print(f"\n{sep}")
        print("MODELO BASE: y ~ masa")
        print(sep)
        print(model_mass.summary())

        print(f"\n{sep}")
        print("CORRELACIÓN RESIDUO vs e_env")
        print(sep)
        print(f"Spearman rho = {rho:.4f}")
        print(f"Spearman p   = {p_spear:.4e}")
        print(f"Perm p       = {p_perm:.4e}")

        print(f"\n{sep}")
        print("MODELO RESIDUAL: residuo ~ e_env")
        print(sep)
        print(model_resid.summary())

        print(f"\n{sep}")
        print("MODELO COMPLETO: y ~ masa + e_env")
        print(sep)
        print(model_full.summary())

        print(f"\n{sep}")
        print("COMPARACIÓN DE MODELOS")
        print(sep)
        print(f"ΔAIC      = {delta_aic:.3f}")
        print(f"ΔBIC      = {delta_bic:.3f}")
        print(f"ΔR²       = {delta_r2:.4f}")
        print(f"ΔR²_adj   = {delta_adj_r2:.4f}")

    return {
        "df": df,
        "target_col": target_col,
        "mass_desc": mass_desc,
        "model_mass": model_mass,
        "model_resid": model_resid,
        "model_full": model_full,
        "rho": float(rho),
        "p_spear": float(p_spear),
        "p_perm": float(p_perm),
        "delta_aic": float(delta_aic),
        "delta_bic": float(delta_bic),
        "delta_r2": float(delta_r2),
        "delta_adj_r2": float(delta_adj_r2),
        "match_diag": match_diag,
    }


# =========================
# CLI
# =========================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SPARC × Chae environmental analysis with galaxy-name normalisation."
    )
    p.add_argument("--sparc", default=_SPARC_PATH_DEFAULT,
                   help="Path to SPARC global table CSV (default: %(default)s)")
    p.add_argument("--chae", default=_CHAE_PATH_DEFAULT,
                   help="Path to Chae environment CSV (default: %(default)s)")
    p.add_argument("--out", default=_OUT_PATH_DEFAULT,
                   help="Output CSV path (default: %(default)s)")
    p.add_argument("--n-perms", type=int, default=_N_PERM_DEFAULT,
                   help="Permutation iterations (default: %(default)s)")
    p.add_argument("--seed", type=int, default=_RANDOM_SEED_DEFAULT,
                   help="Random seed (default: %(default)s)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    sparc_path = Path(args.sparc)
    chae_path = Path(args.chae)

    if not sparc_path.exists():
        print(f"ERROR: SPARC file not found: {sparc_path}", file=sys.stderr)
        sys.exit(1)
    if not chae_path.exists():
        print(f"ERROR: Chae file not found: {chae_path}", file=sys.stderr)
        sys.exit(1)

    sparc = pd.read_csv(sparc_path)
    chae = pd.read_csv(chae_path)

    results = run_analysis(sparc, chae, n_perms=args.n_perms, seed=args.seed, verbose=True)
    df = results["df"]
    target_col = results["target_col"]

    save_cols = [
        c for c in [
            "Galaxy_sparc", "Galaxy_chae", "Galaxy_clean",
            target_col, "log_mass_proxy", "e_env", "e_env_err", "resid_mass",
        ]
        if c in df.columns
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[save_cols].to_csv(out_path, index=False)
    print(f"\nCSV guardado: {out_path}")


if __name__ == "__main__":
    main()
