"""
scripts/scm_mechanism_analysis.py — SCM Mechanism Analysis Pipeline (H1/H2/H3)

Tests three physical mechanisms that may explain the environmental signal in
galaxy rotation-curve tails within the SCM (Motor de Velos) framework:

  H1 — Dark-matter fraction effect:
       High env_proxy → higher DM fraction (f_DM_out) → flatter tail slope.
       Proxy: Pearson/Spearman ρ(f_DM_out, slope_tail); ΔAIC from OLS+H1.

  H2 — Disk-size effect:
       Compact disks (small Rdisk/Rmax) sample deeper potential → steeper tail.
       Proxy: Spearman ρ(Rdisk_Rmax, slope_tail).

  H3 — Baryonic mass effect:
       Residual logMbar signal after controlling for env_proxy and f_DM_out.
       Proxy: partial correlation; coefficient in the full model.

Extended regression:
    slope_tail ~ env_proxy + logMbar + f_DM_out + Rdisk_Rmax

with HC3 heteroskedasticity-robust standard errors.  Four nested models are
compared via ΔAIC relative to the base SCM model.

Public API
----------
load_catalog(catalog_path)                          → pd.DataFrame
compute_fdm_from_rotmods(sparc_dir, r_fraction)     → pd.DataFrame
build_dataset(catalog, fdm)                         → pd.DataFrame
run_correlations(df)                                → list[dict]
run_regressions(df)                                 → list[dict]
model_comparison_table(regression_results)          → pd.DataFrame
plot_h1_diagnostic(df, reg_results, out_dir)        → Path
plot_env_proxy_robustness(df, n_perm, seed)         → dict
main(argv=None)                                     → dict

Usage
-----
Default paths::

    python scripts/scm_mechanism_analysis.py

Custom paths::

    python scripts/scm_mechanism_analysis.py \\
        --catalog data/galaxy_catalog_with_env.csv \\
        --sparc-dir data/SPARC \\
        --out results/mechanism \\
        --n-perm 1000

"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_DEFAULT = "data/galaxy_catalog_with_env.csv"
SPARC_DIR_DEFAULT = "data/SPARC"
OUT_DIR_DEFAULT = "results/mechanism"

OUTER_FRAC_DEFAULT = 0.7
MIN_OUTER_POINTS = 2
N_PERM_DEFAULT = 1000
RANDOM_SEED_DEFAULT = 42

# Column names expected in the catalog
CATALOG_REQUIRED = {"galaxy", "logMbar", "env_proxy", "slope_tail"}
CATALOG_OPTIONAL = {"Rdisk", "Rmax"}

# SPARC rotmod.dat column layout (whitespace-separated, no header)
# Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
_ROTMOD_COLS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# 1. load_catalog
# ---------------------------------------------------------------------------

def load_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Load the galaxy catalog for the mechanism analysis.

    Parameters
    ----------
    catalog_path : str or Path
        CSV with at minimum columns: ``galaxy``, ``logMbar``, ``env_proxy``,
        ``slope_tail``.  ``logM`` is accepted as an alias for ``logMbar`` and
        renamed automatically.  ``Rdisk`` and ``Rmax`` are optional; when
        present they are used to derive ``Rdisk_Rmax``.

    Returns
    -------
    pd.DataFrame
        Catalog with standardised column names and a ``Rdisk_Rmax`` column
        added when both ``Rdisk`` and ``Rmax`` are present and non-zero.

    Raises
    ------
    FileNotFoundError
        If ``catalog_path`` does not exist.
    ValueError
        If required columns are missing after the ``logM``→``logMbar`` rename.
    """
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    df = pd.read_csv(path)

    # Normalise logM → logMbar
    if "logM" in df.columns and "logMbar" not in df.columns:
        df = df.rename(columns={"logM": "logMbar"})

    missing = CATALOG_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Catalog is missing required columns: {sorted(missing)}.  "
            f"Found: {sorted(df.columns.tolist())}"
        )

    # Derived column: Rdisk / Rmax
    if "Rdisk" in df.columns and "Rmax" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rdisk_rmax = np.where(
                df["Rmax"].gt(0),
                df["Rdisk"] / df["Rmax"],
                np.nan,
            )
        df = df.copy()
        df["Rdisk_Rmax"] = rdisk_rmax

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. compute_fdm_from_rotmods
# ---------------------------------------------------------------------------

def _parse_rotmod(path: Path) -> pd.DataFrame | None:
    """Read one SPARC ``_rotmod.dat`` file.

    The file has no header; columns are:
    Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul

    Returns ``None`` if the file cannot be parsed.
    """
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=_ROTMOD_COLS,
            dtype=float,
        )
        df = df[df["Rad"] > 0].reset_index(drop=True)
        return df if not df.empty else None
    except Exception:
        return None


def _fdm_from_rotmod(rc: pd.DataFrame, r_fraction: float) -> dict | None:
    """Compute outer-tail f_DM for a single galaxy rotation curve.

    Parameters
    ----------
    rc : pd.DataFrame
        Rotation curve with columns ``Rad``, ``Vobs``, ``Vgas``, ``Vdisk``,
        ``Vbul``.
    r_fraction : float
        Fraction of Rmax above which points are considered "outer tail".

    Returns
    -------
    dict or None
        Keys: ``r_max_kpc``, ``n_outer_points``, ``f_DM_out``,
        ``v_bar_out``, ``v_obs_out``.  Returns ``None`` when there are
        fewer than :data:`MIN_OUTER_POINTS` outer points or the baryonic
        velocity is non-positive.
    """
    r_max = rc["Rad"].max()
    outer = rc[rc["Rad"] > r_fraction * r_max]

    if len(outer) < MIN_OUTER_POINTS:
        return None

    # Baryonic velocity: quadrature sum of gas + disk + bulge components
    v_bar2 = (
        np.sign(outer["Vgas"]) * outer["Vgas"] ** 2
        + np.sign(outer["Vdisk"]) * outer["Vdisk"] ** 2
        + np.sign(outer["Vbul"]) * outer["Vbul"] ** 2
    )
    v_bar_mean = float(np.sqrt(np.abs(v_bar2.mean())))
    v_obs_mean = float(outer["Vobs"].mean())

    if v_obs_mean <= 0 or v_bar_mean < 0:
        return None

    # f_DM = (V_obs² − V_bar²) / V_obs²  (outer mean)
    f_DM_out = float(1.0 - (v_bar_mean ** 2 / v_obs_mean ** 2))

    return {
        "r_max_kpc": float(r_max),
        "n_outer_points": int(len(outer)),
        "f_DM_out": f_DM_out,
        "v_bar_out": v_bar_mean,
        "v_obs_out": v_obs_mean,
    }


def compute_fdm_from_rotmods(
    sparc_dir: str | Path,
    r_fraction: float = OUTER_FRAC_DEFAULT,
) -> pd.DataFrame:
    """Compute outer-tail dark-matter fraction for each SPARC galaxy.

    Searches ``sparc_dir`` (and ``sparc_dir/raw/``) for files matching the
    pattern ``*_rotmod.dat``.

    Parameters
    ----------
    sparc_dir : str or Path
        Root SPARC directory containing ``*_rotmod.dat`` files.
    r_fraction : float
        Fraction of Rmax defining the outer tail (default 0.7).

    Returns
    -------
    pd.DataFrame
        One row per galaxy with columns:
        ``galaxy``, ``r_max_kpc``, ``n_outer_points``, ``f_DM_out``,
        ``v_bar_out``, ``v_obs_out``.
        Empty DataFrame (with those columns) if no rotmod files are found.
    """
    sparc_dir = Path(sparc_dir)
    candidates: list[Path] = []
    for search_dir in [sparc_dir, sparc_dir / "raw"]:
        if search_dir.is_dir():
            candidates.extend(search_dir.glob("*_rotmod.dat"))

    results = []
    for fp in sorted(candidates):
        galaxy = fp.stem.replace("_rotmod", "")
        rc = _parse_rotmod(fp)
        if rc is None:
            continue
        stats = _fdm_from_rotmod(rc, r_fraction)
        if stats is None:
            continue
        results.append({"galaxy": galaxy, **stats})

    cols = ["galaxy", "r_max_kpc", "n_outer_points", "f_DM_out", "v_bar_out", "v_obs_out"]
    if not results:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(results)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. build_dataset
# ---------------------------------------------------------------------------

def build_dataset(
    catalog: pd.DataFrame,
    fdm: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the galaxy catalog with the f_DM table.

    Parameters
    ----------
    catalog : pd.DataFrame
        Output of :func:`load_catalog`.  Must contain ``galaxy``,
        ``logMbar``, ``env_proxy``, ``slope_tail``.
    fdm : pd.DataFrame
        Output of :func:`compute_fdm_from_rotmods`.  Must contain
        ``galaxy`` and ``f_DM_out``.  May be empty.

    Returns
    -------
    pd.DataFrame
        Inner-joined table with all columns from both inputs.  When
        ``fdm`` is empty, the catalog columns are returned with
        ``f_DM_out`` set to ``NaN``.
    """
    if fdm.empty:
        df = catalog.copy()
        df["f_DM_out"] = np.nan
        return df

    merged = catalog.merge(fdm[["galaxy", "f_DM_out"]], on="galaxy", how="left")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. run_correlations
# ---------------------------------------------------------------------------

def run_correlations(df: pd.DataFrame) -> list[dict]:
    """Compute Spearman correlations for the four H1/H2/H3 variable pairs.

    Pairs computed (when columns are available):

    1. ``env_proxy`` × ``slope_tail``  (base SCM signal)
    2. ``logMbar``   × ``f_DM_out``    (H1: mass → DM fraction)
    3. ``f_DM_out``  × ``slope_tail``  (H1: DM fraction → slope)
    4. ``Rdisk_Rmax``× ``slope_tail``  (H2: disk size → slope)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset from :func:`build_dataset`.

    Returns
    -------
    list of dict
        Each dict has keys: ``pair``, ``x``, ``y``, ``n``,
        ``rho``, ``p_value``, ``significant`` (p < 0.05).
    """
    pairs = [
        ("env_proxy", "slope_tail"),
        ("logMbar", "f_DM_out"),
        ("f_DM_out", "slope_tail"),
        ("Rdisk_Rmax", "slope_tail"),
    ]

    results = []
    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            results.append({
                "pair": f"{x_col} × {y_col}",
                "x": x_col, "y": y_col,
                "n": 0, "rho": np.nan, "p_value": np.nan,
                "significant": False,
            })
            continue

        sub = df[[x_col, y_col]].dropna()
        n = len(sub)
        if n < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(sub[x_col], sub[y_col])

        results.append({
            "pair": f"{x_col} × {y_col}",
            "x": x_col,
            "y": y_col,
            "n": n,
            "rho": float(rho) if not np.isnan(rho) else np.nan,
            "p_value": float(pval) if not np.isnan(pval) else np.nan,
            "significant": bool(not np.isnan(pval) and pval < 0.05),
        })

    return results


# ---------------------------------------------------------------------------
# 5. run_regressions
# ---------------------------------------------------------------------------

def _ols_hc3(
    y: pd.Series,
    X_df: pd.DataFrame,
    model_name: str,
) -> dict:
    """Fit OLS with HC3 robust standard errors.

    Parameters
    ----------
    y : pd.Series
        Dependent variable (``slope_tail``).
    X_df : pd.DataFrame
        Independent variables (will receive a constant column).
    model_name : str
        Label for this model.

    Returns
    -------
    dict
        Keys: ``model``, ``n``, ``k``, ``r2``, ``r2_adj``, ``aic``,
        ``coef`` (dict col→coef), ``pval`` (dict col→p), ``se`` (dict col→se).
    """
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels is required for regression analysis.")

    sub = pd.concat([y, X_df], axis=1).dropna()
    if len(sub) < X_df.shape[1] + 2:
        return {
            "model": model_name, "n": len(sub), "k": X_df.shape[1],
            "r2": np.nan, "r2_adj": np.nan, "aic": np.nan,
            "coef": {}, "pval": {}, "se": {},
        }

    y_fit = sub.iloc[:, 0]
    X_fit = sm.add_constant(sub.iloc[:, 1:])

    fit = sm.OLS(y_fit, X_fit).fit()
    hc3 = fit.get_robustcov_results(cov_type="HC3")

    return {
        "model": model_name,
        "n": int(len(sub)),
        "k": int(X_df.shape[1]),
        "r2": float(fit.rsquared),
        "r2_adj": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "coef": dict(zip(hc3.model.exog_names, hc3.params.tolist())),
        "pval": dict(zip(hc3.model.exog_names, hc3.pvalues.tolist())),
        "se": dict(zip(hc3.model.exog_names, hc3.bse.tolist())),
    }


def run_regressions(df: pd.DataFrame) -> list[dict]:
    """Fit four nested OLS HC3 models.

    Models
    ------
    M0  Base SCM:      slope_tail ~ env_proxy
    M1  +H1:           slope_tail ~ env_proxy + f_DM_out
    M2  +H2/H3:        slope_tail ~ env_proxy + logMbar
    M3  Full:          slope_tail ~ env_proxy + logMbar + f_DM_out [+ Rdisk_Rmax]

    ``Rdisk_Rmax`` is included in M3 only when present in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset from :func:`build_dataset`.

    Returns
    -------
    list of dict
        One dict per model from :func:`_ols_hc3`.
    """
    y = df["slope_tail"]
    results = []

    # M0 — Base SCM
    results.append(_ols_hc3(y, df[["env_proxy"]], "M0_base"))

    # M1 — +H1 (f_DM_out)
    if "f_DM_out" in df.columns:
        results.append(_ols_hc3(y, df[["env_proxy", "f_DM_out"]], "M1_H1"))
    else:
        results.append({"model": "M1_H1", "n": 0, "k": 2,
                        "r2": np.nan, "r2_adj": np.nan, "aic": np.nan,
                        "coef": {}, "pval": {}, "se": {}})

    # M2 — +H2/H3 (logMbar)
    results.append(_ols_hc3(y, df[["env_proxy", "logMbar"]], "M2_H2H3"))

    # M3 — Full model
    m3_cols = ["env_proxy", "logMbar"]
    if "f_DM_out" in df.columns:
        m3_cols.append("f_DM_out")
    if "Rdisk_Rmax" in df.columns:
        m3_cols.append("Rdisk_Rmax")
    results.append(_ols_hc3(y, df[m3_cols], "M3_full"))

    return results


# ---------------------------------------------------------------------------
# 6. model_comparison_table
# ---------------------------------------------------------------------------

def model_comparison_table(regression_results: list[dict]) -> pd.DataFrame:
    """Build a ΔAIC comparison table.

    Parameters
    ----------
    regression_results : list of dict
        Output of :func:`run_regressions`.

    Returns
    -------
    pd.DataFrame
        Columns: ``model``, ``n``, ``k``, ``r2``, ``r2_adj``, ``aic``,
        ``delta_aic``, ``winner``.
        ``delta_aic`` is relative to the model with the lowest AIC.
        ``winner`` is ``True`` only for the model with ΔAIC = 0.
    """
    rows = [
        {
            "model": r["model"],
            "n": r["n"],
            "k": r["k"],
            "r2": r["r2"],
            "r2_adj": r["r2_adj"],
            "aic": r["aic"],
        }
        for r in regression_results
    ]
    df = pd.DataFrame(rows)

    valid = df["aic"].notna()
    if valid.any():
        min_aic = df.loc[valid, "aic"].min()
        df["delta_aic"] = df["aic"] - min_aic
        df["winner"] = df["delta_aic"] == 0.0
    else:
        df["delta_aic"] = np.nan
        df["winner"] = False

    return df


# ---------------------------------------------------------------------------
# 7. plot_h1_diagnostic
# ---------------------------------------------------------------------------

def plot_h1_diagnostic(
    df: pd.DataFrame,
    reg_results: list[dict],
    out_dir: str | Path,
) -> Path:
    """Generate the 3-panel H1 diagnostic figure.

    Panel 1 — f_DM_out vs logMbar (scatter + regression line).
    Panel 2 — f_DM_out vs slope_tail (scatter + regression line).
    Panel 3 — ΔAIC per model (bar chart).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset from :func:`build_dataset`.
    reg_results : list of dict
        Output of :func:`run_regressions`.
    out_dir : str or Path
        Directory where ``h1_diagnostic.png`` and ``h1_diagnostic.pdf``
        are written.

    Returns
    -------
    Path
        Path to the PNG file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct = model_comparison_table(reg_results)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("SCM Mechanism Analysis — H1 Diagnostic", fontsize=11)

    # Panel 1: f_DM_out vs logMbar
    ax1 = axes[0]
    if "f_DM_out" in df.columns and "logMbar" in df.columns:
        sub = df[["logMbar", "f_DM_out"]].dropna()
        ax1.scatter(sub["logMbar"], sub["f_DM_out"],
                    alpha=0.7, s=30, color="steelblue", label=f"N={len(sub)}")
        if len(sub) >= 2:
            z = np.polyfit(sub["logMbar"], sub["f_DM_out"], 1)
            x_line = np.linspace(sub["logMbar"].min(), sub["logMbar"].max(), 100)
            ax1.plot(x_line, np.polyval(z, x_line), "r--", lw=1.5)
        ax1.set_xlabel("logMbar")
        ax1.set_ylabel("f_DM_out")
        ax1.set_title("H1: Mass → DM fraction")
        ax1.legend(fontsize=8)
    else:
        ax1.text(0.5, 0.5, "f_DM_out not available",
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("H1: Mass → DM fraction")

    # Panel 2: f_DM_out vs slope_tail
    ax2 = axes[1]
    if "f_DM_out" in df.columns and "slope_tail" in df.columns:
        sub = df[["f_DM_out", "slope_tail"]].dropna()
        ax2.scatter(sub["f_DM_out"], sub["slope_tail"],
                    alpha=0.7, s=30, color="darkorange", label=f"N={len(sub)}")
        if len(sub) >= 2:
            z = np.polyfit(sub["f_DM_out"], sub["slope_tail"], 1)
            x_line = np.linspace(sub["f_DM_out"].min(), sub["f_DM_out"].max(), 100)
            ax2.plot(x_line, np.polyval(z, x_line), "r--", lw=1.5)
        ax2.set_xlabel("f_DM_out")
        ax2.set_ylabel("slope_tail")
        ax2.set_title("H1: DM fraction → slope")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "f_DM_out / slope_tail not available",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("H1: DM fraction → slope")

    # Panel 3: ΔAIC bar chart
    ax3 = axes[2]
    valid_ct = ct[ct["delta_aic"].notna()]
    if not valid_ct.empty:
        colors = ["gold" if w else "lightcoral" for w in valid_ct["winner"]]
        ax3.bar(valid_ct["model"], valid_ct["delta_aic"], color=colors, edgecolor="k")
        ax3.axhline(2, color="gray", linestyle="--", lw=1, label="ΔAIC = 2")
        ax3.set_xlabel("Model")
        ax3.set_ylabel("ΔAIC")
        ax3.set_title("Model comparison")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No AIC data", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_title("Model comparison")

    plt.tight_layout()
    png_path = out_dir / "h1_diagnostic.png"
    pdf_path = out_dir / "h1_diagnostic.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path


# ---------------------------------------------------------------------------
# 8. plot_env_proxy_robustness
# ---------------------------------------------------------------------------

def plot_env_proxy_robustness(
    df: pd.DataFrame,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = RANDOM_SEED_DEFAULT,
    out_dir: str | Path | None = None,
) -> dict:
    """Permutation test: env_proxy → slope_tail within mass bins.

    Shuffles ``env_proxy`` within tertile mass bins (mass-stratified
    permutation) to preserve the marginal mass distribution.  The observed
    Spearman ρ is compared to the null distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset from :func:`build_dataset`.
    n_perm : int
        Number of permutations (default 1000).
    seed : int
        Random seed.
    out_dir : str, Path, or None
        If provided, saves ``env_robustness.png`` and ``env_robustness.pdf``.

    Returns
    -------
    dict
        Keys: ``rho_obs``, ``p_perm``, ``n_perm``, ``n_galaxies``,
        ``rho_null_mean``, ``rho_null_std``.
    """
    sub = df[["env_proxy", "slope_tail", "logMbar"]].dropna().copy()
    n = len(sub)

    if n < 3:
        return {
            "rho_obs": np.nan, "p_perm": np.nan,
            "n_perm": n_perm, "n_galaxies": n,
            "rho_null_mean": np.nan, "rho_null_std": np.nan,
        }

    rho_obs, _ = spearmanr(sub["env_proxy"], sub["slope_tail"])

    # Assign tertile bins by logMbar for stratified permutation
    sub["_mass_bin"] = pd.qcut(sub["logMbar"], q=3, labels=False, duplicates="drop")

    rng = np.random.default_rng(seed)
    rho_null = np.empty(n_perm)

    # Pre-build bin → positional-index mapping for fast access
    bins_map: dict[int, np.ndarray] = {}
    for bin_id in sub["_mass_bin"].unique():
        mask = (sub["_mass_bin"] == bin_id).values
        bins_map[int(bin_id)] = np.where(mask)[0]

    env_arr = np.array(sub["env_proxy"].values, dtype=float)
    slope_arr = sub["slope_tail"].values

    for i in range(n_perm):
        perm_env = env_arr.copy()
        for pos_idx in bins_map.values():
            perm_env[pos_idx] = rng.permuted(env_arr[pos_idx])
        rho_null[i], _ = spearmanr(perm_env, slope_arr)

    p_perm = float(np.mean(np.abs(rho_null) >= abs(rho_obs)))

    result = {
        "rho_obs": float(rho_obs),
        "p_perm": p_perm,
        "n_perm": n_perm,
        "n_galaxies": n,
        "rho_null_mean": float(rho_null.mean()),
        "rho_null_std": float(rho_null.std()),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(rho_null, bins=40, color="lightgray", edgecolor="k",
                label=f"Null distribution (N={n_perm})")
        ax.axvline(rho_obs, color="crimson", lw=2,
                   label=f"Observed ρ = {rho_obs:.3f}")
        ax.axvline(-abs(rho_obs), color="crimson", lw=2, linestyle="--")
        ax.set_xlabel("Spearman ρ (env_proxy × slope_tail)")
        ax.set_ylabel("Count")
        ax.set_title(f"Stratified permutation test  p = {p_perm:.3f}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / "env_robustness.png", dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / "env_robustness.pdf", bbox_inches="tight")
        plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCM Mechanism Analysis Pipeline — H1/H2/H3 tests."
    )
    parser.add_argument(
        "--catalog", default=CATALOG_DEFAULT, metavar="CSV",
        help=f"Galaxy catalog CSV with logMbar, env_proxy, slope_tail "
             f"(default: {CATALOG_DEFAULT}).",
    )
    parser.add_argument(
        "--sparc-dir", default=SPARC_DIR_DEFAULT, metavar="DIR",
        dest="sparc_dir",
        help=f"Root SPARC directory with *_rotmod.dat files "
             f"(default: {SPARC_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default=OUT_DIR_DEFAULT, metavar="DIR",
        help=f"Output directory for CSVs and figures (default: {OUT_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--n-perm", type=int, default=N_PERM_DEFAULT, dest="n_perm",
        help=f"Permutations for the robustness test (default: {N_PERM_DEFAULT}).",
    )
    parser.add_argument(
        "--r-fraction", type=float, default=OUTER_FRAC_DEFAULT, dest="r_fraction",
        help=f"Outer-tail radius fraction for f_DM (default: {OUTER_FRAC_DEFAULT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the full mechanism analysis pipeline.

    Returns
    -------
    dict with keys:
        n_catalog        — galaxies in the loaded catalog
        n_fdm            — galaxies with f_DM computed
        n_dataset        — galaxies in the merged dataset
        correlations     — list of Spearman correlation dicts
        regressions      — list of OLS model dicts
        model_table      — pd.DataFrame from model_comparison_table
        permutation      — dict from plot_env_proxy_robustness
    """
    args = _parse_args(argv)
    out_dir = Path(args.out)

    # 1 — Load catalog
    catalog = load_catalog(args.catalog)
    print(f"\nGalaxias en catálogo:    {len(catalog)}")

    # 2 — f_DM from rotmods (optional; gracefully absent)
    fdm = compute_fdm_from_rotmods(args.sparc_dir, r_fraction=args.r_fraction)
    print(f"Galaxias con f_DM:       {len(fdm)}")

    # 3 — Merge
    ds = build_dataset(catalog, fdm)
    print(f"Galaxias en dataset:     {len(ds)}")

    # 4 — Correlations
    corr = run_correlations(ds)
    print("\nCorrelaciones Spearman:")
    for c in corr:
        sig = "*" if c["significant"] else " "
        print(f"  {sig} {c['pair']:30s}  ρ={c['rho']:+.3f}  p={c['p_value']:.4f}  N={c['n']}")

    # 5 — Regressions
    if not _HAS_STATSMODELS:
        print("\n[WARN] statsmodels not installed — skipping regressions.")
        reg = []
        ct = pd.DataFrame()
    else:
        reg = run_regressions(ds)
        ct = model_comparison_table(reg)
        print("\nComparación de modelos (ΔAIC):")
        print(ct[["model", "n", "r2", "aic", "delta_aic", "winner"]].to_string(index=False))

    # 6 — Figures
    if not ds.empty:
        plot_h1_diagnostic(ds, reg, out_dir)
        perm = plot_env_proxy_robustness(ds, n_perm=args.n_perm, out_dir=out_dir)
    else:
        perm = {"rho_obs": np.nan, "p_perm": np.nan, "n_perm": args.n_perm,
                "n_galaxies": 0, "rho_null_mean": np.nan, "rho_null_std": np.nan}

    print(f"\nTest de permutación (N={args.n_perm}):")
    print(f"  ρ observado = {perm['rho_obs']:.3f},  p_perm = {perm['p_perm']:.4f}")

    # 7 — Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_csv(out_dir / "dataset.csv", index=False)
    pd.DataFrame(corr).to_csv(out_dir / "correlations.csv", index=False)
    if reg:
        ct.to_csv(out_dir / "model_comparison.csv", index=False)
    pd.DataFrame([perm]).to_csv(out_dir / "permutation_test.csv", index=False)

    print(f"\n  Results written to {out_dir}")

    return {
        "n_catalog": len(catalog),
        "n_fdm": len(fdm),
        "n_dataset": len(ds),
        "correlations": corr,
        "regressions": reg,
        "model_table": ct,
        "permutation": perm,
    }


if __name__ == "__main__":
    main()
