"""
lt_dust_hinge_analysis.py

Bridge module for SCM-Motor de Velos:
- Compute a hinge proxy F3 from LITTLE THINGS-like rotation-curve CSVs
- Merge with dust temperature (T_dust) + control variables (logM, logZ)
- Run regression + matched-pairs test

Inputs expected (CSV):
1) Rotation curves: data/raw/lt_oh2015/{GAL}_rot.csv with columns:
   - r_kpc, Vbary_kms
   Optional: Vobs_kms (not used here)
2) Dust table: data/raw/cigan2021_tdust.csv columns: galaxy, T_dust
3) Mass table: data/raw/lt_masses.csv columns: galaxy, logM
4) Metallicity table: data/raw/lt_metals.csv columns: galaxy, logZ

Outputs:
- results/littlethings_dust/lt_dust_hinge.csv
- results/littlethings_dust/lt_pairs.csv
- results/littlethings_dust/fig_f3_vs_tdust.png
- results/littlethings_dust/run_log.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.stats import wilcoxon


# -----------------------------
# Parameters (SCM-like hinge)
# -----------------------------
@dataclass(frozen=True)
class HingeParams:
    d: float = 0.062
    logg0: float = -10.45
    g_floor: float = 1e-15  # m/s^2 numerical floor
    ext_frac: float = 0.7   # outer region threshold as fraction of Rmax
    ext_last_k: int = 3     # fallback: last K points if ext region empty


KPC_TO_M = 3.085677581e19
KMS_TO_MS = 1_000.0


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns: {missing}")


# -----------------------------
# Load rotation curve
# -----------------------------
def load_lt_rotation_curve(
    galaxy: str,
    data_dir: str | Path = "data/raw/lt_oh2015",
) -> pd.DataFrame:
    path = Path(data_dir) / f"{galaxy}_rot.csv"
    if not path.exists():
        raise FileNotFoundError(f"Rotation curve file not found: {path}")

    df = pd.read_csv(path)
    _ensure_columns(df, ["r_kpc"], where=str(path))

    # accept either Vbary or Vbary_kms
    if "Vbary_kms" not in df.columns and "Vbary" in df.columns:
        df = df.rename(columns={"Vbary": "Vbary_kms"})
    _ensure_columns(df, ["Vbary_kms"], where=str(path))

    df = df[["r_kpc", "Vbary_kms"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("r_kpc").reset_index(drop=True)

    if len(df) < 3:
        raise ValueError(f"{path}: too few points ({len(df)})")

    return df


# -----------------------------
# Compute F3 hinge proxy
# -----------------------------
def compute_F3_from_rc(
    df_rc: pd.DataFrame,
    params: HingeParams = HingeParams(),
) -> float:
    """
    F3 = mean hinge term in external region (r > ext_frac * Rmax).
    hinge = d * max(0, logg0 - log10(g_bar)), g_bar = Vbary^2 / r (SI)

    Notes:
    - uses baryonic acceleration proxy only (consistent if you interpret
      hinge as SCM bar-term trigger)
    """
    r_kpc = df_rc["r_kpc"].to_numpy(dtype=float)
    v_kms = df_rc["Vbary_kms"].to_numpy(dtype=float)

    r_m = r_kpc * KPC_TO_M
    v_ms = v_kms * KMS_TO_MS

    g_bar = (v_ms ** 2) / np.maximum(r_m, 1.0)  # avoid zero division
    g_bar = np.maximum(g_bar, params.g_floor)
    log_gbar = np.log10(g_bar)

    hinge = params.d * np.maximum(0.0, params.logg0 - log_gbar)

    rmax = float(np.nanmax(r_kpc))
    mask_ext = r_kpc > params.ext_frac * rmax

    if int(mask_ext.sum()) == 0:
        # fallback: last K points
        k = min(params.ext_last_k, len(r_kpc))
        mask_ext = np.zeros_like(r_kpc, dtype=bool)
        mask_ext[-k:] = True

    return float(np.nanmean(hinge[mask_ext]))


# -----------------------------
# Build master table
# -----------------------------
def _load_table(path: str | Path, key: str = "galaxy") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing table: {p}")
    df = pd.read_csv(p)
    if key not in df.columns:
        raise ValueError(f"{p}: missing '{key}' column")
    return df


def build_master_table(
    galaxies: list[str],
    f3_by_galaxy: dict[str, float],
    dust_file: str | Path = "data/raw/cigan2021_tdust.csv",
    mass_file: str | Path = "data/raw/lt_masses.csv",
    metal_file: str | Path = "data/raw/lt_metals.csv",
) -> pd.DataFrame:
    dust = _load_table(dust_file)
    mass = _load_table(mass_file)
    metal = _load_table(metal_file)

    for df, col in [(dust, "T_dust"), (mass, "logM"), (metal, "logZ")]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in table")

    # index by galaxy for safe lookup
    dust = dust.set_index("galaxy")
    mass = mass.set_index("galaxy")
    metal = metal.set_index("galaxy")

    rows = []
    for g in galaxies:
        row: dict = {"galaxy": g, "F3_SCM": f3_by_galaxy.get(g, np.nan)}
        row["T_dust"] = dust.loc[g, "T_dust"] if g in dust.index else np.nan
        row["logM"] = mass.loc[g, "logM"] if g in mass.index else np.nan
        row["logZ"] = metal.loc[g, "logZ"] if g in metal.index else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Regression + matched pairs
# -----------------------------
def regress_tdust(
    df: pd.DataFrame,
) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    df2 = df.dropna(subset=["T_dust", "logM", "logZ", "F3_SCM"]).copy()
    if df2.empty:
        return None

    X = sm.add_constant(df2[["logM", "logZ", "F3_SCM"]])
    y = df2["T_dust"]
    return sm.OLS(y, X).fit()


def matched_pairs(
    df: pd.DataFrame,
    tol_mass: float = 0.2,
    tol_metal: float = 0.1,
) -> pd.DataFrame:
    df2 = df.dropna(subset=["T_dust", "logM", "logZ", "F3_SCM"]).copy().reset_index(drop=True)
    pairs = []
    used: set[int] = set()

    for i in range(len(df2)):
        if i in used:
            continue

        row = df2.loc[i]
        candidates = df2.loc[
            (df2.index != i)
            & (~df2.index.isin(list(used)))
            & (np.abs(df2["logM"] - row["logM"]) < tol_mass)
            & (np.abs(df2["logZ"] - row["logZ"]) < tol_metal)
        ]

        if len(candidates) == 0:
            continue

        # choose candidate with maximal |delta_F3_SCM| to maximize contrast
        j = int((candidates["F3_SCM"] - row["F3_SCM"]).abs().idxmax())
        row2 = df2.loc[j]

        pairs.append({
            "gal1": row["galaxy"],
            "gal2": row2["galaxy"],
            "delta_T": float(row["T_dust"] - row2["T_dust"]),
            "delta_F3_SCM": float(row["F3_SCM"] - row2["F3_SCM"]),
        })

        used.add(i)
        used.add(j)

    return pd.DataFrame(pairs)


def wilcoxon_test(pairs_df: pd.DataFrame) -> Tuple[Optional[float], Optional[int]]:
    if pairs_df.empty:
        return None, None
    positive = pairs_df[pairs_df["delta_F3_SCM"] > 0.0]
    if len(positive) < 3:
        return None, int(len(positive))
    stat, p = wilcoxon(positive["delta_T"], alternative="greater")
    return float(p), int(len(positive))


# -----------------------------
# Main runner
# -----------------------------
def main():
    out_dir = Path("results/littlethings_dust")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    # publishable results go here (tracked in git)
    pub_dir = Path("results/lt_dust_hinge")
    pub_dir.mkdir(parents=True, exist_ok=True)

    # smoke test list (replace with N=26 later)
    galaxies = ["DDO210", "DDO69", "DDO75", "DDO70"]

    f3: dict[str, float] = {}
    lines = []
    lines.append("LT dust-hinge bridge run\n")

    for g in galaxies:
        try:
            rc = load_lt_rotation_curve(g)
            f3[g] = compute_F3_from_rc(rc)
            lines.append(f"{g}: F3_SCM={f3[g]:.6f}\n")
        except Exception as e:
            f3[g] = np.nan
            lines.append(f"{g}: ERROR {repr(e)}\n")

    df = build_master_table(galaxies, f3)
    df_out = out_dir / "lt_dust_hinge.csv"
    df.to_csv(df_out, index=False)
    # publishable results table
    df.to_csv(pub_dir / "lt_hinge_dust_results.csv", index=False)

    model = regress_tdust(df)
    reg_lines = []
    if model is not None:
        lines.append("\n=== OLS: T_dust ~ logM + logZ + F3_SCM ===\n")
        lines.append(model.summary().as_text() + "\n")
        if len(df.dropna()) < 8:
            lines.append("\nWARNING: N<8 -> do not interpret p-values as evidence.\n")
        # build publishable regression summary
        reg_lines.append("OLS: T_dust ~ logM + logZ + F3_SCM\n")
        reg_lines.append(f"N = {int(model.nobs)}\n")
        reg_lines.append(f"R-squared = {model.rsquared:.4f}\n\n")
        reg_lines.append(model.summary().as_text() + "\n")
        if model.nobs < 8:
            reg_lines.append("\nWARNING: N<8 -> do not interpret p-values as evidence.\n")
    else:
        reg_lines.append("No regression results (insufficient data).\n")
    (pub_dir / "lt_regression_summary.txt").write_text("".join(reg_lines), encoding="utf-8")

    pairs = matched_pairs(df)
    pairs_out = out_dir / "lt_pairs.csv"
    pairs.to_csv(pairs_out, index=False)

    pval, npos = wilcoxon_test(pairs)
    lines.append("\n=== Matched pairs ===\n")
    lines.append(f"pairs_total={len(pairs)}\n")
    lines.append(f"pairs_deltaF3_pos={npos}\n")
    lines.append(f"wilcoxon_p={pval}\n")

    # publishable matched-pairs summary
    mp_lines = []
    mp_lines.append("Matched-pairs analysis: LITTLE THINGS hinge vs dust temperature\n\n")
    mp_lines.append(f"n_pairs = {len(pairs)}\n")
    if not pairs.empty:
        median_delta_T = float(np.median(pairs["delta_T"]))
        median_delta_F3_SCM = float(np.median(pairs["delta_F3_SCM"]))
        mp_lines.append(f"median_delta_T = {median_delta_T:.4f}\n")
        mp_lines.append(f"median_delta_F3_SCM = {median_delta_F3_SCM:.6f}\n")
    mp_lines.append(f"pairs_deltaF3_pos = {npos}\n")
    mp_lines.append(f"wilcoxon_p = {pval}\n")
    if npos is not None and npos < 3:
        mp_lines.append("\nWARNING: N<3 positive pairs -> Wilcoxon p not computed.\n")
    if not pairs.empty:
        mp_lines.append("\nPairs table:\n")
        mp_lines.append(pairs.to_string(index=False) + "\n")
    (pub_dir / "lt_matched_pairs.txt").write_text("".join(mp_lines), encoding="utf-8")

    # Plot
    fig_path = out_dir / "fig_f3_vs_tdust.png"
    plt.figure()
    plt.scatter(df["F3_SCM"], df["T_dust"])
    plt.xlabel("F3_SCM (hinge proxy)")
    plt.ylabel("T_dust (K)")
    plt.title("LITTLE THINGS: F3_SCM vs T_dust")
    for _, r in df.iterrows():
        if pd.notna(r["F3_SCM"]) and pd.notna(r["T_dust"]):
            plt.annotate(r["galaxy"], (r["F3_SCM"], r["T_dust"]), fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    log_path.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] wrote: {df_out}")
    print(f"[OK] wrote: {pairs_out}")
    print(f"[OK] wrote: {fig_path}")
    print(f"[OK] wrote: {log_path}")
    print(f"[OK] wrote: {pub_dir / 'lt_hinge_dust_results.csv'}")
    print(f"[OK] wrote: {pub_dir / 'lt_regression_summary.txt'}")
    print(f"[OK] wrote: {pub_dir / 'lt_matched_pairs.txt'}")


if __name__ == "__main__":
    main()
