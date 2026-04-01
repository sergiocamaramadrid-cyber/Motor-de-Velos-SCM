#!/usr/bin/env python3
"""
Exploratory environment-interaction analysis for the SCM framework.

This script tests whether the environmental modulation of outer-disc dynamics
(F3) is itself environment-dependent, using exploratory stratified models of
the form:

    F3 ~ delta_mass
    F3 ~ delta_mass + C(env)
    F3 ~ delta_mass * C(env)

Important:
- This is an exploratory stratified analysis, not a confirmatory test.
- If env is derived from delta_mass itself, results must be interpreted with
  caution and should not be presented as an independent validation.

Outputs are written to:
    results/environment_interaction/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf


DEFAULT_INPUT = Path("results/universal_term_comparison_full.csv")
DEFAULT_OUTPUT = Path("results/environment_interaction")

# Threshold schemes:
# - percentile methods use quantiles of delta_mass
# - fixed methods use physical threshold values in delta_mass units
THRESH_METHODS: Dict[str, Tuple[str, float, float]] = {
    "percentile_33_67": ("percentile", 0.33, 0.67),
    "percentile_25_75": ("percentile", 0.25, 0.75),
    "fixed_0.5": ("fixed", -0.5, 0.5),
    "fixed_0.3": ("fixed", -0.3, 0.3),
}

BOOTSTRAP_ITER = 1000
BOOTSTRAP_SEED = 42


def detect_column(df: pd.DataFrame, candidates: list, label: str) -> str:
    """Return the first matching column name from a list of aliases."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise ValueError(
        f"Could not find a column for '{label}'. "
        f"Tried aliases: {candidates}. Available columns: {list(df.columns)}"
    )


def compute_aicc(model) -> float:
    """
    Compute AICc from a fitted statsmodels OLS result.

    AICc = AIC + 2k(k+1)/(n-k-1)
    where k includes the intercept.
    """
    n = int(model.nobs)
    k = int(model.df_model) + 1  # include intercept
    if n - k - 1 <= 0:
        return np.nan
    return model.aic + (2 * k * (k + 1)) / (n - k - 1)


def resolve_thresholds(
    delta: pd.Series,
    method_kind: str,
    low: float,
    high: float,
) -> Tuple[float, float]:
    """Resolve thresholds either from quantiles or fixed values."""
    if method_kind == "percentile":
        return float(delta.quantile(low)), float(delta.quantile(high))
    if method_kind == "fixed":
        return float(low), float(high)
    raise ValueError(f"Unknown threshold kind: {method_kind}")


def classify_env(delta: pd.Series, low_thresh: float, high_thresh: float) -> pd.Series:
    """
    Assign environment classes from delta_mass thresholds.

    void:         delta_mass < low_thresh
    intermediate: low_thresh <= delta_mass <= high_thresh
    filament:     delta_mass > high_thresh
    """
    env = pd.Series("intermediate", index=delta.index, dtype="object")
    env.loc[delta < low_thresh] = "void"
    env.loc[delta > high_thresh] = "filament"
    return env


def fit_models(df_reg: pd.DataFrame) -> dict:
    """Fit baseline, additive, and interaction models with HC3 robust errors."""
    m1 = smf.ols("F3 ~ delta_mass", data=df_reg).fit(cov_type="HC3")
    m2 = smf.ols("F3 ~ delta_mass + C(env)", data=df_reg).fit(cov_type="HC3")
    m3 = smf.ols("F3 ~ delta_mass * C(env)", data=df_reg).fit(cov_type="HC3")

    rows = []
    for name, mod in [("model1", m1), ("model2", m2), ("model3", m3)]:
        aicc_val = compute_aicc(mod)
        rows.append(
            {
                "model": name,
                "formula": mod.model.formula,
                "nobs": int(mod.nobs),
                "aic": float(mod.aic),
                "aicc": float(aicc_val) if not math.isnan(aicc_val) else np.nan,
                "bic": float(mod.bic),
                "rsquared": float(mod.rsquared),
                "rsquared_adj": float(mod.rsquared_adj),
            }
        )

    return {
        "model1": m1,
        "model2": m2,
        "model3": m3,
        "comparison": pd.DataFrame(rows),
    }


def bootstrap_interaction(df_reg: pd.DataFrame, n_iter: int, seed: int) -> pd.DataFrame:
    """Bootstrap the interaction coefficient from model 3."""
    rng = np.random.default_rng(seed)
    n = len(df_reg)
    rows = []

    for i in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        boot_df = df_reg.iloc[idx].copy()
        try:
            mod = smf.ols("F3 ~ delta_mass * C(env)", data=boot_df).fit(cov_type="HC3")
            coef = mod.params.get("C(env)[T.filament]:delta_mass", np.nan)
        except Exception:
            coef = np.nan
        rows.append({"iteration": i, "beta_int": coef})

    return pd.DataFrame(rows)


def fit_group_slope(sub: pd.DataFrame) -> dict:
    """Fit a simple slope for one group using linregress."""
    if len(sub) < 3:
        return {"slope": np.nan, "intercept": np.nan, "stderr": np.nan, "pvalue": np.nan, "n": len(sub)}
    res = stats.linregress(sub["delta_mass"], sub["F3"])
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "stderr": float(res.stderr),
        "pvalue": float(res.pvalue),
        "n": len(sub),
    }


def welch_test_slopes(group_a: dict, group_b: dict) -> Tuple[float, float]:
    """
    Approximate Welch-style test for difference between two independently estimated slopes.
    Returns (t_stat, p_value).
    """
    if any(np.isnan(group_a[k]) for k in ["slope", "stderr"]) or any(
        np.isnan(group_b[k]) for k in ["slope", "stderr"]
    ):
        return np.nan, np.nan
    if group_a["n"] <= 2 or group_b["n"] <= 2:
        return np.nan, np.nan

    se_diff = math.sqrt(group_a["stderr"] ** 2 + group_b["stderr"] ** 2)
    if se_diff == 0:
        return np.nan, np.nan

    t_stat = (group_a["slope"] - group_b["slope"]) / se_diff
    num = (group_a["stderr"] ** 2 + group_b["stderr"] ** 2) ** 2
    den = (
        (group_a["stderr"] ** 4) / max(group_a["n"] - 2, 1)
        + (group_b["stderr"] ** 4) / max(group_b["n"] - 2, 1)
    )
    dof = num / den if den > 0 else np.nan
    if np.isnan(dof):
        return np.nan, np.nan

    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
    return float(t_stat), float(p_val)


def threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Run the interaction model under all threshold definitions."""
    rows = []

    for method, (kind, low_raw, high_raw) in THRESH_METHODS.items():
        low, high = resolve_thresholds(df["delta_mass"], kind, low_raw, high_raw)
        env = classify_env(df["delta_mass"], low, high)
        sub = df.copy()
        sub["env"] = pd.Categorical(env, categories=["void", "intermediate", "filament"], ordered=True)

        df_reg = sub[sub["env"].isin(["void", "filament"])].copy()
        if len(df_reg) < 10 or df_reg["env"].nunique() < 2:
            continue

        mod = smf.ols("F3 ~ delta_mass * C(env)", data=df_reg).fit(cov_type="HC3")
        aicc_val = compute_aicc(mod)
        rows.append(
            {
                "method": method,
                "kind": kind,
                "low_thresh": low,
                "high_thresh": high,
                "n_void": int((df_reg["env"] == "void").sum()),
                "n_filament": int((df_reg["env"] == "filament").sum()),
                "interaction_coef": float(mod.params.get("C(env)[T.filament]:delta_mass", np.nan)),
                "interaction_pvalue": float(mod.pvalues.get("C(env)[T.filament]:delta_mass", np.nan)),
                "aic": float(mod.aic),
                "aicc": float(aicc_val) if not math.isnan(aicc_val) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def add_fit_line(ax, sub: pd.DataFrame, color: str) -> None:
    """Add a fitted line for one subgroup if enough points exist."""
    if len(sub) < 3:
        return
    mod = smf.ols("F3 ~ delta_mass", data=sub).fit(cov_type="HC3")
    x = np.linspace(sub["delta_mass"].min(), sub["delta_mass"].max(), 100)
    y = mod.predict(pd.DataFrame({"delta_mass": x}))
    ax.plot(x, y, color=color, linewidth=2)


def plot_figure(df: pd.DataFrame, outpath: Path) -> None:
    """
    Create a 2x2 figure:
    - all groups
    - void
    - intermediate
    - filament
    """
    colors = {"void": "tab:blue", "intermediate": "tab:gray", "filament": "tab:red"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    # Panel 1: all groups
    ax = axes[0]
    for group in ["void", "intermediate", "filament"]:
        sub = df[df["env"] == group]
        ax.scatter(sub["delta_mass"], sub["F3"], s=28, alpha=0.75, label=group, color=colors[group])
        if group in ("void", "filament"):
            add_fit_line(ax, sub, colors[group])
    ax.set_title("All environments")
    ax.set_xlabel(r"$\delta_{\mathrm{mass}}$")
    ax.set_ylabel(r"$F_3$")
    ax.legend(frameon=False)

    # Panels 2-4: per group
    for ax, group in zip(axes[1:], ["void", "intermediate", "filament"]):
        sub = df[df["env"] == group]
        ax.scatter(sub["delta_mass"], sub["F3"], s=28, alpha=0.8, color=colors[group])
        add_fit_line(ax, sub, colors[group])
        ax.set_title(group.capitalize())
        ax.set_xlabel(r"$\delta_{\mathrm{mass}}$")
        ax.set_ylabel(r"$F_3$")

    fig.suptitle("Exploratory environment interaction analysis", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def write_summary(
    outpath: Path,
    main_method: str,
    low_th: float,
    high_th: float,
    comparison: pd.DataFrame,
    interaction_coef: float,
    interaction_p: float,
    welch_t: float,
    welch_p: float,
) -> None:
    """Write a compact human-readable summary."""
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("Environment Interaction Analysis (exploratory)\n")
        f.write("================================================\n\n")
        f.write(f"Main threshold method: {main_method}\n")
        f.write(f"Resolved thresholds: low={low_th:.4f}, high={high_th:.4f}\n\n")
        f.write("Model comparison:\n")
        f.write(comparison.to_string(index=False))
        f.write("\n\n")
        f.write(f"Interaction coefficient (filament:delta_mass): {interaction_coef:.6f}\n")
        f.write(f"Interaction p-value: {interaction_p:.6g}\n\n")
        f.write("Welch-style slope-difference test (filament vs void):\n")
        f.write(f"t-statistic: {welch_t:.6f}\n")
        f.write(f"p-value: {welch_p:.6g}\n\n")
        f.write(
            "Interpretation note: this is an exploratory stratified analysis. "
            "If env is derived from delta_mass itself, results should not be "
            "treated as an independent confirmatory test.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory environment interaction analysis for SCM.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input CSV with galaxy, delta_mass, F3 (or aliases).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Output directory.",
    )
    parser.add_argument(
        "--main-method",
        default="fixed_0.5",
        choices=list(THRESH_METHODS.keys()),
        help="Threshold scheme used for the main figure and main summaries.",
    )
    parser.add_argument(
        "--bootstrap-iter",
        type=int,
        default=BOOTSTRAP_ITER,
        help="Bootstrap iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help="Bootstrap seed.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    galaxy_col = detect_column(df, ["galaxy", "name"], "galaxy")
    delta_col = detect_column(df, ["delta_mass", "delta", "env_delta"], "delta_mass")
    f3_col = detect_column(df, ["F3", "f3", "F3_SCM", "delta_f3"], "F3")

    df = df.rename(columns={galaxy_col: "galaxy", delta_col: "delta_mass", f3_col: "F3"})
    df = df[["galaxy", "delta_mass", "F3"]].dropna().copy()

    if len(df) < 10:
        raise ValueError("Not enough rows after cleaning to run the interaction analysis.")

    # Main threshold scheme
    kind, low_raw, high_raw = THRESH_METHODS[args.main_method]
    low_th, high_th = resolve_thresholds(df["delta_mass"], kind, low_raw, high_raw)
    df["env"] = classify_env(df["delta_mass"], low_th, high_th)
    df["env"] = pd.Categorical(df["env"], categories=["void", "intermediate", "filament"], ordered=True)

    df_reg = df[df["env"].isin(["void", "filament"])].copy()
    if len(df_reg) < 10 or df_reg["env"].nunique() < 2:
        raise ValueError("Main thresholding produced too few void/filament galaxies for regression.")

    models = fit_models(df_reg)
    comparison = models["comparison"]
    m3 = models["model3"]

    interaction_coef = float(m3.params.get("C(env)[T.filament]:delta_mass", np.nan))
    interaction_p = float(m3.pvalues.get("C(env)[T.filament]:delta_mass", np.nan))

    # Coefficients
    conf = m3.conf_int()
    coef_df = pd.DataFrame(
        {
            "coefficient": m3.params.index,
            "estimate": m3.params.values,
            "std_err": m3.bse.values,
            "p_value": m3.pvalues.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
        }
    )
    coef_df.to_csv(output_dir / "interaction_coefficients.csv", index=False)

    # Model comparison
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    # Bootstrap
    boot_df = bootstrap_interaction(df_reg, n_iter=args.bootstrap_iter, seed=args.seed)
    boot_df.to_csv(output_dir / "bootstrap_beta_int.csv", index=False)

    # Group slopes + Welch
    void_stats = fit_group_slope(df[df["env"] == "void"])
    filament_stats = fit_group_slope(df[df["env"] == "filament"])
    welch_t, welch_p = welch_test_slopes(filament_stats, void_stats)

    welch_rows = [
        {"group": "void", **void_stats},
        {"group": "filament", **filament_stats},
        {
            "group": "welch_diff_filament_minus_void",
            "slope": np.nan,
            "intercept": np.nan,
            "stderr": np.nan,
            "pvalue": welch_p,
            "n": np.nan,
        },
    ]
    pd.DataFrame(welch_rows).to_csv(output_dir / "welch_group_slopes.csv", index=False)

    # Sensitivity
    sens_df = threshold_sensitivity(df)
    sens_df.to_csv(output_dir / "threshold_sensitivity.csv", index=False)

    # Figure
    plot_figure(df, output_dir / "figure_env_interaction.pdf")

    # Text summary
    write_summary(
        output_dir / "interaction_model_summary.txt",
        args.main_method,
        low_th,
        high_th,
        comparison,
        interaction_coef,
        interaction_p,
        welch_t,
        welch_p,
    )

    print(f"All outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
