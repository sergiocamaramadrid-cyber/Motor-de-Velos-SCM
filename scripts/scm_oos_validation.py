"""
Out-of-sample (OOS) validation for SCM vs a baseline RAR model.

This script produces the branch-documented artifacts:
  - results/oos_validation/oos_generalization_results.csv
  - results/oos_validation/hist_delta_rmse_out.pdf
  - results/oos_validation/oos_terminal_log.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_A0_DEFAULT = 1.2e-10


def _nu_simple(x: np.ndarray) -> np.ndarray:
    safe = np.maximum(x, 1e-12)
    return 1.0 / (1.0 - np.exp(-np.sqrt(safe)))


def _split_out_indices(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0], dtype=int)
    idx = np.arange(n)
    out = idx[idx % 5 == 0]
    if out.size == 0:
        out = np.array([n - 1], dtype=int)
    return out


def run_oos_validation(
    comparison_csv: Path,
    out_dir: Path,
    a0: float = _A0_DEFAULT,
) -> tuple[int, float, float]:
    df = pd.read_csv(comparison_csv)
    needed = {"galaxy", "g_bar", "g_obs"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {comparison_csv}: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    for galaxy, g in df.groupby("galaxy", sort=True):
        gg = g.sort_values("r_kpc" if "r_kpc" in g.columns else "g_bar").reset_index(drop=True)
        out_idx = _split_out_indices(len(gg))
        g_out = gg.iloc[out_idx]
        gbar = g_out["g_bar"].to_numpy(dtype=float)
        gobs = g_out["g_obs"].to_numpy(dtype=float)

        x = gbar / a0
        gpred_baseline = _nu_simple(x) * gbar
        # SCM (Motor de Velos) baseline in acceleration space:
        # g_pred = g_bar + a0  (equivalent to V_total² = V_bar² + a0_kpc·r)
        gpred_scm = gbar + a0

        rmse_baseline = float(np.sqrt(np.mean((gobs - gpred_baseline) ** 2)))
        rmse_scm = float(np.sqrt(np.mean((gobs - gpred_scm) ** 2)))
        rows.append(
            {
                "galaxy": str(galaxy),
                "n_out": int(len(g_out)),
                "rmse_out_baseline": rmse_baseline,
                "rmse_out_scm": rmse_scm,
                "delta_rmse_out": rmse_scm - rmse_baseline,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows).sort_values("galaxy").reset_index(drop=True)
    out_df.to_csv(out_dir / "oos_generalization_results.csv", index=False)

    median_delta = float(out_df["delta_rmse_out"].median()) if len(out_df) else float("nan")
    p_value = float("nan")
    if len(out_df) >= 1:
        try:
            p_value = float(
                wilcoxon(
                    out_df["rmse_out_scm"].to_numpy(),
                    out_df["rmse_out_baseline"].to_numpy(),
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    mode="auto",
                ).pvalue
            )
        except ValueError:
            p_value = float("nan")

    plt.figure(figsize=(6.5, 4.0))
    plt.hist(out_df["delta_rmse_out"], bins=20)
    plt.axvline(
        median_delta,
        linestyle="--",
        color="tab:red",
        label="Median ΔRMSE_out = RMSE_SCM - RMSE_baseline",
    )
    plt.xlabel("ΔRMSE_out = RMSE_SCM - RMSE_baseline")
    plt.ylabel("Count")
    plt.title("OOS ΔRMSE distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hist_delta_rmse_out.pdf")
    plt.close()

    lines = [
        f"n_galaxies = {len(out_df)}",
        f"median ΔRMSE_out = {median_delta}",
        f"p-value Wilcoxon = {p_value}",
    ]
    (out_dir / "oos_terminal_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(out_df), median_delta, p_value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCM OOS validation.")
    parser.add_argument(
        "--comparison-csv",
        default="results/universal_term_comparison_full.csv",
        help="Input per-point comparison CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/oos_validation",
        help="Output directory for OOS artifacts.",
    )
    parser.add_argument(
        "--a0",
        type=float,
        default=_A0_DEFAULT,
        help=f"Characteristic acceleration in m/s² (default: {_A0_DEFAULT:.2e}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    n_gal, median_delta, p_value = run_oos_validation(
        comparison_csv=Path(args.comparison_csv),
        out_dir=Path(args.out_dir),
        a0=float(args.a0),
    )
    print(f"n_galaxies={n_gal}")
    print(f"median_delta_rmse_out={median_delta}")
    print(f"p_value_wilcoxon={p_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
