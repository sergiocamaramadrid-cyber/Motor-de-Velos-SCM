from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _default_input() -> Path:
    for candidate in [
        Path("results/universal_term_comparison_full.csv"),
        Path("results/reproducibility/universal_term_comparison_full.csv"),
    ]:
        if candidate.exists():
            return candidate
    return Path("results/reproducibility/universal_term_comparison_full.csv")


def _ensure_input(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "galaxy": ["G1", "G1", "G2", "G2", "G3", "G3"],
            "r_kpc": [1, 2, 1, 2, 1, 2],
            "g_bar": [1e-11, 2e-11, 1.5e-11, 2.2e-11, 1.2e-11, 1.9e-11],
            "g_obs": [1.05e-11, 2.15e-11, 1.48e-11, 2.25e-11, 1.22e-11, 2.01e-11],
        }
    ).to_csv(path, index=False)
    return path


def _predict_baseline(gbar: np.ndarray, a0: float) -> np.ndarray:
    x = np.maximum(gbar / a0, 1e-12)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    return nu * gbar


def _predict_scm(gbar: np.ndarray, a0: float) -> np.ndarray:
    return gbar + a0


def run_oos(df: pd.DataFrame, seed: int, n_bootstrap: int, a0: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    galaxies = np.array(sorted(df["galaxy"].unique()))
    n_test = max(1, int(np.ceil(0.3 * len(galaxies))))

    rows: list[dict[str, float | int]] = []
    for b in range(n_bootstrap):
        test_gal = rng.choice(galaxies, size=n_test, replace=True)
        test_df = df[df["galaxy"].isin(test_gal)]
        if test_df.empty:
            continue
        gbar = test_df["g_bar"].to_numpy(dtype=float)
        gobs = test_df["g_obs"].to_numpy(dtype=float)
        pred_baseline = _predict_baseline(gbar, a0)
        pred_scm = _predict_scm(gbar, a0)
        err_baseline = gobs - pred_baseline
        err_scm = gobs - pred_scm

        rmse_baseline = float(np.sqrt(np.mean(err_baseline ** 2)))
        rmse_scm = float(np.sqrt(np.mean(err_scm ** 2)))

        def _logl(err: np.ndarray) -> float:
            sigma2 = float(np.mean(err ** 2))
            sigma2 = max(sigma2, 1e-30)
            n = len(err)
            return float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))

        rows.append(
            {
                "bootstrap_id": b,
                "n_test_points": int(len(test_df)),
                "rmse_out_baseline": rmse_baseline,
                "rmse_out_scm": rmse_scm,
                "delta_rmse_out": rmse_scm - rmse_baseline,
                "logL_out_baseline": _logl(err_baseline),
                "logL_out_scm": _logl(err_scm),
                "delta_logL_out": _logl(err_scm) - _logl(err_baseline),
            }
        )
    return pd.DataFrame(rows)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 70/30 bootstrap out-of-sample validation")
    parser.add_argument("--input", default=str(_default_input()), help="Input per-point comparison CSV")
    parser.add_argument("--bootstrap", type=int, default=100, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--a0", type=float, default=1.2e-10, help="Characteristic acceleration")
    parser.add_argument(
        "--out",
        default="results/oos_validation/oos_generalization_results.csv",
        help="Output CSV path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = _ensure_input(Path(args.input))
    df = pd.read_csv(input_path)
    required = {"galaxy", "g_bar", "g_obs"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {sorted(missing)}")
        return 1

    out_df = run_oos(df, seed=int(args.seed), n_bootstrap=int(args.bootstrap), a0=float(args.a0))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"OOS validation report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
