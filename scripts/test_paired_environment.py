#!/usr/bin/env python3
"""
test_paired_environment.py

Test de pares emparejados para el Framework SCM.

Objetivo:
    Evaluar si delta_f3 depende sistemáticamente del entorno externo
    (logSigmaHI_out) una vez controladas masa bariónica y escala de disco.

Salidas:
    results/paired_environment/paired_sample.csv
    results/paired_environment/paired_stats_summary.csv
    results/paired_environment/paired_bootstrap.csv
    results/paired_environment/placebo_tests.csv
    results/paired_environment/delta_f3_vs_delta_logSigmaHI.png
    results/paired_environment/run_metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT = "data/sparc_175_master.csv"
DEFAULT_OUTPUT_DIR = "results/paired_environment"
DEFAULT_SEED = 42


@dataclass
class MatchConfig:
    calipers: List[float]
    radial_cuts: List[float]
    direction: str
    min_tail_points: int
    min_inclination: float
    quality_good_values: Tuple[str, ...]
    bootstrap_n: int
    placebo_n: int
    main_radial_cut: float


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_std(x: pd.Series) -> float:
    val = float(np.nanstd(x, ddof=1))
    return val if np.isfinite(val) and val > 0 else 1.0


def aic_bic_from_residuals(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> Tuple[float, float]:
    n = len(y_true)
    if n <= 0:
        return np.nan, np.nan
    resid = y_true - y_pred
    rss = float(np.sum(resid ** 2))
    if rss <= 0:
        rss = 1e-12
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic


def apply_quality_filters(
    df: pd.DataFrame,
    min_tail_points: int,
    min_inclination: float,
    quality_good_values: Tuple[str, ...],
) -> pd.DataFrame:
    required = ["galaxy", "delta_f3", "F3", "logSigmaHI_out", "logMbar", "logRd", "fit_ok"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    mask = (
        df["fit_ok"].astype(bool)
        & np.isfinite(df["delta_f3"])
        & np.isfinite(df["F3"])
        & np.isfinite(df["logSigmaHI_out"])
        & np.isfinite(df["logMbar"])
        & np.isfinite(df["logRd"])
    )

    if "n_tail_points" in df.columns:
        mask &= df["n_tail_points"].fillna(0) >= min_tail_points

    if "inclination" in df.columns:
        mask &= df["inclination"].fillna(-999) >= min_inclination

    if "quality_flag" in df.columns:
        q = df["quality_flag"].astype(str).str.lower().str.strip()
        mask &= q.isin([v.lower() for v in quality_good_values])

    out = df.loc[mask].copy()
    out = out.drop_duplicates(subset=["galaxy"]).reset_index(drop=True)
    return out


def add_normalized_matching_columns(
    df: pd.DataFrame, cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    out = df.copy()
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    for col in cols:
        mu = float(np.nanmean(out[col]))
        sd = safe_std(out[col])
        means[col] = mu
        stds[col] = sd
        out[f"{col}_z"] = (out[col] - mu) / sd

    return out, means, stds


def pair_distance(row_i: pd.Series, row_j: pd.Series, cols: List[str]) -> float:
    s = 0.0
    for col in cols:
        dz = float(row_i[f"{col}_z"] - row_j[f"{col}_z"])
        s += dz * dz
    return float(np.sqrt(s))


def build_pairs_without_replacement(
    df: pd.DataFrame,
    caliper: float,
    match_cols: List[str],
) -> pd.DataFrame:
    n = len(df)
    if n < 2:
        return pd.DataFrame()

    used = np.zeros(n, dtype=bool)
    pairs: List[dict] = []

    dist = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = pair_distance(df.iloc[i], df.iloc[j], match_cols)
            dist[i, j] = d
            dist[j, i] = d

    candidate_edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= caliper:
                candidate_edges.append((dist[i, j], i, j))

    candidate_edges.sort(key=lambda x: x[0])

    for d, i, j in candidate_edges:
        if used[i] or used[j]:
            continue
        used[i] = True
        used[j] = True
        a = df.iloc[i]
        b = df.iloc[j]

        if float(a["logSigmaHI_out"]) < float(b["logSigmaHI_out"]):
            a, b = b, a

        pairs.append(
            {
                "galaxy_A": a["galaxy"],
                "galaxy_B": b["galaxy"],
                "logMbar_A": a["logMbar"],
                "logMbar_B": b["logMbar"],
                "logRd_A": a["logRd"],
                "logRd_B": b["logRd"],
                "logSigmaHI_out_A": a["logSigmaHI_out"],
                "logSigmaHI_out_B": b["logSigmaHI_out"],
                "delta_f3_A": a["delta_f3"],
                "delta_f3_B": b["delta_f3"],
                "F3_A": a["F3"],
                "F3_B": b["F3"],
                "match_distance": d,
            }
        )

    out = pd.DataFrame(pairs)
    if out.empty:
        return out

    out["delta_logSigma"] = out["logSigmaHI_out_A"] - out["logSigmaHI_out_B"]
    out["delta_delta_f3"] = out["delta_f3_A"] - out["delta_f3_B"]
    out["delta_F3"] = out["F3_A"] - out["F3_B"]
    return out


def _binom_pvalue(n_pos: int, n_total: int, direction: str) -> float:
    if n_total <= 0:
        return np.nan
    alt = {"two-sided": "two-sided", "greater": "greater", "less": "less"}[direction]
    return float(stats.binomtest(n_pos, n_total, p=0.5, alternative=alt).pvalue)


def sign_test(diff: np.ndarray, direction: str) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]

    n_total = int(len(diff))
    n_pos = int(np.sum(diff > 0))
    p = _binom_pvalue(n_pos, n_total, direction)

    return {
        "n_pairs_nonzero": n_total,
        "n_positive": n_pos,
        "sign_fraction_positive": (n_pos / n_total) if n_total > 0 else np.nan,
        "p_sign": p,
    }


def wilcoxon_test(diff: np.ndarray, direction: str) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]

    if len(diff) < 1:
        return {"wilcoxon_stat": np.nan, "p_wilcoxon": np.nan}

    alt = {"two-sided": "two-sided", "greater": "greater", "less": "less"}[direction]
    stat, p = stats.wilcoxon(diff, alternative=alt, zero_method="wilcox", correction=False, mode="auto")
    return {"wilcoxon_stat": float(stat), "p_wilcoxon": float(p)}


def regression_on_differences(x: np.ndarray, y: np.ndarray, direction: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "reg_n": len(x),
            "reg_slope": np.nan,
            "reg_intercept": np.nan,
            "reg_r": np.nan,
            "reg_p": np.nan,
            "reg_stderr": np.nan,
            "reg_rmse": np.nan,
            "reg_mae": np.nan,
            "reg_aic": np.nan,
            "reg_bic": np.nan,
        }

    lr = stats.linregress(x, y)
    slope = float(lr.slope)
    intercept = float(lr.intercept)
    r = float(lr.rvalue)
    p_two = float(lr.pvalue)
    stderr = float(lr.stderr)

    if direction == "two-sided":
        p_use = p_two
    elif direction == "greater":
        p_use = p_two / 2 if slope > 0 else 1.0 - p_two / 2
    elif direction == "less":
        p_use = p_two / 2 if slope < 0 else 1.0 - p_two / 2
    else:
        raise ValueError(f"direction no válida: {direction}")

    y_pred = intercept + slope * x
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y - y_pred)))
    aic, bic = aic_bic_from_residuals(y, y_pred, k=2)

    return {
        "reg_n": int(len(x)),
        "reg_slope": slope,
        "reg_intercept": intercept,
        "reg_r": r,
        "reg_p": float(p_use),
        "reg_stderr": stderr,
        "reg_rmse": rmse,
        "reg_mae": mae,
        "reg_aic": aic,
        "reg_bic": bic,
    }


def spearman_test(x: np.ndarray, y: np.ndarray, direction: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {"spearman_rho": np.nan, "spearman_p": np.nan}

    rho, p_two = stats.spearmanr(x, y)

    if direction == "two-sided":
        p_use = float(p_two)
    elif direction == "greater":
        p_use = float(p_two / 2) if rho > 0 else float(1.0 - p_two / 2)
    elif direction == "less":
        p_use = float(p_two / 2) if rho < 0 else float(1.0 - p_two / 2)
    else:
        raise ValueError(f"direction no válida: {direction}")

    return {"spearman_rho": float(rho), "spearman_p": p_use}


def evaluate_pairs(pairs: pd.DataFrame, direction: str, response_col: str = "delta_delta_f3") -> Dict[str, float]:
    if pairs.empty:
        return {"n_pairs": 0}

    diff = pairs[response_col].to_numpy(dtype=float)
    x = pairs["delta_logSigma"].to_numpy(dtype=float)

    out = {"n_pairs": int(len(pairs))}
    out.update(sign_test(diff, direction))
    out.update(wilcoxon_test(diff, direction))
    out.update(regression_on_differences(x, diff, direction))
    out.update(spearman_test(x, diff, direction))

    out["median_effect"] = float(np.nanmedian(diff)) if len(diff) > 0 else np.nan
    out["mean_effect"] = float(np.nanmean(diff)) if len(diff) > 0 else np.nan
    out["std_effect"] = float(np.nanstd(diff, ddof=1)) if len(diff) > 1 else np.nan
    return out


def bootstrap_pairs(
    pairs: pd.DataFrame,
    n_bootstrap: int,
    direction: str,
    rng: np.random.Generator,
    response_col: str = "delta_delta_f3",
) -> pd.DataFrame:
    rows: List[dict] = []
    if pairs.empty:
        return pd.DataFrame()

    for b in range(n_bootstrap):
        boot = pairs.sample(n=len(pairs), replace=True, random_state=int(rng.integers(0, 2**31 - 1))).copy()
        stats_row = evaluate_pairs(boot, direction, response_col=response_col)
        stats_row["bootstrap_id"] = b
        rows.append(stats_row)

    return pd.DataFrame(rows)


def placebo_permutation_logsigma(
    pairs: pd.DataFrame,
    n_perm: int,
    direction: str,
    rng: np.random.Generator,
    response_col: str = "delta_delta_f3",
) -> pd.DataFrame:
    rows: List[dict] = []
    if pairs.empty:
        return pd.DataFrame()

    base = pairs.copy()

    for i in range(n_perm):
        vals = np.concatenate(
            [
                base["logSigmaHI_out_A"].to_numpy(dtype=float),
                base["logSigmaHI_out_B"].to_numpy(dtype=float),
            ]
        )
        rng.shuffle(vals)

        perm = base.copy()
        n = len(perm)
        perm["logSigmaHI_out_A"] = vals[:n]
        perm["logSigmaHI_out_B"] = vals[n:]

        swap = perm["logSigmaHI_out_A"] < perm["logSigmaHI_out_B"]
        cols_a = ["logSigmaHI_out_A", "delta_f3_A", "F3_A", "logMbar_A", "logRd_A", "galaxy_A"]
        cols_b = ["logSigmaHI_out_B", "delta_f3_B", "F3_B", "logMbar_B", "logRd_B", "galaxy_B"]

        for ca, cb in zip(cols_a, cols_b):
            tmp = perm.loc[swap, ca].copy()
            perm.loc[swap, ca] = perm.loc[swap, cb].values
            perm.loc[swap, cb] = tmp.values

        perm["delta_logSigma"] = perm["logSigmaHI_out_A"] - perm["logSigmaHI_out_B"]
        perm["delta_delta_f3"] = perm["delta_f3_A"] - perm["delta_f3_B"]
        perm["delta_F3"] = perm["F3_A"] - perm["F3_B"]

        row = evaluate_pairs(perm, direction, response_col=response_col)
        row["perm_id"] = i
        rows.append(row)

    return pd.DataFrame(rows)


def placebo_random_response(
    pairs: pd.DataFrame,
    n_perm: int,
    direction: str,
    rng: np.random.Generator,
    response_col: str = "delta_delta_f3",
) -> pd.DataFrame:
    rows: List[dict] = []
    if pairs.empty:
        return pd.DataFrame()

    base = pairs.copy()

    for i in range(n_perm):
        perm = base.copy()
        y = perm[response_col].to_numpy(dtype=float).copy()
        rng.shuffle(y)
        perm[response_col] = y

        row = evaluate_pairs(perm, direction, response_col=response_col)
        row["perm_id"] = i
        rows.append(row)

    return pd.DataFrame(rows)


def make_plot(pairs: pd.DataFrame, outpath: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    x = pairs["delta_logSigma"].to_numpy(dtype=float)
    y = pairs["delta_delta_f3"].to_numpy(dtype=float)

    plt.scatter(x, y, alpha=0.8)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)

    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) >= 3:
        lr = stats.linregress(x[mask], y[mask])
        xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 200)
        yy = lr.intercept + lr.slope * xx
        plt.plot(xx, yy, linewidth=1.5)

    plt.xlabel(r"$\Delta \log \Sigma_{\mathrm{HI,out}}$")
    plt.ylabel(r"$\Delta(\delta F_3)$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test de pares emparejados para SCM.")
    p.add_argument("--in", dest="input_csv", default=DEFAULT_INPUT, help="CSV maestro de entrada.")
    p.add_argument("--out", dest="output_dir", default=DEFAULT_OUTPUT_DIR, help="Directorio de salida.")
    p.add_argument(
        "--calipers",
        default="0.5,0.75,1.0",
        help="Lista separada por comas de calipers en distancia normalizada.",
    )
    p.add_argument(
        "--radial-cuts",
        default="0.6,0.7,0.8",
        help="Lista separada por comas de cortes radiales a reportar.",
    )
    p.add_argument(
        "--main-radial-cut",
        type=float,
        default=0.7,
        help="Corte radial principal a destacar en outputs finales.",
    )
    p.add_argument(
        "--direction",
        choices=["two-sided", "greater", "less"],
        default="two-sided",
        help="Dirección de la hipótesis estadística.",
    )
    p.add_argument("--min-tail-points", type=int, default=3)
    p.add_argument("--min-inclination", type=float, default=30.0)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--placebo-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    rng = np.random.default_rng(args.seed)

    config = MatchConfig(
        calipers=[float(x) for x in args.calipers.split(",") if x.strip()],
        radial_cuts=[float(x) for x in args.radial_cuts.split(",") if x.strip()],
        direction=args.direction,
        min_tail_points=args.min_tail_points,
        min_inclination=args.min_inclination,
        quality_good_values=("good", "ok", "usable", "clean"),
        bootstrap_n=args.bootstrap_n,
        placebo_n=args.placebo_n,
        main_radial_cut=float(args.main_radial_cut),
    )

    df = pd.read_csv(input_csv)
    n_input = len(df)

    dfq = apply_quality_filters(
        df,
        min_tail_points=config.min_tail_points,
        min_inclination=config.min_inclination,
        quality_good_values=config.quality_good_values,
    )
    n_after_filters = len(dfq)

    if n_after_filters < 6:
        raise RuntimeError(f"Muestra demasiado pequeña tras filtros: {n_after_filters}")

    match_cols = ["logMbar", "logRd"]
    dfm, means, stds = add_normalized_matching_columns(dfq, match_cols)

    results_rows: List[dict] = []
    best_pairs: Optional[pd.DataFrame] = None
    best_key: Optional[Tuple[float, float]] = None

    for cal in config.calipers:
        for rcut in config.radial_cuts:
            pairs = build_pairs_without_replacement(dfm, caliper=cal, match_cols=match_cols)

            if pairs.empty or len(pairs) < 3:
                results_rows.append(
                    {
                        "caliper": cal,
                        "radial_cut": rcut,
                        "n_pairs": 0,
                        "status": "too_few_pairs",
                    }
                )
                continue

            stats_main = evaluate_pairs(pairs, direction=config.direction, response_col="delta_delta_f3")
            stats_secondary = evaluate_pairs(pairs, direction=config.direction, response_col="delta_F3")

            row = {
                "caliper": cal,
                "radial_cut": rcut,
                "status": "ok",
                **{f"main_{k}": v for k, v in stats_main.items()},
                **{f"secondary_{k}": v for k, v in stats_secondary.items()},
            }
            results_rows.append(row)

            pairs.to_csv(outdir / f"pairs_c{cal:g}_r{rcut:g}.csv", index=False)

            if np.isclose(rcut, config.main_radial_cut) and best_pairs is None:
                best_pairs = pairs.copy()
                best_key = (cal, rcut)

    summary = pd.DataFrame(results_rows)
    summary.to_csv(outdir / "paired_stats_summary.csv", index=False)

    if best_pairs is None:
        ok_rows = summary.loc[summary["status"] == "ok"].copy()
        if ok_rows.empty:
            raise RuntimeError("No se pudieron construir pares válidos para ningún caliper.")
        pick = ok_rows.iloc[0]
        cal = float(pick["caliper"])
        rcut = float(pick["radial_cut"])
        best_pairs = pd.read_csv(outdir / f"pairs_c{cal:g}_r{rcut:g}.csv")
        best_key = (cal, rcut)

    best_pairs.to_csv(outdir / "paired_sample.csv", index=False)

    boot = bootstrap_pairs(
        best_pairs,
        n_bootstrap=config.bootstrap_n,
        direction=config.direction,
        rng=rng,
        response_col="delta_delta_f3",
    )
    boot.to_csv(outdir / "paired_bootstrap.csv", index=False)

    placebo1 = placebo_permutation_logsigma(
        best_pairs,
        n_perm=config.placebo_n,
        direction=config.direction,
        rng=rng,
        response_col="delta_delta_f3",
    )
    placebo1["placebo_type"] = "permute_logSigmaHI_out"

    placebo2 = placebo_random_response(
        best_pairs,
        n_perm=config.placebo_n,
        direction=config.direction,
        rng=rng,
        response_col="delta_delta_f3",
    )
    placebo2["placebo_type"] = "permute_delta_delta_f3"

    placebo = pd.concat([placebo1, placebo2], ignore_index=True)
    placebo.to_csv(outdir / "placebo_tests.csv", index=False)

    make_plot(
        best_pairs,
        outdir / "delta_f3_vs_delta_logSigmaHI.png",
        title=f"Paired environment test (caliper={best_key[0]:g}, rcut={best_key[1]:g})",
    )

    metadata = {
        "script": "scripts/test_paired_environment.py",
        "input_csv": str(input_csv),
        "input_sha256": sha256_of_file(input_csv),
        "output_dir": str(outdir),
        "seed": args.seed,
        "direction": config.direction,
        "calipers": config.calipers,
        "radial_cuts": config.radial_cuts,
        "main_radial_cut": config.main_radial_cut,
        "match_cols": match_cols,
        "response_primary": "delta_f3",
        "response_secondary": "F3",
        "predictor_environment": "logSigmaHI_out",
        "n_input_rows": n_input,
        "n_rows_after_filters": n_after_filters,
        "normalization_means": means,
        "normalization_stds": stds,
        "best_pair_selection": {"caliper": best_key[0], "radial_cut": best_key[1]},
    }
    with (outdir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("OK")
    print(f"Pares guardados en: {outdir / 'paired_sample.csv'}")
    print(f"Resumen guardado en: {outdir / 'paired_stats_summary.csv'}")
    print(f"Bootstrap guardado en: {outdir / 'paired_bootstrap.csv'}")
    print(f"Placebos guardados en: {outdir / 'placebo_tests.csv'}")
    print(f"Figura guardada en: {outdir / 'delta_f3_vs_delta_logSigmaHI.png'}")
    print(f"Metadata guardada en: {outdir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
