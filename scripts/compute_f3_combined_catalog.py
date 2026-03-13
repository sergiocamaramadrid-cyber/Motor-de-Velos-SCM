#!/usr/bin/env python3

"""
Calcula observables del Framework SCM para el catálogo combinado.

Entradas:
- results/combined/framework_master_catalog.csv
- results/LITTLE_THINGS_Oh2015/little_things_rotcurves.csv

Salida:
- results/combined/f3_combined_catalog.csv

Definiciones oficiales de esta versión:
- f3_scm := pendiente dlog(V)/dlog(R) ajustada en la cola externa
- delta_f3 := f3_scm - 0.5
- cola externa := puntos con r_scaled >= tail_rmin
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


F3_REFERENCE_SLOPE = 0.5
DEFAULT_TAIL_RMIN = 0.7
DEFAULT_MIN_TAIL_POINTS = 3
FIT_METHOD = "polyfit_log10"
QUALITY_RESTRICTED_TAIL_THRESHOLD = 0.7


def compute_tail_slope(r: np.ndarray, v: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    mask = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    r = r[mask]
    v = v[mask]

    if len(r) < 2:
        return np.nan

    log_r = np.log10(r)
    log_v = np.log10(v)

    slope, _ = np.polyfit(log_r, log_v, 1)
    return float(slope)


def classify_quality(
    fit_ok: bool,
    n_tail_points: int,
    tail_rmin: float,
    min_tail_points: int,
) -> str:
    if not fit_ok:
        return "no_valid_tail_fit"
    if n_tail_points == min_tail_points:
        return "minimal_tail"
    if tail_rmin > QUALITY_RESTRICTED_TAIL_THRESHOLD:
        return "restricted_tail"
    return "ok"


def compute_f3_from_rotcurve(
    df: pd.DataFrame,
    tail_rmin: float = DEFAULT_TAIL_RMIN,
    min_tail_points: int = DEFAULT_MIN_TAIL_POINTS,
) -> dict:
    r = df["r_scaled"].to_numpy(dtype=float)
    v = df["v_scaled"].to_numpy(dtype=float)

    finite_mask = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    r = r[finite_mask]
    v = v[finite_mask]

    tail_mask = r >= tail_rmin
    r_tail = r[tail_mask]
    v_tail = v[tail_mask]

    if len(r_tail) < min_tail_points:
        return {
            "f3_scm": np.nan,
            "delta_f3": np.nan,
            "tail_slope": np.nan,
            "n_tail_points": int(len(r_tail)),
            "tail_rmin": float(tail_rmin),
            "fit_method": FIT_METHOD,
            "fit_ok": False,
            "fit_ok_reason": "insufficient_tail_points",
            "quality_flag": "no_valid_tail_fit",
        }

    slope = compute_tail_slope(r_tail, v_tail)

    if not np.isfinite(slope):
        return {
            "f3_scm": np.nan,
            "delta_f3": np.nan,
            "tail_slope": np.nan,
            "n_tail_points": int(len(r_tail)),
            "tail_rmin": float(tail_rmin),
            "fit_method": FIT_METHOD,
            "fit_ok": False,
            "fit_ok_reason": "invalid_polyfit",
            "quality_flag": "no_valid_tail_fit",
        }

    fit_ok = True
    quality_flag = classify_quality(
        fit_ok=fit_ok,
        n_tail_points=int(len(r_tail)),
        tail_rmin=float(tail_rmin),
        min_tail_points=min_tail_points,
    )

    return {
        "f3_scm": float(slope),
        "delta_f3": float(slope - F3_REFERENCE_SLOPE),
        "tail_slope": float(slope),
        "n_tail_points": int(len(r_tail)),
        "tail_rmin": float(tail_rmin),
        "fit_method": FIT_METHOD,
        "fit_ok": True,
        "fit_ok_reason": "ok",
        "quality_flag": quality_flag,
    }


def build_f3_catalog(
    master: pd.DataFrame,
    rot: pd.DataFrame,
    tail_rmin: float = DEFAULT_TAIL_RMIN,
    min_tail_points: int = DEFAULT_MIN_TAIL_POINTS,
) -> pd.DataFrame:
    rows = []

    for _, row in master.iterrows():
        galaxy = row["galaxy"]
        source_catalog = row.get("source_catalog", "UNKNOWN")
        rotcurve_available = bool(row.get("rotcurve_available", False))

        sub = rot[rot["galaxy"] == galaxy].copy()

        if (not rotcurve_available) or len(sub) == 0:
            rows.append(
                {
                    "galaxy": galaxy,
                    "source_catalog": source_catalog,
                    "f3_scm": np.nan,
                    "delta_f3": np.nan,
                    "tail_slope": np.nan,
                    "n_tail_points": 0,
                    "tail_rmin": float(tail_rmin),
                    "fit_method": FIT_METHOD,
                    "fit_ok": False,
                    "fit_ok_reason": "no_rotcurve_data",
                    "quality_flag": "no_rotcurve_data",
                }
            )
            continue

        metrics = compute_f3_from_rotcurve(
            sub,
            tail_rmin=tail_rmin,
            min_tail_points=min_tail_points,
        )

        rows.append(
            {
                "galaxy": galaxy,
                "source_catalog": source_catalog,
                **metrics,
            }
        )

    f3 = pd.DataFrame(rows)
    combined = master.merge(
        f3.drop(columns=["source_catalog"], errors="ignore"),
        on="galaxy",
        how="left",
        validate="one_to_one",
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-path",
        type=Path,
        default=Path("results/combined/framework_master_catalog.csv"),
    )
    parser.add_argument(
        "--rot-path",
        type=Path,
        default=Path("results/LITTLE_THINGS_Oh2015/little_things_rotcurves.csv"),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("results/combined/f3_combined_catalog.csv"),
    )
    parser.add_argument(
        "--tail-rmin",
        type=float,
        default=DEFAULT_TAIL_RMIN,
    )
    parser.add_argument(
        "--min-tail-points",
        type=int,
        default=DEFAULT_MIN_TAIL_POINTS,
    )
    args = parser.parse_args()

    if not args.master_path.exists():
        raise FileNotFoundError(args.master_path)

    if not args.rot_path.exists():
        raise FileNotFoundError(args.rot_path)

    master = pd.read_csv(args.master_path)
    rot = pd.read_csv(args.rot_path)

    combined = build_f3_catalog(
        master=master,
        rot=rot,
        tail_rmin=args.tail_rmin,
        min_tail_points=args.min_tail_points,
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_path, index=False)

    print("F3 combined catalog creado:")
    print(args.out_path)
    print(f"N galaxias: {len(combined)}")
    print(f"Fit OK: {int(combined['fit_ok'].fillna(False).sum())}")


if __name__ == "__main__":
    main()
