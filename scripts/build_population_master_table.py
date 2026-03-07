from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


_F3_REQUIRED_COLUMNS = [
    "galaxy",
    "delta_f3",
    "deep_slope",
    "n_tail_points",
    "tail_r_min",
    "tail_r_max",
]
_F3_LEGACY_MIN_COLUMNS = ["galaxy", "deep_slope"]

_OUTPUT_OPTIONAL_COLUMNS = ["logSigmaHI_out", "logMstar", "Rdisk", "inclination"]


def _norm_galaxy(value: object) -> str:
    return str(value).strip().upper().replace(" ", "")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _group_first_non_null(df: pd.DataFrame, value_col: str) -> pd.Series:
    sub = df[["galaxy_norm", value_col]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("galaxy_norm", sort=False)[value_col].first()


def _load_f3_catalog(path: Path) -> pd.DataFrame:
    f3 = pd.read_csv(path)
    missing_legacy = [c for c in _F3_LEGACY_MIN_COLUMNS if c not in f3.columns]
    if missing_legacy:
        raise ValueError(f"F3 catalog missing required columns: {missing_legacy}")

    out = f3.copy()
    if "delta_f3" not in out.columns:
        # Legacy F3 tables may not store delta_f3/expected_slope explicitly.
        # Use the deep-regime reference slope 0.5 as project default.
        expected = out["expected_slope"] if "expected_slope" in out.columns else 0.5
        out["delta_f3"] = out["deep_slope"] - expected
    if "n_tail_points" not in out.columns:
        out["n_tail_points"] = np.nan
    if "tail_r_min" not in out.columns:
        out["tail_r_min"] = np.nan
    if "tail_r_max" not in out.columns:
        out["tail_r_max"] = np.nan

    out = out[_F3_REQUIRED_COLUMNS].copy()
    out["galaxy_norm"] = out["galaxy"].map(_norm_galaxy)
    return out


def _extract_from_full_catalog(path: Path) -> pd.DataFrame:
    full = pd.read_csv(path)
    galaxy_col = _first_existing_column(full, ["galaxy", "Galaxy"])
    if galaxy_col is None:
        return pd.DataFrame(columns=["galaxy_norm", *_OUTPUT_OPTIONAL_COLUMNS])

    full = full.copy()
    full["galaxy_norm"] = full[galaxy_col].map(_norm_galaxy)
    out = pd.DataFrame({"galaxy_norm": full["galaxy_norm"].drop_duplicates()})

    aliases = {
        "logSigmaHI_out": ["logSigmaHI_out"],
        "logMstar": ["logMstar"],
        "Rdisk": ["Rdisk"],
        "inclination": ["inclination", "Inc"],
    }
    for out_col, candidates in aliases.items():
        col = _first_existing_column(full, candidates)
        if col is None:
            continue
        values = _group_first_non_null(full, col)
        if not values.empty:
            out = out.merge(values.rename(out_col), left_on="galaxy_norm", right_index=True, how="left")

    return out


def _extract_from_master_catalog(path: Path) -> pd.DataFrame:
    master = pd.read_csv(path)
    galaxy_col = _first_existing_column(master, ["galaxy", "Galaxy"])
    if galaxy_col is None:
        return pd.DataFrame(columns=["galaxy_norm", *_OUTPUT_OPTIONAL_COLUMNS])

    master = master.copy()
    master["galaxy_norm"] = master[galaxy_col].map(_norm_galaxy)
    out = pd.DataFrame({"galaxy_norm": master["galaxy_norm"].drop_duplicates()})

    sigma_col = _first_existing_column(master, ["logSigmaHI_out"])
    if sigma_col is not None:
        out = out.merge(_group_first_non_null(master, sigma_col).rename("logSigmaHI_out"), left_on="galaxy_norm", right_index=True, how="left")

    mstar_col = _first_existing_column(master, ["logMstar"])
    if mstar_col is not None:
        out = out.merge(_group_first_non_null(master, mstar_col).rename("logMstar"), left_on="galaxy_norm", right_index=True, how="left")
    elif "L_3.6" in master.columns:
        # Consistent with existing SPARC utilities in this repo:
        # Mstar_1e9 = 0.5 * L_3.6  ->  logMstar = log10(Mstar_1e9) + 9.
        l36 = pd.to_numeric(master["L_3.6"], errors="coerce")
        log_mstar = pd.Series(np.where(l36 > 0, np.log10(0.5 * l36) + 9.0, np.nan), index=master.index)
        tmp = pd.DataFrame({"galaxy_norm": master["galaxy_norm"], "logMstar": log_mstar})
        out = out.merge(_group_first_non_null(tmp, "logMstar").rename("logMstar"), left_on="galaxy_norm", right_index=True, how="left")

    rdisk_col = _first_existing_column(master, ["Rdisk"])
    if rdisk_col is not None:
        out = out.merge(_group_first_non_null(master, rdisk_col).rename("Rdisk"), left_on="galaxy_norm", right_index=True, how="left")

    inc_col = _first_existing_column(master, ["inclination", "Inc"])
    if inc_col is not None:
        out = out.merge(_group_first_non_null(master, inc_col).rename("inclination"), left_on="galaxy_norm", right_index=True, how="left")

    return out


def build_population_master_table(
    f3_catalog_path: Path,
    out_path: Path,
    *,
    full_catalog_path: Path | None = None,
    master_catalog_path: Path | None = None,
) -> pd.DataFrame:
    f3 = _load_f3_catalog(f3_catalog_path)
    out = f3.copy()

    if full_catalog_path is not None:
        full = _extract_from_full_catalog(full_catalog_path)
        out = out.merge(full, on="galaxy_norm", how="left")

    if master_catalog_path is not None:
        master = _extract_from_master_catalog(master_catalog_path)
        out = out.merge(master, on="galaxy_norm", how="left", suffixes=("", "_master"))
        for col in _OUTPUT_OPTIONAL_COLUMNS:
            master_col = f"{col}_master"
            if master_col in out.columns:
                if col in out.columns:
                    out[col] = out[col].combine_first(out[master_col])
                else:
                    out[col] = out[master_col]
                out = out.drop(columns=[master_col])

    for col in _OUTPUT_OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out.drop(columns=["galaxy_norm"]).sort_values("galaxy").reset_index(drop=True)
    out = out[_F3_REQUIRED_COLUMNS + _OUTPUT_OPTIONAL_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build galaxy-level master table for population analysis."
    )
    parser.add_argument(
        "--f3-catalog",
        required=True,
        help="Path to f3_catalog.csv with delta_f3 and deep/tail columns.",
    )
    parser.add_argument(
        "--full-catalog",
        default=None,
        help="Optional per-point catalog (e.g. sparc_full.csv) with logSigmaHI_out.",
    )
    parser.add_argument(
        "--master-catalog",
        default=None,
        help="Optional SPARC master table for logMstar/Rdisk/inclination completion.",
    )
    parser.add_argument(
        "--out",
        default="results/SPARC/population_master_table.csv",
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    table = build_population_master_table(
        f3_catalog_path=Path(args.f3_catalog),
        full_catalog_path=Path(args.full_catalog) if args.full_catalog else None,
        master_catalog_path=Path(args.master_catalog) if args.master_catalog else None,
        out_path=Path(args.out),
    )
    print(f"Master table written: {args.out}")
    print(f"Rows: {len(table)}")
    print(f"Columns: {list(table.columns)}")


if __name__ == "__main__":
    main()
