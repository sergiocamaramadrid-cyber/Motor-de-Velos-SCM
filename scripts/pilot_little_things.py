#!/usr/bin/env python3
"""
Pilot run for LITTLE THINGS using the repository's blind-test models.

Selects a reproducible random sample of galaxies from the global LITTLE THINGS
table and writes:
  - per-galaxy predictions for BTFR and interpolation baselines
  - a compact pilot summary
  - metadata for reproducibility
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.blind_test_little_things import (
        A0_DEFAULT,
        GAS_FRACTION_DEFAULT,
        REQUIRED_COLS,
        predict_logv_btfr,
        predict_logv_interp,
    )
except ModuleNotFoundError:  # pragma: no cover - runtime path fallback
    from blind_test_little_things import (
        A0_DEFAULT,
        GAS_FRACTION_DEFAULT,
        REQUIRED_COLS,
        predict_logv_btfr,
        predict_logv_interp,
    )

DEFAULT_CATALOG = Path("data/little_things_global.csv")
SCRIPT_VERSION = "pilot-v0.1"


def get_git_hash() -> str:
    """Return current git commit hash (or 'unknown' outside a git checkout)."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def compute_file_hash(filepath: str | Path) -> str:
    """Return SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with Path(filepath).open("rb") as fh:
        for block in iter(lambda: fh.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


def load_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Load LITTLE THINGS global table and validate required columns."""
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def load_galaxy_list(catalog_path: str | Path) -> list[str]:
    """Return galaxy identifiers from the LITTLE THINGS global table."""
    df = load_catalog(catalog_path)
    return df["galaxy_id"].tolist()


def run_pilot(
    catalog_path: str | Path,
    outdir: str | Path,
    n: int = 10,
    seed: int = 42,
    gas_fraction: float = GAS_FRACTION_DEFAULT,
    a0: float = A0_DEFAULT,
) -> tuple[Path, Path, Path]:
    """Run pilot and write outputs; returns (predictions, summary, metadata)."""
    rng = random.Random(seed)
    np.random.seed(seed)

    catalog = load_catalog(catalog_path)
    sample_n = min(n, len(catalog))
    sample_ids = rng.sample(catalog["galaxy_id"].tolist(), sample_n)
    pilot_df = catalog[catalog["galaxy_id"].isin(sample_ids)].copy()
    pilot_df = pilot_df.set_index("galaxy_id").loc[sample_ids].reset_index()

    pred_btfr = predict_logv_btfr(
        pilot_df["logM"].to_numpy(), gas_fraction=gas_fraction, a0=a0
    )
    pred_interp = predict_logv_interp(
        pilot_df["log_gbar"].to_numpy(),
        pilot_df["log_j"].to_numpy(),
        a0=a0,
    )

    pilot_df["pred_logV_btfr"] = pred_btfr
    pilot_df["pred_logV_interp"] = pred_interp
    pilot_df["residual_btfr"] = pred_btfr - pilot_df["logVobs"].to_numpy()
    pilot_df["residual_interp"] = pred_interp - pilot_df["logVobs"].to_numpy()
    pilot_df["best_model"] = np.where(
        np.abs(pilot_df["residual_interp"]) < np.abs(pilot_df["residual_btfr"]),
        "interp",
        "btfr",
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    predictions_path = outdir / f"pilot_predictions_n{sample_n}_seed{seed}-{SCRIPT_VERSION}.csv"
    pilot_df.to_csv(predictions_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "model": "btfr",
                "rmse": float(np.sqrt(np.mean(np.square(pilot_df["residual_btfr"])))),
                "mae": float(np.mean(np.abs(pilot_df["residual_btfr"]))),
                "bias": float(np.mean(pilot_df["residual_btfr"])),
                "n_galaxies": int(sample_n),
            },
            {
                "model": "interp",
                "rmse": float(np.sqrt(np.mean(np.square(pilot_df["residual_interp"])))),
                "mae": float(np.mean(np.abs(pilot_df["residual_interp"]))),
                "bias": float(np.mean(pilot_df["residual_interp"])),
                "n_galaxies": int(sample_n),
            },
        ]
    )
    summary_path = outdir / f"pilot_summary_n{sample_n}_seed{seed}-{SCRIPT_VERSION}.csv"
    summary.to_csv(summary_path, index=False)

    metadata = {
        "git_commit": get_git_hash(),
        "catalog_file": str(Path(catalog_path)),
        "catalog_hash": compute_file_hash(catalog_path),
        "seed": seed,
        "n_requested": n,
        "n_selected": sample_n,
        "sample_ids": sample_ids,
        "gas_fraction": gas_fraction,
        "a0": a0,
        "script_version": SCRIPT_VERSION,
    }
    metadata_path = outdir / f"pilot_metadata_n{sample_n}_seed{seed}.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    return predictions_path, summary_path, metadata_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot test on LITTLE THINGS galaxies")
    parser.add_argument("--n", type=int, default=10, help="Number of galaxies to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG), help="Path to catalog CSV")
    parser.add_argument("--outdir", type=str, default="results/pilot", help="Output directory")
    parser.add_argument("--gas-fraction", type=float, default=GAS_FRACTION_DEFAULT, help="M_gas / M_star for BTFR")
    parser.add_argument("--a0", type=float, default=A0_DEFAULT, help="Characteristic acceleration in m/s^2")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pred_path, summary_path, meta_path = run_pilot(
        catalog_path=args.catalog,
        outdir=args.outdir,
        n=args.n,
        seed=args.seed,
        gas_fraction=args.gas_fraction,
        a0=args.a0,
    )
    print("Pilot completed:")
    print(f"  predictions: {pred_path}")
    print(f"  summary:     {summary_path}")
    print(f"  metadata:    {meta_path}")


if __name__ == "__main__":
    main()
