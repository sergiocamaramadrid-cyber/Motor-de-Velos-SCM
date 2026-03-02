from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _create_test_contract_tables(tmp_path: Path) -> tuple[Path, Path]:
    galaxies = pd.DataFrame({"galaxy": ["G002", "G001"]})
    rc_points = pd.DataFrame(
        {
            "galaxy": ["G001", "G001", "G002", "G002"],
            "r_kpc": [8.0, 12.0, 8.0, 12.0],
            "vobs_kms": [35.0, 30.0, 37.0, 32.0],
            "vgas_kms": [10.0, 9.0, 11.0, 10.0],
            "vdisk_kms": [15.0, 14.0, 15.5, 14.5],
            "vbul_kms": [0.0, 0.0, 0.0, 0.0],
        }
    )
    galaxies_path = tmp_path / "galaxies.csv"
    rc_points_path = tmp_path / "rc_points.csv"
    galaxies.to_csv(galaxies_path, index=False)
    rc_points.to_csv(rc_points_path, index=False)
    return galaxies_path, rc_points_path


def test_ingest_then_catalog_e2e_module(tmp_path):
    galaxies_path, rc_points_path = _create_test_contract_tables(tmp_path)
    ingest_out = tmp_path / "ingested"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_big_sparc_contract",
            "--galaxies",
            str(galaxies_path),
            "--rc-points",
            str(rc_points_path),
            "--out",
            str(ingest_out),
        ],
        check=True,
    )

    rc_ingested = pd.read_parquet(ingest_out / "rc_points.parquet")
    expected_vbar = np.sqrt(
        rc_ingested["vgas_kms"] ** 2 + rc_ingested["vdisk_kms"] ** 2 + rc_ingested["vbul_kms"] ** 2
    )
    np.testing.assert_allclose(rc_ingested["vbar_kms"], expected_vbar)
    assert rc_ingested["galaxy"].tolist() == sorted(rc_ingested["galaxy"].tolist())

    f3_out = tmp_path / "f3_module"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.generate_f3_catalog_from_contract",
            "--rc-points",
            str(ingest_out / "rc_points.parquet"),
            "--out",
            str(f3_out),
            "--deep-threshold",
            "100",
            "--min-deep",
            "2",
        ],
        check=True,
    )

    catalog = pd.read_parquet(f3_out / "f3_catalog.parquet")
    assert set(catalog.columns) == {"galaxy", "n_points", "n_deep", "beta_slope", "beta_intercept"}
    assert (catalog["n_deep"] >= 2).all()
    assert catalog["beta_slope"].notna().all()


def test_catalog_script_entrypoint(tmp_path):
    galaxies_path, rc_points_path = _create_test_contract_tables(tmp_path)
    ingest_out = tmp_path / "ingested_script"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_big_sparc_contract",
            "--galaxies",
            str(galaxies_path),
            "--rc-points",
            str(rc_points_path),
            "--out",
            str(ingest_out),
        ],
        check=True,
    )

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "generate_f3_catalog_from_contract.py"
    direct_out = tmp_path / "f3_script"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--rc-points",
            str(ingest_out / "rc_points.parquet"),
            "--out",
            str(direct_out),
            "--deep-threshold",
            "100",
            "--min-deep",
            "2",
        ],
        check=True,
    )
    assert (direct_out / "f3_catalog.parquet").exists()
