from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CONV = 1e6 / 3.085677581e19
G0 = 1.2e-10


def _write_contract_inputs(tmp_path: Path) -> tuple[Path, Path]:
    galaxies = pd.DataFrame({"galaxy": ["G_B", "G_A"]})
    galaxies_path = tmp_path / "galaxies.csv"
    galaxies.to_csv(galaxies_path, index=False)

    rows = []
    for galaxy, vbars in (("G_B", [18.0, 20.0, 22.0]), ("G_A", [20.0, 23.0, 26.0])):
        for i, vbar in enumerate(vbars):
            r = 8.0 + i
            g_bar = (vbar ** 2 / r) * CONV
            g_obs = np.sqrt(g_bar * G0)
            rows.append(
                {
                    "galaxy": galaxy,
                    "r_kpc": r,
                    "v_obs_kms": np.sqrt(g_obs * r / CONV),
                    "vgas_kms": 0.2 * vbar,
                    "vdisk_kms": 0.7 * vbar,
                    "vbul_kms": np.sqrt(max(vbar**2 - (0.2 * vbar) ** 2 - (0.7 * vbar) ** 2, 0)),
                }
            )
    rc_points_path = tmp_path / "rc_points.csv"
    pd.DataFrame(rows).to_csv(rc_points_path, index=False)
    return galaxies_path, rc_points_path


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def test_e2e_ingest_module_and_catalog_script(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    galaxies, rc_points = _write_contract_inputs(tmp_path)
    ingested = tmp_path / "ingested"
    catalog = tmp_path / "f3_catalog.csv"

    _run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_big_sparc_contract",
            "--galaxies",
            str(galaxies),
            "--rc-points",
            str(rc_points),
            "--out-dir",
            str(ingested),
        ],
        cwd=repo_root,
    )

    rc_ingested = pd.read_parquet(ingested / "rc_points.parquet")
    assert "vbar_kms" in rc_ingested.columns
    assert rc_ingested[["galaxy", "r_kpc"]].values.tolist() == sorted(
        rc_ingested[["galaxy", "r_kpc"]].values.tolist()
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts/generate_f3_catalog_from_contract.py"),
            "--input-dir",
            str(ingested),
            "--out",
            str(catalog),
            "--min-deep",
            "3",
        ],
        cwd=repo_root,
    )

    df = pd.read_csv(catalog)
    assert list(df["galaxy"]) == ["G_A", "G_B"]
    assert df["selected"].all()
    assert np.isfinite(df["beta"]).all()


def test_e2e_ingest_script_and_catalog_module(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    galaxies, rc_points = _write_contract_inputs(tmp_path)
    ingested = tmp_path / "ingested_script"
    catalog = tmp_path / "f3_catalog.parquet"

    _run(
        [
            sys.executable,
            str(repo_root / "scripts/ingest_big_sparc_contract.py"),
            "--galaxies",
            str(galaxies),
            "--rc-points",
            str(rc_points),
            "--out-dir",
            str(ingested),
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            "-m",
            "scripts.generate_f3_catalog_from_contract",
            "--input-dir",
            str(ingested),
            "--out",
            str(catalog),
            "--min-deep",
            "4",
        ],
        cwd=repo_root,
    )

    df = pd.read_parquet(catalog)
    assert (~df["selected"]).all()
    assert df["n_deep"].eq(3).all()
