from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.contract_utils import CONTRACT_COLUMNS, compute_vbar_kms, read_table, validate_contract
from scripts.ingest_big_sparc_contract import ingest
from scripts.generate_f3_catalog_from_contract import generate_catalog

_REPO_ROOT = Path(__file__).parent.parent


def _make_galaxies() -> pd.DataFrame:
    return pd.DataFrame({"galaxy": ["TG000", "TG001"]})


def _make_rc_points(include_vbar: bool = False) -> pd.DataFrame:
    rows = []
    for g, base in [("TG000", 85.0), ("TG001", 95.0)]:
        for i, r in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            vobs = base + i
            row = {"galaxy": g, "r_kpc": r, "vobs_kms": vobs, "vobs_err_kms": 5.0}
            if include_vbar:
                row["vbar_kms"] = 0.8 * vobs
            else:
                row["v_gas"] = 0.3 * vobs
                row["v_disk"] = 0.7 * vobs
                row["v_bul"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def fixture_dir(tmp_path) -> Path:
    root = tmp_path / "fx"
    root.mkdir()
    _make_galaxies().to_csv(root / "galaxies.csv", index=False)
    _make_rc_points().to_csv(root / "rc_points.csv", index=False)
    return root


def test_contract_utils_basics(tmp_path):
    assert float(compute_vbar_kms(3.0, 4.0)) == pytest.approx(5.0)
    p = tmp_path / "a.csv"
    df = pd.DataFrame({"galaxy": ["A"]})
    df.to_csv(p, index=False)
    pd.testing.assert_frame_equal(read_table(p), df)
    with pytest.raises(ValueError):
        validate_contract(pd.DataFrame({"galaxy": ["A"]}))


def test_ingest_and_f3_e2e(fixture_dir, tmp_path):
    out = tmp_path / "ingest"
    ingested = ingest(fixture_dir / "galaxies.csv", fixture_dir / "rc_points.csv", out)
    assert list(ingested.columns) == CONTRACT_COLUMNS
    assert (out / "big_sparc_contract.parquet").exists()

    cat_out = tmp_path / "f3"
    catalog = generate_catalog(out / "big_sparc_contract.parquet", cat_out, min_deep=1, vbar_deep=500.0)
    assert (cat_out / "f3_catalog.csv").exists()
    assert set(
        [
            "galaxy",
            "n_points",
            "deep_slope",
            "F3_slope",
            "expected_slope",
            "delta_f3",
            "n_tail_points",
            "tail_points_used",
            "tail_r_min",
            "tail_r_max",
            "tail_points_used",
            "f3_flag",
        ]
    ).issubset(catalog.columns)
    assert np.allclose(
        catalog["delta_f3"].values,
        catalog["deep_slope"].values - catalog["expected_slope"].values,
        atol=1e-4,
        equal_nan=True,
    )
    assert (catalog["n_tail_points"] == 5).all()
    assert (catalog["tail_points_used"] == 5).all()


def test_f3_min_deep_flag_behavior(fixture_dir, tmp_path):
    out = tmp_path / "ingest"
    ingest(fixture_dir / "galaxies.csv", fixture_dir / "rc_points.csv", out)
    p = out / "big_sparc_contract.parquet"

    c_lo = generate_catalog(p, tmp_path / "c_lo", min_deep=1, vbar_deep=500.0)
    c_hi = generate_catalog(p, tmp_path / "c_hi", min_deep=999, vbar_deep=500.0)
    assert c_lo["deep_slope"].notna().all()
    assert c_hi["deep_slope"].isna().all()


def test_f3_tail_points_reflect_available_deep_points(tmp_path):
    contract = tmp_path / "contract.csv"
    out_dir = tmp_path / "f3_out"
    pd.DataFrame(
        [
            {"galaxy": "G_SMALL", "r_kpc": 1.0, "vobs_kms": 20.0, "vobs_err_kms": 1.0, "vbar_kms": 10.0},
            {"galaxy": "G_SMALL", "r_kpc": 2.0, "vobs_kms": 25.0, "vobs_err_kms": 1.0, "vbar_kms": 20.0},
            {"galaxy": "G_SMALL", "r_kpc": 3.0, "vobs_kms": 30.0, "vobs_err_kms": 1.0, "vbar_kms": 30.0},
        ]
    ).to_csv(contract, index=False)

    catalog = generate_catalog(contract, out_dir, min_deep=3, vbar_deep=500.0)
    assert int(catalog.loc[0, "n_tail_points"]) == 3
    assert int(catalog.loc[0, "tail_points_used"]) == 5
    assert float(catalog.loc[0, "tail_r_min"]) == pytest.approx(1.0)
    assert float(catalog.loc[0, "tail_r_max"]) == pytest.approx(3.0)


def test_f3_tail_points_cli_and_api_override(fixture_dir, tmp_path):
    out = tmp_path / "ingest"
    ingest(fixture_dir / "galaxies.csv", fixture_dir / "rc_points.csv", out)
    p = out / "big_sparc_contract.parquet"

    catalog = generate_catalog(p, tmp_path / "custom_tail", min_deep=1, vbar_deep=500.0, tail_points=4)
    assert (catalog["n_tail_points"] == 4).all()
    assert (catalog["tail_points_used"] == 4).all()


def test_ingest_cli_module_and_script(fixture_dir, tmp_path):
    out1 = tmp_path / "mod"
    result1 = subprocess.run(
        [sys.executable, "-m", "scripts.ingest_big_sparc_contract", "--galaxies", str(fixture_dir / "galaxies.csv"), "--rc-points", str(fixture_dir / "rc_points.csv"), "--out", str(out1)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result1.returncode == 0, result1.stderr
    assert (out1 / "big_sparc_contract.parquet").exists()

    out2 = tmp_path / "script"
    script = str(_REPO_ROOT / "scripts" / "ingest_big_sparc_contract.py")
    result2 = subprocess.run(
        [sys.executable, script, "--galaxies", str(fixture_dir / "galaxies.csv"), "--rc-points", str(fixture_dir / "rc_points.csv"), "--out", str(out2)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result2.returncode == 0, result2.stderr
    assert (out2 / "big_sparc_contract.parquet").exists()


def test_f3_cli_module_and_script(fixture_dir, tmp_path):
    ingest_out = tmp_path / "ingest"
    ingest(fixture_dir / "galaxies.csv", fixture_dir / "rc_points.csv", ingest_out)
    inp = ingest_out / "big_sparc_contract.parquet"

    out1 = tmp_path / "mod"
    result1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.generate_f3_catalog_from_contract",
            "--input",
            str(inp),
            "--out",
            str(out1),
            "--min-deep",
            "1",
            "--vbar-deep",
            "500.0",
            "--tail-points",
            "4",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result1.returncode == 0, result1.stderr
    assert (out1 / "f3_catalog.csv").exists()
    c1 = pd.read_csv(out1 / "f3_catalog.csv")
    assert (c1["tail_points_used"] == 4).all()

    out2 = tmp_path / "script"
    script = str(_REPO_ROOT / "scripts" / "generate_f3_catalog_from_contract.py")
    result2 = subprocess.run(
        [
            sys.executable,
            script,
            "--input",
            str(inp),
            "--out",
            str(out2),
            "--min-deep",
            "1",
            "--vbar-deep",
            "500.0",
            "--tail-points",
            "4",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result2.returncode == 0, result2.stderr
    assert (out2 / "f3_catalog.csv").exists()
    c2 = pd.read_csv(out2 / "f3_catalog.csv")
    assert (c2["tail_points_used"] == 4).all()


def test_f3_cli_tail_points_must_be_at_least_three(fixture_dir, tmp_path):
    ingest_out = tmp_path / "ingest"
    ingest(fixture_dir / "galaxies.csv", fixture_dir / "rc_points.csv", ingest_out)
    inp = ingest_out / "big_sparc_contract.parquet"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.generate_f3_catalog_from_contract",
            "--input",
            str(inp),
            "--out",
            str(tmp_path / "invalid"),
            "--tail-points",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode != 0
    assert "--tail-points must be >= 3" in (result.stderr + result.stdout)
