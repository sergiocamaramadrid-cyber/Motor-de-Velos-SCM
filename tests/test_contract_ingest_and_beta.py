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
    assert set(["galaxy", "n_points", "deep_slope", "f3_flag"]).issubset(catalog.columns)


def test_f3_min_deep_flag_behavior(fixture_dir, tmp_path):
    out = tmp_path / "ingest"
    ingest(fixture_dir / "galaxies.csv", fixture_dir / "rc_points.csv", out)
    p = out / "big_sparc_contract.parquet"

    c_lo = generate_catalog(p, tmp_path / "c_lo", min_deep=1, vbar_deep=500.0)
    c_hi = generate_catalog(p, tmp_path / "c_hi", min_deep=999, vbar_deep=500.0)
    assert c_lo["deep_slope"].notna().all()
    assert c_hi["deep_slope"].isna().all()


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
        [sys.executable, "-m", "scripts.generate_f3_catalog_from_contract", "--input", str(inp), "--out", str(out1), "--min-deep", "1", "--vbar-deep", "500.0"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result1.returncode == 0, result1.stderr
    assert (out1 / "f3_catalog.csv").exists()

    out2 = tmp_path / "script"
    script = str(_REPO_ROOT / "scripts" / "generate_f3_catalog_from_contract.py")
    result2 = subprocess.run(
        [sys.executable, script, "--input", str(inp), "--out", str(out2), "--min-deep", "1", "--vbar-deep", "500.0"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result2.returncode == 0, result2.stderr
    assert (out2 / "f3_catalog.csv").exists()
