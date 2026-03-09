from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_compute_vbar_scalar_returns_array():
    from scripts.contract_utils import compute_vbar_kms

    out = compute_vbar_kms(10.0, 20.0, 0.0)
    assert hasattr(out, "shape")
    assert out.shape == ()
    assert float(out) > 0.0


def test_validate_contract_accepts_expected_columns():
    from scripts.contract_utils import CONTRACT_COLUMNS, validate_contract

    df = pd.DataFrame([{c: 1 for c in CONTRACT_COLUMNS}])
    out = validate_contract(df)
    assert list(out.columns) == CONTRACT_COLUMNS


@pytest.mark.skipif(sys.platform.startswith("win"), reason="test CLI path style tuned for unix-like CI")
def test_ingest_cli_fails_cleanly_when_data_missing(tmp_path: Path):
    out_csv = tmp_path / "out.csv"
    cmd = [
        sys.executable,
        "scripts/ingest_big_sparc_contract.py",
        "--data-root",
        str(tmp_path / "missing_data"),
        "--out",
        str(out_csv),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "No existe data_root" in proc.stderr


def test_build_sanity_summary_shape():
    from scripts.ingest_big_sparc_contract import build_sanity_summary

    df = pd.DataFrame(
        [
            {
                "galaxy": "A",
                "source_file": "A.dat",
                "n_points_curve": 10,
                "rmax_kpc": 5.0,
                "vmax_obs_kms": 100.0,
                "tail_frac": 0.7,
                "n_tail_points": 4,
                "F3_SCM": 0.52,
                "delta_f3": 0.02,
                "beta": 0.50,
                "n_beta_points": 5,
                "logSigmaHI_out": 1.2,
                "logSigmaHI_out_proxy": 1.2,
                "quality_flag_tail_ok": True,
                "quality_flag_beta_ok": True,
                "contract_version": "SPARC_MASTER_v1.0",
            }
        ]
    )
    out = build_sanity_summary(df)
    assert len(out) == 1
    assert "n_galaxies" in out.columns
