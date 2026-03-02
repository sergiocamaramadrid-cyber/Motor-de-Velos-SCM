"""
test_adapters_sparc_contract.py

Integration tests for the class-based SPARCAdapter and the
``src/cli/ingest.py`` CLI.

Validates:
    - SPARCAdapter.ingest() returns contract-compliant DataFrames with
      survey/instrument metadata
    - CLI end-to-end: parquet files written, correct columns, galaxy_id matches
    - BIGSPARCAdapter raises NotImplementedError (stub contract)
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.adapters.base import IngestConfig
from src.adapters.sparc import SPARCAdapter
from src.adapters.big_sparc import BIGSPARCAdapter


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_ROTMOD_CONTENT = (
    "# comment\n"
    "Rad Vobs eVobs Vgas Vdisk Vbul\n"
    "0.5 40 2 20 30 0\n"
    "1.0 50 2 25 35 0\n"
    "2.0 55 3 28 40 0\n"
)


def _write_sparc_dir(base: Path, name: str = "TEST") -> Path:
    sparc_dir = base / "SPARC"
    sparc_dir.mkdir(exist_ok=True)
    (sparc_dir / f"{name}_rotmod.dat").write_text(_ROTMOD_CONTENT)
    return sparc_dir


# ---------------------------------------------------------------------------
# SPARCAdapter unit tests
# ---------------------------------------------------------------------------

class TestSPARCAdapter:
    def test_ingest_returns_contract(self, tmp_path):
        sparc_dir = _write_sparc_dir(tmp_path)
        cfg = IngestConfig(survey="SPARC", instrument="VLA")
        adapter = SPARCAdapter()
        df_gal, df_rc = adapter.ingest(sparc_dir, cfg)

        assert "galaxy_id" in df_gal.columns
        assert "survey"    in df_gal.columns
        assert "instrument" in df_gal.columns
        assert df_gal.iloc[0]["galaxy_id"]  == "TEST"
        assert df_gal.iloc[0]["survey"]     == "SPARC"
        assert df_gal.iloc[0]["instrument"] == "VLA"

        assert "r_kpc"    in df_rc.columns
        assert "vrot_kms" in df_rc.columns
        assert len(df_rc) == 3

    def test_ingest_no_instrument(self, tmp_path):
        sparc_dir = _write_sparc_dir(tmp_path)
        cfg = IngestConfig()
        adapter = SPARCAdapter()
        df_gal, _ = adapter.ingest(sparc_dir, cfg)
        assert df_gal.iloc[0]["instrument"] == ""

    def test_ingest_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            SPARCAdapter().ingest(empty, IngestConfig())

    def test_ingest_vstar_derived(self, tmp_path):
        sparc_dir = _write_sparc_dir(tmp_path)
        cfg = IngestConfig(survey="SPARC")
        _, df_rc = SPARCAdapter().ingest(sparc_dir, cfg)
        assert "vstar_kms" in df_rc.columns
        assert (df_rc["vstar_kms"] >= 0).all()

    def test_adapter_name(self):
        assert SPARCAdapter.name == "sparc"


# ---------------------------------------------------------------------------
# BIGSPARCAdapter stub test
# ---------------------------------------------------------------------------

class TestBIGSPARCAdapterStub:
    def test_raises_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError):
            BIGSPARCAdapter().ingest(tmp_path, IngestConfig())

    def test_adapter_name(self):
        assert BIGSPARCAdapter.name == "big-sparc"


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

def _run_cli(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"Command failed: {cmd}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    return r.stdout


def test_cli_ingest_sparc(tmp_path: Path):
    sparc_dir = _write_sparc_dir(tmp_path)
    out_dir   = tmp_path / "out"

    _run_cli([
        sys.executable, "src/cli/ingest.py",
        "--survey",     "sparc",
        "--input",      str(sparc_dir),
        "--out",        str(out_dir),
    ])

    assert (out_dir / "galaxies.parquet").exists()
    assert (out_dir / "rc_points.parquet").exists()

    df_gal = pd.read_parquet(out_dir / "galaxies.parquet")
    df_rc  = pd.read_parquet(out_dir / "rc_points.parquet")

    assert "galaxy_id"   in df_gal.columns
    assert "survey"      in df_gal.columns
    assert "instrument"  in df_gal.columns
    assert df_gal.iloc[0]["galaxy_id"] == "TEST"

    assert "r_kpc"    in df_rc.columns
    assert "vrot_kms" in df_rc.columns


def test_cli_ingest_sparc_with_instrument(tmp_path: Path):
    sparc_dir = _write_sparc_dir(tmp_path)
    out_dir   = tmp_path / "out2"

    _run_cli([
        sys.executable, "src/cli/ingest.py",
        "--survey",     "sparc",
        "--input",      str(sparc_dir),
        "--out",        str(out_dir),
        "--instrument", "MeerKAT",
    ])

    df_gal = pd.read_parquet(out_dir / "galaxies.parquet")
    assert df_gal.iloc[0]["instrument"] == "MeerKAT"
