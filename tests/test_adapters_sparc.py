"""
test_adapters_sparc.py — Integration tests for src.adapters.sparc_adapter
and src.adapters.big_sparc_adapter.

Uses synthetic *_rotmod.dat fixtures (written to tmp_path) to avoid any
external data dependency.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adapters.sparc_adapter import read_rotmod, ingest_sparc_dir
from src.adapters.big_sparc_adapter import (
    detect_format,
    ingest_big_sparc,
    ingest_catalog_file,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_ROTMOD_CONTENT = """\
# synthetic rotmod fixture
Rad Vobs eVobs Vgas Vdisk Vbul
0.5   40.0 2.0 10.0 35.0 0.0
1.0   55.0 2.5 15.0 45.0 0.0
3.0   70.0 3.0 20.0 55.0 0.0
8.0   75.0 3.5 18.0 50.0 0.0
20.0  65.0 4.0 12.0 35.0 0.0
"""

_ROTMOD_VBAR = """\
# rotmod with Vbar
Rad Vobs eVobs Vbar
1.0  60.0 3.0 50.0
5.0  80.0 4.0 70.0
15.0 85.0 5.0 45.0
30.0 78.0 6.0 20.0
"""


def _write_rotmod(directory: Path, name: str, content: str) -> Path:
    p = directory / f"{name}_rotmod.dat"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Tests: sparc_adapter.read_rotmod
# ---------------------------------------------------------------------------

class TestReadRotmod:
    def test_basic_parse(self, tmp_path):
        p = _write_rotmod(tmp_path, "NGC9999", _ROTMOD_CONTENT)
        gid, rc = read_rotmod(p)
        assert gid == "NGC9999"
        assert "r_kpc"    in rc.columns
        assert "vrot_kms" in rc.columns
        assert len(rc) == 5

    def test_galaxy_id_extraction(self, tmp_path):
        p = _write_rotmod(tmp_path, "MYSURV0042", _ROTMOD_CONTENT)
        gid, _ = read_rotmod(p)
        assert gid == "MYSURV0042"

    def test_vstar_derived_from_disk_bul(self, tmp_path):
        p = _write_rotmod(tmp_path, "NGC_COMP", _ROTMOD_CONTENT)
        _, rc = read_rotmod(p)
        assert "vstar_kms" in rc.columns
        assert (rc["vstar_kms"] >= 0).all()

    def test_vbar_column_used_directly(self, tmp_path):
        p = _write_rotmod(tmp_path, "NGC_VBAR", _ROTMOD_VBAR)
        _, rc = read_rotmod(p)
        assert "vbar_kms" in rc.columns
        assert "vstar_kms" not in rc.columns

    def test_error_on_missing_required_col(self, tmp_path):
        bad = "# bad file\nFoo Bar\n1.0 2.0\n"
        p = tmp_path / "BAD_rotmod.dat"
        p.write_text(bad)
        with pytest.raises(ValueError, match="missing radius"):
            read_rotmod(p)


# ---------------------------------------------------------------------------
# Tests: sparc_adapter.ingest_sparc_dir
# ---------------------------------------------------------------------------

class TestIngestSparcDir:
    def test_two_files(self, tmp_path):
        _write_rotmod(tmp_path, "NGC0001", _ROTMOD_CONTENT)
        _write_rotmod(tmp_path, "NGC0002", _ROTMOD_VBAR)
        gal, rc = ingest_sparc_dir(tmp_path)
        assert len(gal) == 2
        assert set(gal["galaxy_id"]) == {"NGC0001", "NGC0002"}
        assert "r_kpc" in rc.columns

    def test_sorted_output(self, tmp_path):
        _write_rotmod(tmp_path, "ZZZ", _ROTMOD_CONTENT)
        _write_rotmod(tmp_path, "AAA", _ROTMOD_CONTENT)
        gal, _ = ingest_sparc_dir(tmp_path)
        assert list(gal["galaxy_id"]) == ["AAA", "ZZZ"]

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_sparc_dir(tmp_path)


# ---------------------------------------------------------------------------
# Tests: big_sparc_adapter
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_directory(self, tmp_path):
        assert detect_format(tmp_path) == "rotmod_dir"

    def test_file(self, tmp_path):
        f = tmp_path / "catalog.csv"
        f.write_text("Galaxy,Rad,Vobs\nNGC1,1.0,50.0\n")
        assert detect_format(f) == "catalog_file"

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detect_format(tmp_path / "ghost.parquet")


class TestIngestCatalogFile:
    def test_csv_with_galaxy_column(self, tmp_path):
        csv_content = (
            "Galaxy,Rad,Vobs,eVobs,Vgas,Vdisk,Vbul\n"
            "NGC1,1.0,50.0,2.0,10.0,40.0,0.0\n"
            "NGC1,3.0,65.0,3.0,15.0,50.0,0.0\n"
            "NGC2,1.0,45.0,2.0, 8.0,35.0,0.0\n"
        )
        p = tmp_path / "big_sparc.csv"
        p.write_text(csv_content)
        gal, rc = ingest_catalog_file(p)
        assert set(gal["galaxy_id"]) == {"NGC1", "NGC2"}
        assert len(rc) == 3

    def test_whitespace_mrt(self, tmp_path):
        mrt_content = (
            "# comment\n"
            "Galaxy Rad Vobs eVobs Vbar\n"
            "UGC1   2.0 70.0 3.0  55.0\n"
            "UGC1   6.0 80.0 4.0  60.0\n"
        )
        p = tmp_path / "catalog.mrt"
        p.write_text(mrt_content)
        gal, rc = ingest_catalog_file(p)
        assert gal.iloc[0]["galaxy_id"] == "UGC1"
        assert "vbar_kms" in rc.columns

    def test_parquet_roundtrip(self, tmp_path):
        df = pd.DataFrame({
            "Galaxy": ["G1", "G1", "G2"],
            "Rad":    [1.0, 3.0, 2.0],
            "Vobs":   [50.0, 60.0, 55.0],
            "Vgas":   [10.0, 12.0, 8.0],
            "Vdisk":  [40.0, 48.0, 45.0],
            "Vbul":   [0.0, 0.0, 0.0],
        })
        p = tmp_path / "catalog.parquet"
        df.to_parquet(p, index=False)
        gal, rc = ingest_catalog_file(p)
        assert len(gal) == 2


class TestIngestBigSparc:
    def test_dispatch_rotmod_dir(self, tmp_path):
        _write_rotmod(tmp_path, "NGC_BIG", _ROTMOD_CONTENT)
        gal, rc = ingest_big_sparc(tmp_path)
        assert len(gal) == 1

    def test_dispatch_catalog_file(self, tmp_path):
        csv = (
            "Galaxy,Rad,Vobs,Vgas,Vdisk,Vbul\n"
            "M1,1.0,50.0,10.0,40.0,0.0\n"
        )
        p = tmp_path / "big.csv"
        p.write_text(csv)
        gal, rc = ingest_big_sparc(p)
        assert len(gal) == 1
