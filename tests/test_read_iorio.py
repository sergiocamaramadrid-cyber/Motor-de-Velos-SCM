"""
tests/test_read_iorio.py
------------------------
Unit-tests for src/read_iorio.py.

Covers:
  - Column-index validation (mandatory: R=0, Vobs=1, errV=2, Vgas=3, Vdisk=4, Vbul=5)
  - Auto-detection of delimiter (whitespace, tab, comma)
  - Optional sigma_V column (index 6)
  - validate_header raises ValueError on missing mandatory columns
  - FileNotFoundError on missing file
  - ValueError on too-few columns
  - read_batch tolerates one bad file and reads the rest
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.read_iorio import (
    MANDATORY_COLS,
    MIN_COLS,
    OPTIONAL_COLS,
    _detect_delimiter,
    read_galaxy,
    read_batch,
    validate_header,
)


# ---------------------------------------------------------------------------
# Helpers to write synthetic galaxy files
# ---------------------------------------------------------------------------

#: Typical data rows (6 mandatory + 1 optional sigma_V column)
_ROWS_6COL = [
    "0.50  50.0  5.0  10.0  30.0   0.0",
    "1.00  80.0  4.0  12.0  50.0   5.0",
    "2.00 110.0  6.0  14.0  70.0  10.0",
    "4.00 130.0  7.0  16.0  80.0  15.0",
]

_ROWS_7COL = [row + "  8.0" for row in _ROWS_6COL]  # extra sigma_V at index 6


def _write_ws(tmp_path: Path, rows: list[str], name: str = "NGC0001") -> Path:
    """Write a whitespace-delimited file with a comment header."""
    fp = tmp_path / f"{name}_rotmod.txt"
    fp.write_text(
        "# R[kpc]  Vobs[km/s]  errV  Vgas  Vdisk  Vbul\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return fp


def _write_tab(tmp_path: Path, rows: list[str], name: str = "NGC0002") -> Path:
    """Write a tab-delimited file."""
    fp = tmp_path / f"{name}_rotmod.txt"
    tab_rows = [r.replace("  ", "\t").replace(" ", "\t") for r in rows]
    fp.write_text("\n".join(tab_rows) + "\n", encoding="utf-8")
    return fp


def _write_csv(tmp_path: Path, rows: list[str], name: str = "NGC0003") -> Path:
    """Write a comma-delimited file."""
    fp = tmp_path / f"{name}_rotmod.txt"
    csv_rows = [",".join(r.split()) for r in rows]
    fp.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    return fp


# ---------------------------------------------------------------------------
# Column-index constants
# ---------------------------------------------------------------------------

def test_mandatory_col_indices():
    """Validate that MANDATORY_COLS maps names to the documented 0-based indices."""
    assert MANDATORY_COLS["R"]     == 0
    assert MANDATORY_COLS["Vobs"]  == 1
    assert MANDATORY_COLS["errV"]  == 2
    assert MANDATORY_COLS["Vgas"]  == 3
    assert MANDATORY_COLS["Vdisk"] == 4
    assert MANDATORY_COLS["Vbul"]  == 5
    assert len(MANDATORY_COLS) == MIN_COLS


def test_optional_sigma_v_index():
    """sigma_V must be at index 6 in OPTIONAL_COLS."""
    assert 6 in OPTIONAL_COLS
    assert OPTIONAL_COLS[6] == "sigma_V"


# ---------------------------------------------------------------------------
# Delimiter auto-detection
# ---------------------------------------------------------------------------

def test_detect_delimiter_whitespace():
    assert _detect_delimiter("0.5  50.0  5.0  10.0  30.0  0.0") == r"\s+"


def test_detect_delimiter_tab():
    assert _detect_delimiter("0.5\t50.0\t5.0\t10.0\t30.0\t0.0") == "\t"


def test_detect_delimiter_comma():
    assert _detect_delimiter("0.5,50.0,5.0,10.0,30.0,0.0") == ","


# ---------------------------------------------------------------------------
# read_galaxy – whitespace-delimited (6 mandatory cols)
# ---------------------------------------------------------------------------

def test_read_galaxy_whitespace_6col(tmp_path):
    fp = _write_ws(tmp_path, _ROWS_6COL)
    df = read_galaxy(fp)
    assert list(df.columns[:6]) == list(MANDATORY_COLS.keys())
    assert len(df) == len(_ROWS_6COL)
    assert df["R"].iloc[0] == pytest.approx(0.50)
    assert df["Vobs"].iloc[1] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# read_galaxy – tab-delimited
# ---------------------------------------------------------------------------

def test_read_galaxy_tab_delimited(tmp_path):
    fp = _write_tab(tmp_path, _ROWS_6COL)
    df = read_galaxy(fp)  # auto-detect
    assert list(df.columns[:6]) == list(MANDATORY_COLS.keys())
    assert len(df) == len(_ROWS_6COL)


def test_read_galaxy_tab_explicit_delimiter(tmp_path):
    fp = _write_tab(tmp_path, _ROWS_6COL, name="NGC0010")
    df = read_galaxy(fp, delimiter="\t")
    assert "Vobs" in df.columns


# ---------------------------------------------------------------------------
# read_galaxy – comma-delimited
# ---------------------------------------------------------------------------

def test_read_galaxy_comma_delimited(tmp_path):
    fp = _write_csv(tmp_path, _ROWS_6COL)
    df = read_galaxy(fp)  # auto-detect
    assert list(df.columns[:6]) == list(MANDATORY_COLS.keys())


# ---------------------------------------------------------------------------
# Optional sigma_V column (index 6)
# ---------------------------------------------------------------------------

def test_read_galaxy_with_sigma_v(tmp_path):
    fp = _write_ws(tmp_path, _ROWS_7COL, name="NGC0004")
    df = read_galaxy(fp)
    assert "sigma_V" in df.columns
    assert df.shape[1] == 7
    assert df["sigma_V"].iloc[0] == pytest.approx(8.0)


def test_read_galaxy_without_sigma_v(tmp_path):
    fp = _write_ws(tmp_path, _ROWS_6COL, name="NGC0005")
    df = read_galaxy(fp)
    assert "sigma_V" not in df.columns
    assert df.shape[1] == 6


# ---------------------------------------------------------------------------
# validate_header
# ---------------------------------------------------------------------------

def test_validate_header_ok(tmp_path):
    fp = _write_ws(tmp_path, _ROWS_6COL, name="NGC0006")
    df = read_galaxy(fp)
    validate_header(df, "NGC0006")  # should not raise


def test_validate_header_missing_column():
    df = pd.DataFrame({"R": [1.0], "Vobs": [50.0], "errV": [5.0]})
    with pytest.raises(ValueError, match="missing mandatory column"):
        validate_header(df, "FakeGalaxy")


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_read_galaxy_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_galaxy("/nonexistent/path/galaxy.txt")


def test_read_galaxy_too_few_columns(tmp_path):
    fp = tmp_path / "bad.txt"
    fp.write_text("0.5  50.0  5.0\n1.0  80.0  4.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least"):
        read_galaxy(fp)


# ---------------------------------------------------------------------------
# read_batch
# ---------------------------------------------------------------------------

def test_read_batch_success(tmp_path):
    paths = [
        _write_ws(tmp_path, _ROWS_6COL, name="NGC0007"),
        _write_ws(tmp_path, _ROWS_6COL, name="NGC0008"),
    ]
    result = read_batch(paths)
    assert set(result.keys()) == {"NGC0007_rotmod", "NGC0008_rotmod"}


def test_read_batch_skips_bad_file(tmp_path):
    good = _write_ws(tmp_path, _ROWS_6COL, name="NGC0009")
    bad  = tmp_path / "NGC_bad_rotmod.txt"
    bad.write_text("only two cols\n1 2\n", encoding="utf-8")
    result = read_batch([good, bad])
    # good file loaded, bad file skipped
    assert "NGC0009_rotmod" in result
    assert "NGC_bad_rotmod" not in result
