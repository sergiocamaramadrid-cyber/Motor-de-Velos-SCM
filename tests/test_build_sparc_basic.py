"""
tests/test_build_sparc_basic.py — Unit tests for scripts/build_sparc_basic.py.

Uses synthetic in-memory data so no real SPARC download is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_sparc_basic import (  # noqa: E402
    _ci_lookup,
    _load_sparc_table,
    build_sparc_basic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparc_csv(tmp_path: Path, include_coords: bool = False) -> Path:
    """Write a minimal synthetic SPARC galaxy table and return the path."""
    data = {
        "Galaxy": ["NGC0300", "NGC0891", "NGC2403", "UGC04325", "NGC5055"],
        "T": [7, 3, 6, 8, 4],
        "D": [2.1, 9.6, 3.2, 11.5, 8.0],
        "e_D": [0.2, 1.0, 0.3, 1.2, 0.8],
        "Inc": [42.0, 88.0, 63.0, 41.0, 59.0],
        "e_Inc": [2.0, 2.0, 2.0, 3.0, 2.0],
        "L36": [1.2, 8.5, 4.3, 0.9, 15.0],
        "e_L36": [0.1, 0.5, 0.3, 0.1, 1.0],
        "Re": [2.5, 3.1, 5.0, 1.2, 4.8],
        "MHI": [0.3, 0.8, 2.1, 0.5, 1.4],
        "Vflat": [80.0, 220.0, 135.0, 90.0, 185.0],
        "e_Vflat": [5.0, 10.0, 5.0, 8.0, 7.0],
        "Q": [2, 1, 1, 3, 1],
        "Ref": [1, 1, 1, 1, 1],
    }
    if include_coords:
        data["RAdeg"] = [13.72, 35.64, 114.21, 122.00, 198.96]
        data["DEdeg"] = [-37.69, 42.35, 65.60, 50.00, 42.03]

    df = pd.DataFrame(data)
    out = tmp_path / "SPARC_Lelli2016c.csv"
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Tests for _ci_lookup
# ---------------------------------------------------------------------------

class TestCiLookup:
    def test_exact_match(self):
        assert _ci_lookup(["Galaxy", "RA", "Dec"], ["RA"]) == "RA"

    def test_case_insensitive(self):
        assert _ci_lookup(["galaxy", "ra", "dec"], ["RA"]) == "ra"

    def test_returns_first_candidate(self):
        result = _ci_lookup(["Galaxy", "RAdeg", "RA"], ["RAdeg", "RA"])
        assert result == "RAdeg"

    def test_not_found_returns_none(self):
        assert _ci_lookup(["Galaxy", "T", "D"], ["RA", "RAdeg"]) is None


# ---------------------------------------------------------------------------
# Tests for _load_sparc_table
# ---------------------------------------------------------------------------

class TestLoadSparcTable:
    def test_loads_csv(self, tmp_path):
        _make_sparc_csv(tmp_path)
        df = _load_sparc_table(tmp_path)
        assert "Galaxy" in df.columns
        assert len(df) == 5

    def test_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SPARC galaxy table not found"):
            _load_sparc_table(tmp_path)


# ---------------------------------------------------------------------------
# Tests for build_sparc_basic
# ---------------------------------------------------------------------------

class TestBuildSparcBasic:
    def test_with_coords_in_table(self, tmp_path):
        """When RA/Dec are in the source CSV they pass straight through."""
        _make_sparc_csv(tmp_path, include_coords=True)
        out = tmp_path / "sparc_basic.csv"
        result = build_sparc_basic(tmp_path, out, resolve_coords=False)
        assert "ra" in result.columns
        assert "dec" in result.columns
        assert result["ra"].notna().all()
        assert result["dec"].notna().all()

    def test_without_coords_no_resolve(self, tmp_path):
        """When coords are missing and resolve is off, ra/dec are NaN."""
        _make_sparc_csv(tmp_path, include_coords=False)
        out = tmp_path / "sparc_basic.csv"
        result = build_sparc_basic(tmp_path, out, resolve_coords=False)
        assert "ra" in result.columns
        assert result["ra"].isna().all()

    def test_output_file_created(self, tmp_path):
        _make_sparc_csv(tmp_path)
        out = tmp_path / "sparc_basic.csv"
        build_sparc_basic(tmp_path, out, resolve_coords=False)
        assert out.exists()
        df = pd.read_csv(out)
        assert "galaxy" in df.columns

    def test_required_columns_present(self, tmp_path):
        _make_sparc_csv(tmp_path, include_coords=True)
        out = tmp_path / "sparc_basic.csv"
        result = build_sparc_basic(tmp_path, out, resolve_coords=False)
        for col in ["galaxy", "ra", "dec", "D", "Vflat", "Inc"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_galaxy_column_lowercased(self, tmp_path):
        """Output uses lowercase 'galaxy', not 'Galaxy'."""
        _make_sparc_csv(tmp_path, include_coords=True)
        out = tmp_path / "sparc_basic.csv"
        result = build_sparc_basic(tmp_path, out, resolve_coords=False)
        assert "galaxy" in result.columns
        assert "Galaxy" not in result.columns

    def test_n_rows_matches_input(self, tmp_path):
        _make_sparc_csv(tmp_path, include_coords=True)
        out = tmp_path / "sparc_basic.csv"
        result = build_sparc_basic(tmp_path, out, resolve_coords=False)
        assert len(result) == 5

    def test_raises_if_no_sparc_table(self, tmp_path):
        out = tmp_path / "sparc_basic.csv"
        with pytest.raises(FileNotFoundError):
            build_sparc_basic(tmp_path, out, resolve_coords=False)

    def test_output_dir_created(self, tmp_path):
        """Output parent directory is created automatically."""
        _make_sparc_csv(tmp_path, include_coords=True)
        nested = tmp_path / "nested" / "out.csv"
        build_sparc_basic(tmp_path, nested, resolve_coords=False)
        assert nested.exists()

    def test_idempotent(self, tmp_path):
        """Running twice produces the same result."""
        _make_sparc_csv(tmp_path, include_coords=True)
        out = tmp_path / "sparc_basic.csv"
        r1 = build_sparc_basic(tmp_path, out, resolve_coords=False)
        r2 = build_sparc_basic(tmp_path, out, resolve_coords=False)
        pd.testing.assert_frame_equal(r1, r2)
