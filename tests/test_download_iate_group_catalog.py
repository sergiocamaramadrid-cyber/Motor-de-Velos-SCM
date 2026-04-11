"""
tests/test_download_iate_group_catalog.py — Tests for download_iate_group_catalog.py.

All network calls are mocked; no real download is performed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.download_iate_group_catalog import (
    CATALOG_URL,
    DEFAULT_OUT,
    _FALLBACK_COLUMNS,
    _download_text,
    download_iate_group_catalog,
    main,
    parse_dat,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SAMPLE_WITH_HEADER = textwrap.dedent("""\
    # GroupID RA_deg Dec_deg z N_members sigma_v_kms log_Mh_Msun R200_Mpc
    1  150.23  2.45  0.081  5  312.4  13.2  0.92
    2  210.10  -1.33  0.103  3  180.0  12.8  0.71
    3  55.00   30.12  0.057  12  450.3  14.1  1.40
""")

_SAMPLE_WITHOUT_HEADER = textwrap.dedent("""\
    1  150.23  2.45  0.081  5  312.4  13.2  0.92
    2  210.10  -1.33  0.103  3  180.0  12.8  0.71
""")

_SAMPLE_CUSTOM_HEADER = textwrap.dedent("""\
    # ID RA DEC Redshift Ngal sigma logMh R200
    10  100.0  5.0  0.10  7  200.0  13.5  1.00
""")


# ---------------------------------------------------------------------------
# parse_dat — with header
# ---------------------------------------------------------------------------

class TestParseDatWithHeader:
    def test_returns_dataframe(self):
        df = parse_dat(_SAMPLE_WITH_HEADER)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = parse_dat(_SAMPLE_WITH_HEADER)
        assert len(df) == 3

    def test_column_names_from_header(self):
        df = parse_dat(_SAMPLE_WITH_HEADER)
        assert list(df.columns) == [
            "GroupID", "RA_deg", "Dec_deg", "z",
            "N_members", "sigma_v_kms", "log_Mh_Msun", "R200_Mpc",
        ]

    def test_numeric_types(self):
        df = parse_dat(_SAMPLE_WITH_HEADER)
        assert pd.api.types.is_numeric_dtype(df["RA_deg"])
        assert pd.api.types.is_numeric_dtype(df["z"])

    def test_first_row_values(self):
        df = parse_dat(_SAMPLE_WITH_HEADER)
        assert df["GroupID"].iloc[0] == 1
        assert abs(df["RA_deg"].iloc[0] - 150.23) < 1e-9
        assert df["N_members"].iloc[0] == 5

    def test_negative_dec(self):
        df = parse_dat(_SAMPLE_WITH_HEADER)
        assert df["Dec_deg"].iloc[1] < 0


# ---------------------------------------------------------------------------
# parse_dat — without header (fallback column names)
# ---------------------------------------------------------------------------

class TestParseDatWithoutHeader:
    def test_returns_dataframe(self):
        df = parse_dat(_SAMPLE_WITHOUT_HEADER)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = parse_dat(_SAMPLE_WITHOUT_HEADER)
        assert len(df) == 2

    def test_fallback_column_names_applied(self):
        df = parse_dat(_SAMPLE_WITHOUT_HEADER)
        assert list(df.columns) == _FALLBACK_COLUMNS

    def test_numeric_types(self):
        df = parse_dat(_SAMPLE_WITHOUT_HEADER)
        assert pd.api.types.is_numeric_dtype(df["RA_deg"])

    def test_values(self):
        df = parse_dat(_SAMPLE_WITHOUT_HEADER)
        assert abs(df["z"].iloc[0] - 0.081) < 1e-9


# ---------------------------------------------------------------------------
# parse_dat — custom header
# ---------------------------------------------------------------------------

class TestParseDatCustomHeader:
    def test_custom_column_names(self):
        df = parse_dat(_SAMPLE_CUSTOM_HEADER)
        assert list(df.columns) == [
            "ID", "RA", "DEC", "Redshift", "Ngal", "sigma", "logMh", "R200"
        ]

    def test_row_count(self):
        df = parse_dat(_SAMPLE_CUSTOM_HEADER)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# parse_dat — file-like input
# ---------------------------------------------------------------------------

class TestParseDatFilelike:
    def test_accepts_stringio(self):
        from io import StringIO
        df = parse_dat(StringIO(_SAMPLE_WITH_HEADER))
        assert len(df) == 3


# ---------------------------------------------------------------------------
# parse_dat — error handling
# ---------------------------------------------------------------------------

class TestParseDatErrors:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_dat("")

    def test_only_comments_raises(self):
        with pytest.raises(ValueError, match="no data"):
            parse_dat("# GroupID RA Dec\n# another comment\n")

    def test_blank_lines_only_raises(self):
        with pytest.raises(ValueError):
            parse_dat("\n\n   \n")


# ---------------------------------------------------------------------------
# parse_dat — extra columns (no matching fallback)
# ---------------------------------------------------------------------------

class TestParseDatExtraColumns:
    def test_generic_column_names_when_count_mismatch(self):
        content = "1  2  3  4  5  6  7  8  9\n"  # 9 cols, fallback has 8
        df = parse_dat(content)
        assert list(df.columns) == [f"col{i}" for i in range(9)]


# ---------------------------------------------------------------------------
# _download_text
# ---------------------------------------------------------------------------

class TestDownloadText:
    def test_success(self):
        mock_response = MagicMock()
        mock_response.read.return_value = b"# GroupID\n1 150.0\n"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _download_text("http://example.com/test.dat")
        assert "GroupID" in result

    def test_retries_on_failure(self):
        import urllib.error

        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Could not download"):
                _download_text("http://example.com/test.dat", retries=2)


# ---------------------------------------------------------------------------
# download_iate_group_catalog — mocked network
# ---------------------------------------------------------------------------

class TestDownloadIateGroupCatalog:
    def test_creates_csv_file(self, tmp_path):
        out = tmp_path / "iate_group_catalog.csv"
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            return_value=_SAMPLE_WITH_HEADER,
        ):
            df = download_iate_group_catalog(out)
        assert out.exists()
        assert len(df) == 3

    def test_returns_dataframe(self, tmp_path):
        out = tmp_path / "catalog.csv"
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            return_value=_SAMPLE_WITH_HEADER,
        ):
            df = download_iate_group_catalog(out)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_csv_readable(self, tmp_path):
        out = tmp_path / "catalog.csv"
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            return_value=_SAMPLE_WITH_HEADER,
        ):
            download_iate_group_catalog(out)
        df2 = pd.read_csv(out)
        assert list(df2.columns) == [
            "GroupID", "RA_deg", "Dec_deg", "z",
            "N_members", "sigma_v_kms", "log_Mh_Msun", "R200_Mpc",
        ]

    def test_skips_download_if_file_exists(self, tmp_path):
        out = tmp_path / "catalog.csv"
        # Pre-write a CSV
        existing = pd.DataFrame({"GroupID": [99], "RA_deg": [1.0]})
        existing.to_csv(out, index=False)

        with patch(
            "scripts.download_iate_group_catalog._download_text"
        ) as mock_dl:
            df = download_iate_group_catalog(out)
        mock_dl.assert_not_called()
        assert df["GroupID"].iloc[0] == 99

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "subdir" / "catalog.csv"
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            return_value=_SAMPLE_WITH_HEADER,
        ):
            download_iate_group_catalog(out)
        assert out.exists()

    def test_download_error_raises(self, tmp_path):
        out = tmp_path / "catalog.csv"
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            side_effect=RuntimeError("network error"),
        ):
            with pytest.raises(RuntimeError, match="network error"):
                download_iate_group_catalog(out)


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        out = str(tmp_path / "catalog.csv")
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            return_value=_SAMPLE_WITH_HEADER,
        ):
            result = main(["--out", out])
        assert set(result.keys()) == {"out_path", "n_groups", "columns"}

    def test_n_groups(self, tmp_path):
        out = str(tmp_path / "catalog.csv")
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            return_value=_SAMPLE_WITH_HEADER,
        ):
            result = main(["--out", out])
        assert result["n_groups"] == 3

    def test_columns_in_result(self, tmp_path):
        out = str(tmp_path / "catalog.csv")
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            return_value=_SAMPLE_WITH_HEADER,
        ):
            result = main(["--out", out])
        assert "GroupID" in result["columns"]

    def test_default_out_path_used(self, tmp_path, monkeypatch):
        # Override DEFAULT_OUT to point inside tmp_path
        monkeypatch.chdir(tmp_path)
        out_path = tmp_path / "data" / "iate" / "iate_group_catalog.csv"
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            return_value=_SAMPLE_WITH_HEADER,
        ):
            result = main([])
        assert result["n_groups"] == 3

    def test_exit_on_network_error(self, tmp_path):
        out = str(tmp_path / "catalog.csv")
        with patch(
            "scripts.download_iate_group_catalog._download_text",
            side_effect=RuntimeError("network error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main(["--out", out])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_catalog_url(self):
        assert "iate.conicet.unc.edu.ar" in CATALOG_URL
        assert "FINAL_Group.dat" in CATALOG_URL

    def test_fallback_columns_length(self):
        assert len(_FALLBACK_COLUMNS) == 8

    def test_default_out_is_path(self):
        assert isinstance(DEFAULT_OUT, Path)
