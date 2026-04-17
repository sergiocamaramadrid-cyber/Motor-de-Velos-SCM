"""
tests/test_load_scm_data.py — Tests for scripts/load_scm_data.py.

Covers:
  1. load_catalogs() — loading CSVs from a directory.
  2. print_summary() — report formatting (smoke tests).
  3. export_excel() — Excel workbook creation.
  4. main() CLI — end-to-end invocation.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.load_scm_data import (
    CATALOG_FILES,
    export_excel,
    load_catalogs,
    main,
    print_summary,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spectral(n: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "galaxy": [f"G{i:02d}" for i in range(n)],
        "n_points_raw": rng.integers(10, 30, n),
        "rmin_kpc": rng.uniform(0.1, 1.0, n),
        "rmax_kpc": rng.uniform(10, 30, n),
        "lambda_dom_kpc": rng.uniform(2, 8, n),
        "peak_freq_1perkpc": rng.uniform(0.1, 0.5, n),
        "peak_power": rng.uniform(10, 100, n),
    })


def _make_summary(n: int = 5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "galaxy": [f"G{i:02d}" for i in range(n)],
        "logMbar": rng.uniform(9.0, 11.5, n),
        "Vmax": rng.uniform(50, 250, n),
        "slope_tail": rng.uniform(-0.2, 0.2, n),
        "env_proxy": rng.uniform(0.0, 1.0, n),
    })


def _make_peaks(n: int = 5, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "galaxy": [f"G{i:02d}" for i in range(n)],
        "peak_index": np.arange(n),
        "freq_1perkpc": rng.uniform(0.05, 0.5, n),
        "power": rng.uniform(1, 50, n),
    })


def _write_catalogs(tmp_path: Path, n: int = 5) -> Path:
    """Write all three catalog CSVs to *tmp_path* and return the directory."""
    _make_spectral(n).to_csv(tmp_path / "SCM_spectral_catalog.csv", index=False)
    _make_summary(n).to_csv(tmp_path / "SCM_summary.csv", index=False)
    _make_peaks(n).to_csv(tmp_path / "SCM_peaks.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. CATALOG_FILES constant
# ---------------------------------------------------------------------------

class TestCatalogFilesConstant:
    def test_has_three_entries(self):
        assert len(CATALOG_FILES) == 3

    def test_contains_spectral_key(self):
        assert "spectral_catalog" in CATALOG_FILES

    def test_contains_summary_key(self):
        assert "summary_catalog" in CATALOG_FILES

    def test_contains_peaks_key(self):
        assert "peaks_catalog" in CATALOG_FILES

    def test_filenames_are_csv(self):
        for filename in CATALOG_FILES.values():
            assert filename.endswith(".csv")

    def test_spectral_filename(self):
        assert CATALOG_FILES["spectral_catalog"] == "SCM_spectral_catalog.csv"

    def test_summary_filename(self):
        assert CATALOG_FILES["summary_catalog"] == "SCM_summary.csv"

    def test_peaks_filename(self):
        assert CATALOG_FILES["peaks_catalog"] == "SCM_peaks.csv"


# ---------------------------------------------------------------------------
# 2. load_catalogs()
# ---------------------------------------------------------------------------

class TestLoadCatalogs:
    def test_returns_dict_with_three_keys(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        assert set(cats.keys()) == {"spectral_catalog", "summary_catalog", "peaks_catalog"}

    def test_each_value_is_dataframe(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        for df in cats.values():
            assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_written_data(self, tmp_path):
        _write_catalogs(tmp_path, n=7)
        cats = load_catalogs(tmp_path)
        for df in cats.values():
            assert len(df) == 7

    def test_spectral_catalog_columns_present(self, tmp_path):
        _write_catalogs(tmp_path)
        df = load_catalogs(tmp_path)["spectral_catalog"]
        assert "galaxy" in df.columns
        assert "lambda_dom_kpc" in df.columns

    def test_summary_catalog_columns_present(self, tmp_path):
        _write_catalogs(tmp_path)
        df = load_catalogs(tmp_path)["summary_catalog"]
        assert "galaxy" in df.columns
        assert "Vmax" in df.columns

    def test_peaks_catalog_columns_present(self, tmp_path):
        _write_catalogs(tmp_path)
        df = load_catalogs(tmp_path)["peaks_catalog"]
        assert "galaxy" in df.columns
        assert "power" in df.columns

    def test_accepts_string_path(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(str(tmp_path))
        assert len(cats) == 3

    def test_accepts_path_object(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(Path(tmp_path))
        assert len(cats) == 3

    def test_missing_spectral_raises_file_not_found(self, tmp_path):
        _write_catalogs(tmp_path)
        (tmp_path / "SCM_spectral_catalog.csv").unlink()
        with pytest.raises(FileNotFoundError, match="SCM_spectral_catalog.csv"):
            load_catalogs(tmp_path)

    def test_missing_summary_raises_file_not_found(self, tmp_path):
        _write_catalogs(tmp_path)
        (tmp_path / "SCM_summary.csv").unlink()
        with pytest.raises(FileNotFoundError, match="SCM_summary.csv"):
            load_catalogs(tmp_path)

    def test_missing_peaks_raises_file_not_found(self, tmp_path):
        _write_catalogs(tmp_path)
        (tmp_path / "SCM_peaks.csv").unlink()
        with pytest.raises(FileNotFoundError, match="SCM_peaks.csv"):
            load_catalogs(tmp_path)

    def test_missing_all_files_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_catalogs(tmp_path)

    def test_error_message_includes_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            load_catalogs(tmp_path)

    def test_data_values_preserved(self, tmp_path):
        spec = _make_spectral(n=3, seed=42)
        spec.to_csv(tmp_path / "SCM_spectral_catalog.csv", index=False)
        _make_summary(3).to_csv(tmp_path / "SCM_summary.csv", index=False)
        _make_peaks(3).to_csv(tmp_path / "SCM_peaks.csv", index=False)
        cats = load_catalogs(tmp_path)
        pd.testing.assert_frame_equal(
            cats["spectral_catalog"].reset_index(drop=True),
            spec.reset_index(drop=True),
        )

    def test_empty_csv_loads_without_error(self, tmp_path):
        """load_catalogs must not crash on empty CSVs."""
        for filename in CATALOG_FILES.values():
            pd.DataFrame(columns=["galaxy"]).to_csv(tmp_path / filename, index=False)
        cats = load_catalogs(tmp_path)
        for df in cats.values():
            assert len(df) == 0


# ---------------------------------------------------------------------------
# 3. print_summary()
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def _capture(self, catalogs):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_summary(catalogs)
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_prints_each_catalog_name(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        output = self._capture(cats)
        for name in CATALOG_FILES:
            assert name in output

    def test_prints_shape(self, tmp_path):
        _write_catalogs(tmp_path, n=6)
        cats = load_catalogs(tmp_path)
        output = self._capture(cats)
        assert "6" in output  # row count appears somewhere

    def test_prints_column_names(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        output = self._capture(cats)
        assert "galaxy" in output

    def test_runs_without_error_on_empty_catalogs(self, tmp_path):
        for filename in CATALOG_FILES.values():
            pd.DataFrame(columns=["galaxy"]).to_csv(tmp_path / filename, index=False)
        cats = load_catalogs(tmp_path)
        print_summary(cats)  # must not raise

    def test_separator_printed(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        output = self._capture(cats)
        assert "=" * 60 in output


# ---------------------------------------------------------------------------
# 4. export_excel()
# ---------------------------------------------------------------------------

class TestExportExcel:
    def test_creates_xlsx_file(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "out" / "test.xlsx"
        export_excel(cats, out)
        assert out.exists()

    def test_returns_path_object(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "catalogs.xlsx"
        result = export_excel(cats, out)
        assert isinstance(result, Path)
        assert result == out

    def test_creates_parent_directories(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "deep" / "nested" / "dir" / "catalogs.xlsx"
        export_excel(cats, out)
        assert out.exists()

    def test_excel_has_correct_sheet_names(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "catalogs.xlsx"
        export_excel(cats, out)
        xl = pd.ExcelFile(out)
        assert set(xl.sheet_names) == set(CATALOG_FILES.keys())

    def test_excel_sheets_have_correct_row_counts(self, tmp_path):
        _write_catalogs(tmp_path, n=8)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "catalogs.xlsx"
        export_excel(cats, out)
        for sheet in CATALOG_FILES:
            df = pd.read_excel(out, sheet_name=sheet)
            assert len(df) == 8

    def test_excel_spectral_sheet_contains_expected_columns(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "catalogs.xlsx"
        export_excel(cats, out)
        df = pd.read_excel(out, sheet_name="spectral_catalog")
        assert "galaxy" in df.columns

    def test_excel_summary_sheet_contains_expected_columns(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "catalogs.xlsx"
        export_excel(cats, out)
        df = pd.read_excel(out, sheet_name="summary_catalog")
        assert "galaxy" in df.columns

    def test_excel_peaks_sheet_contains_expected_columns(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "catalogs.xlsx"
        export_excel(cats, out)
        df = pd.read_excel(out, sheet_name="peaks_catalog")
        assert "galaxy" in df.columns

    def test_accepts_string_out_path(self, tmp_path):
        _write_catalogs(tmp_path)
        cats = load_catalogs(tmp_path)
        out = str(tmp_path / "catalogs.xlsx")
        result = export_excel(cats, out)
        assert Path(out).exists()

    def test_data_values_roundtrip(self, tmp_path):
        spec = _make_spectral(n=4, seed=99)
        spec.to_csv(tmp_path / "SCM_spectral_catalog.csv", index=False)
        _make_summary(4).to_csv(tmp_path / "SCM_summary.csv", index=False)
        _make_peaks(4).to_csv(tmp_path / "SCM_peaks.csv", index=False)
        cats = load_catalogs(tmp_path)
        out = tmp_path / "round.xlsx"
        export_excel(cats, out)
        roundtripped = pd.read_excel(out, sheet_name="spectral_catalog")
        assert list(roundtripped["galaxy"]) == list(spec["galaxy"])


# ---------------------------------------------------------------------------
# 5. _parse_args()
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_data_dir_required(self):
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_data_dir_parsed(self, tmp_path):
        args = _parse_args(["--data-dir", str(tmp_path)])
        assert args.data_dir == str(tmp_path)

    def test_excel_defaults_to_none(self, tmp_path):
        args = _parse_args(["--data-dir", str(tmp_path)])
        assert args.excel is None

    def test_excel_parsed(self, tmp_path):
        out = str(tmp_path / "out.xlsx")
        args = _parse_args(["--data-dir", str(tmp_path), "--excel", out])
        assert args.excel == out


# ---------------------------------------------------------------------------
# 6. main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_returns_dict(self, tmp_path):
        _write_catalogs(tmp_path)
        result = main(["--data-dir", str(tmp_path)])
        assert isinstance(result, dict)

    def test_returns_catalogs_key(self, tmp_path):
        _write_catalogs(tmp_path)
        result = main(["--data-dir", str(tmp_path)])
        assert "catalogs" in result

    def test_returns_data_dir_key(self, tmp_path):
        _write_catalogs(tmp_path)
        result = main(["--data-dir", str(tmp_path)])
        assert "data_dir" in result

    def test_returns_excel_path_key(self, tmp_path):
        _write_catalogs(tmp_path)
        result = main(["--data-dir", str(tmp_path)])
        assert "excel_path" in result

    def test_catalogs_value_is_dict_of_dataframes(self, tmp_path):
        _write_catalogs(tmp_path)
        result = main(["--data-dir", str(tmp_path)])
        assert isinstance(result["catalogs"], dict)
        for df in result["catalogs"].values():
            assert isinstance(df, pd.DataFrame)

    def test_excel_path_none_when_not_requested(self, tmp_path):
        _write_catalogs(tmp_path)
        result = main(["--data-dir", str(tmp_path)])
        assert result["excel_path"] is None

    def test_excel_path_set_when_requested(self, tmp_path):
        _write_catalogs(tmp_path)
        out = tmp_path / "catalogs.xlsx"
        result = main(["--data-dir", str(tmp_path), "--excel", str(out)])
        assert result["excel_path"] == str(out)

    def test_excel_file_created_when_requested(self, tmp_path):
        _write_catalogs(tmp_path)
        out = tmp_path / "catalogs.xlsx"
        main(["--data-dir", str(tmp_path), "--excel", str(out)])
        assert out.exists()

    def test_missing_data_dir_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "no_such_dir"
        with pytest.raises(FileNotFoundError):
            main(["--data-dir", str(missing)])

    def test_data_dir_in_return_dict_is_string(self, tmp_path):
        _write_catalogs(tmp_path)
        result = main(["--data-dir", str(tmp_path)])
        assert isinstance(result["data_dir"], str)

    def test_catalogs_have_three_entries(self, tmp_path):
        _write_catalogs(tmp_path)
        result = main(["--data-dir", str(tmp_path)])
        assert len(result["catalogs"]) == 3

    def test_row_count_in_returned_catalogs(self, tmp_path):
        _write_catalogs(tmp_path, n=9)
        result = main(["--data-dir", str(tmp_path)])
        for df in result["catalogs"].values():
            assert len(df) == 9

    def test_no_excel_no_file_created(self, tmp_path):
        _write_catalogs(tmp_path)
        before = list(tmp_path.glob("*.xlsx"))
        main(["--data-dir", str(tmp_path)])
        after = list(tmp_path.glob("*.xlsx"))
        assert before == after

    def test_excel_export_sheets_match_catalogs(self, tmp_path):
        _write_catalogs(tmp_path)
        out = tmp_path / "out.xlsx"
        result = main(["--data-dir", str(tmp_path), "--excel", str(out)])
        xl = pd.ExcelFile(result["excel_path"])
        assert set(xl.sheet_names) == set(CATALOG_FILES.keys())
