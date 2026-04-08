"""
tests/test_build_sample_csv.py — Tests for scripts/build_sample_csv.py.

Covers:
  - load_lt_global: happy path, missing columns, too-few rows
  - load_predictions: happy path, missing columns
  - collect_rmax: populated dir, empty dir, missing dir
  - standardise: normal, constant series
  - build_catalog: full merge, no predictions, no rot_dir
  - Output CSV: column contract, row count, dtype
  - main(): smoke test, output message
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------
import importlib
import sys

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
build_sample_csv = importlib.import_module("build_sample_csv")

load_lt_global = build_sample_csv.load_lt_global
load_predictions = build_sample_csv.load_predictions
collect_rmax = build_sample_csv.collect_rmax
standardise = build_sample_csv.standardise
build_catalog = build_sample_csv.build_catalog
main = build_sample_csv.main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LT_GLOBAL_CSV = textwrap.dedent(
    """\
    galaxy_id,logM,logVobs,log_gbar,log_j
    CVnIdwA,6.20,1.1139,-12.4609,0.7160
    DDO43,7.60,1.4771,-12.0964,1.6232
    DDO46,7.80,1.5798,-11.7572,1.7616
    DDO47,8.10,1.6721,-11.8542,2.0871
    DDO50,8.20,1.5682,-12.1246,1.9106
    """
)

PREDICTIONS_CSV = textwrap.dedent(
    """\
    galaxy_id,logVobs,logV_btfr,logV_interp,residual_btfr,residual_interp
    CVnIdwA,1.1139,1.2951,1.0048,0.1812,-0.1091
    DDO43,1.4771,1.6451,1.3680,0.1680,-0.1091
    DDO46,1.5798,1.6951,1.4706,0.1153,-0.1092
    DDO47,1.6721,1.7701,1.5630,0.0980,-0.1091
    DDO50,1.5682,1.7951,1.4591,0.2269,-0.1091
    """
)


@pytest.fixture
def lt_global_csv(tmp_path) -> Path:
    p = tmp_path / "little_things_global.csv"
    p.write_text(LT_GLOBAL_CSV)
    return p


@pytest.fixture
def predictions_csv(tmp_path) -> Path:
    p = tmp_path / "predictions.csv"
    p.write_text(PREDICTIONS_CSV)
    return p


@pytest.fixture
def rot_dir(tmp_path) -> Path:
    d = tmp_path / "lt_oh2015"
    d.mkdir()
    (d / "DDO46_rot.csv").write_text("r_kpc,Vbary_kms\n0.5,20.0\n1.5,35.0\n2.5,42.0\n")
    (d / "DDO47_rot.csv").write_text("r_kpc,Vbary_kms\n0.3,18.0\n2.0,38.0\n")
    return d


# ---------------------------------------------------------------------------
# load_lt_global
# ---------------------------------------------------------------------------

class TestLoadLtGlobal:
    def test_returns_dataframe(self, lt_global_csv):
        df = load_lt_global(lt_global_csv)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, lt_global_csv):
        df = load_lt_global(lt_global_csv)
        assert set(df.columns) == {"galaxy", "logM", "log_j"}

    def test_row_count(self, lt_global_csv):
        df = load_lt_global(lt_global_csv)
        assert len(df) == 5

    def test_galaxy_id_renamed(self, lt_global_csv):
        df = load_lt_global(lt_global_csv)
        assert "galaxy" in df.columns
        assert "galaxy_id" not in df.columns

    def test_logM_dtype(self, lt_global_csv):
        df = load_lt_global(lt_global_csv)
        assert pd.api.types.is_float_dtype(df["logM"])

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("galaxy_id,logM\nA,7.0\nB,8.0\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_lt_global(p)

    def test_too_few_rows_raises(self, tmp_path):
        p = tmp_path / "one_row.csv"
        p.write_text("galaxy_id,logM,log_j\nA,7.0,1.5\n")
        with pytest.raises(ValueError, match="at least 2"):
            load_lt_global(p)


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------

class TestLoadPredictions:
    def test_returns_dataframe(self, predictions_csv):
        df = load_predictions(predictions_csv)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, predictions_csv):
        df = load_predictions(predictions_csv)
        assert set(df.columns) == {"galaxy", "delta_f3"}

    def test_row_count(self, predictions_csv):
        df = load_predictions(predictions_csv)
        assert len(df) == 5

    def test_values_match_residual_btfr(self, predictions_csv):
        df = load_predictions(predictions_csv)
        assert pytest.approx(df.loc[df["galaxy"] == "CVnIdwA", "delta_f3"].iloc[0], abs=1e-4) == 0.1812

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad_pred.csv"
        p.write_text("galaxy_id,logVobs\nA,1.5\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_predictions(p)


# ---------------------------------------------------------------------------
# collect_rmax
# ---------------------------------------------------------------------------

class TestCollectRmax:
    def test_returns_dict(self, rot_dir):
        result = collect_rmax(rot_dir)
        assert isinstance(result, dict)

    def test_correct_galaxies(self, rot_dir):
        result = collect_rmax(rot_dir)
        assert set(result.keys()) == {"DDO46", "DDO47"}

    def test_rmax_values(self, rot_dir):
        result = collect_rmax(rot_dir)
        assert pytest.approx(result["DDO46"], abs=1e-6) == 2.5
        assert pytest.approx(result["DDO47"], abs=1e-6) == 2.0

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty_rot"
        d.mkdir()
        assert collect_rmax(d) == {}

    def test_nonexistent_dir(self, tmp_path):
        assert collect_rmax(tmp_path / "no_such_dir") == {}

    def test_skips_file_without_r_kpc(self, tmp_path):
        d = tmp_path / "rot"
        d.mkdir()
        (d / "X99_rot.csv").write_text("radius,V\n1.0,20.0\n")
        result = collect_rmax(d)
        assert "X99" not in result


# ---------------------------------------------------------------------------
# standardise
# ---------------------------------------------------------------------------

class TestStandardise:
    def test_mean_zero(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = standardise(s)
        assert pytest.approx(z.mean(), abs=1e-10) == 0.0

    def test_std_one(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = standardise(s)
        assert pytest.approx(z.std(ddof=1), abs=1e-10) == 1.0

    def test_constant_series_returns_nan(self):
        s = pd.Series([3.0, 3.0, 3.0])
        z = standardise(s)
        assert z.isna().all()

    def test_preserves_index(self):
        s = pd.Series([10.0, 20.0, 30.0], index=[5, 6, 7])
        z = standardise(s)
        assert list(z.index) == [5, 6, 7]


# ---------------------------------------------------------------------------
# build_catalog — output contract
# ---------------------------------------------------------------------------

class TestBuildCatalog:
    EXPECTED_COLS = {"galaxy", "logM", "delta_mass_std", "slope_tail", "Rmax_kpc", "delta_f3"}

    def test_returns_dataframe(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        df = build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        assert isinstance(df, pd.DataFrame)

    def test_column_contract(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        df = build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        assert set(df.columns) == self.EXPECTED_COLS

    def test_row_count_matches_lt_global(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        df = build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        assert len(df) == 5

    def test_delta_mass_std_is_z_score(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        df = build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        assert pytest.approx(df["delta_mass_std"].mean(), abs=1e-10) == 0.0
        assert pytest.approx(df["delta_mass_std"].std(ddof=1), abs=1e-10) == 1.0

    def test_delta_f3_equals_slope_tail_minus_half(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        df = build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        for _, row in df.dropna(subset=["slope_tail"]).iterrows():
            assert pytest.approx(row["delta_f3"], abs=1e-9) == row["slope_tail"] - 0.5

    def test_rmax_populated_for_known_galaxies(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        df = build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        row46 = df.loc[df["galaxy"] == "DDO46", "Rmax_kpc"]
        row47 = df.loc[df["galaxy"] == "DDO47", "Rmax_kpc"]
        assert pytest.approx(float(row46.iloc[0]), abs=1e-6) == 2.5
        assert pytest.approx(float(row47.iloc[0]), abs=1e-6) == 2.0

    def test_rmax_nan_for_galaxies_without_rot_curve(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        df = build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        row = df.loc[df["galaxy"] == "CVnIdwA", "Rmax_kpc"]
        assert pd.isna(float(row.iloc[0]))

    def test_writes_csv_to_disk(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "sub" / "cat.csv"
        build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        assert out.exists()

    def test_csv_readable_and_matches_dataframe(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        df = build_catalog(lt_global_csv, predictions_csv, rot_dir, out)
        df2 = pd.read_csv(out)
        assert list(df.columns) == list(df2.columns)
        assert len(df) == len(df2)

    def test_no_predictions_slope_tail_all_nan(self, lt_global_csv, rot_dir, tmp_path):
        out = tmp_path / "cat_nopred.csv"
        df = build_catalog(lt_global_csv, None, rot_dir, out)
        assert df["slope_tail"].isna().all()

    def test_no_predictions_delta_f3_all_nan(self, lt_global_csv, rot_dir, tmp_path):
        out = tmp_path / "cat_nopred.csv"
        df = build_catalog(lt_global_csv, None, rot_dir, out)
        assert df["delta_f3"].isna().all()

    def test_no_rot_dir_rmax_all_nan(self, lt_global_csv, predictions_csv, tmp_path):
        out = tmp_path / "cat_norot.csv"
        df = build_catalog(lt_global_csv, predictions_csv, None, out)
        assert df["Rmax_kpc"].isna().all()

    def test_missing_predictions_file_yields_nan(self, lt_global_csv, rot_dir, tmp_path):
        out = tmp_path / "cat.csv"
        nonexistent = tmp_path / "no_pred.csv"
        df = build_catalog(lt_global_csv, nonexistent, rot_dir, out)
        assert df["slope_tail"].isna().all()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_returns_dataframe(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "main_out.csv"
        df = main([
            "--lt-global", str(lt_global_csv),
            "--predictions", str(predictions_csv),
            "--rot-dir", str(rot_dir),
            "--out", str(out),
        ])
        assert isinstance(df, pd.DataFrame)

    def test_main_writes_file(self, lt_global_csv, predictions_csv, rot_dir, tmp_path):
        out = tmp_path / "main_out.csv"
        main([
            "--lt-global", str(lt_global_csv),
            "--predictions", str(predictions_csv),
            "--rot-dir", str(rot_dir),
            "--out", str(out),
        ])
        assert out.exists()

    def test_main_prints_summary(self, lt_global_csv, predictions_csv, rot_dir, tmp_path, capsys):
        out = tmp_path / "main_out.csv"
        main([
            "--lt-global", str(lt_global_csv),
            "--predictions", str(predictions_csv),
            "--rot-dir", str(rot_dir),
            "--out", str(out),
        ])
        captured = capsys.readouterr()
        assert "5" in captured.out
        assert "galaxies" in captured.out
