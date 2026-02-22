"""
test_pipeline.py
----------------
Tests for the SCM analysis pipeline.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scm_models import v_piso, v_baryon, fit_piso, scm_universal_term
from src.read_iorio import read_galaxy, read_batch, validate_header
from src.scm_analysis import analyse_galaxy, run_pipeline
from src.sensitivity import sensitivity_upsilon, sensitivity_errV, run_sensitivity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_galaxy_txt(tmp_path: Path, name: str = "TestGal", n_pts: int = 15) -> Path:
    """Write a minimal synthetic rotation curve file and return its path."""
    r = np.linspace(0.5, 10.0, n_pts)
    vobs = v_piso(r, 120.0, 2.0) + np.random.default_rng(0).normal(0, 3.0, n_pts)
    errv = np.full(n_pts, 5.0)
    vgas = v_piso(r, 40.0, 2.4)
    vdisk = v_piso(r, 90.0, 1.6)
    vbul = np.zeros(n_pts)

    fp = tmp_path / f"{name}_rotmod.txt"
    with open(fp, "w") as f:
        f.write("# R Vobs errV Vgas Vdisk Vbul\n")
        for i in range(n_pts):
            f.write(f"{r[i]:.4f}  {vobs[i]:.4f}  {errv[i]:.4f}  "
                    f"{vgas[i]:.4f}  {vdisk[i]:.4f}  {vbul[i]:.4f}\n")
    return fp


# ---------------------------------------------------------------------------
# scm_models tests
# ---------------------------------------------------------------------------

class TestVPiso:
    def test_asymptotic(self):
        r = np.array([1e6])
        v = v_piso(r, 100.0, 1.0)
        assert abs(float(v[0]) - 100.0) < 0.01

    def test_shape(self):
        r = np.linspace(1, 10, 20)
        v = v_piso(r, 100.0, 2.0)
        assert v.shape == (20,)
        assert (v > 0).all()
        assert (np.diff(v) >= 0).all()  # monotonically rising or flat


class TestVBaryon:
    def test_zero_bul(self):
        vgas = np.array([10.0, 20.0])
        vdisk = np.array([30.0, 40.0])
        vbul = np.zeros(2)
        vb = v_baryon(vgas, vdisk, vbul)
        expected = np.sqrt(vgas**2 + vdisk**2)
        np.testing.assert_allclose(vb, expected, rtol=1e-10)

    def test_upsilon_scaling(self):
        vgas = np.array([10.0])
        vdisk = np.array([20.0])
        vbul = np.zeros(1)
        vb1 = v_baryon(vgas, vdisk, vbul, upsilon_disk=1.0)
        vb2 = v_baryon(vgas, vdisk, vbul, upsilon_disk=2.0)
        assert vb2 > vb1


class TestFitPiso:
    def test_recovers_params(self):
        np.random.seed(1)
        r = np.linspace(0.3, 15.0, 30)
        v_true = v_piso(r, 130.0, 2.5)
        vobs = v_true + np.random.normal(0, 2.0, 30)
        df = pd.DataFrame({"R": r, "Vobs": vobs, "errV": np.full(30, 3.0),
                           "Vgas": np.zeros(30), "Vdisk": vobs * 0.6, "Vbul": np.zeros(30)})
        res = fit_piso(df)
        assert res["success"]
        assert abs(res["v_inf"] - 130.0) < 20.0
        assert res["r_c"] > 0

    def test_positive_r_c(self):
        """r_c must always be positive (bounds enforced)."""
        np.random.seed(2)
        r = np.linspace(0.5, 20.0, 25)
        v_true = v_piso(r, 220.0, 4.0)
        vobs = v_true + np.random.normal(0, 5.0, 25)
        df = pd.DataFrame({"R": r, "Vobs": vobs, "errV": np.full(25, 8.0),
                           "Vgas": np.zeros(25), "Vdisk": vobs * 0.7, "Vbul": np.zeros(25)})
        res = fit_piso(df)
        if res["success"]:
            assert res["r_c"] > 0


class TestScmUniversalTerm:
    def test_finite_positive(self):
        np.random.seed(3)
        r = np.linspace(0.5, 10.0, 20)
        vobs = v_piso(r, 100.0, 2.0) + np.random.normal(0, 3.0, 20)
        df = pd.DataFrame({
            "R": r,
            "Vobs": vobs,
            "errV": np.full(20, 5.0),
            "Vgas": v_piso(r, 30.0, 2.5),
            "Vdisk": v_piso(r, 75.0, 1.8),
            "Vbul": np.zeros(20),
        })
        u = scm_universal_term(df)
        assert np.isfinite(u)
        assert u >= 0


# ---------------------------------------------------------------------------
# read_iorio tests
# ---------------------------------------------------------------------------

class TestReadGalaxy:
    def test_reads_sample_file(self, tmp_path):
        fp = _make_galaxy_txt(tmp_path)
        df = read_galaxy(fp)
        assert len(df) == 15
        for col in ["R", "Vobs", "errV", "Vgas", "Vdisk", "Vbul"]:
            assert col in df.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_galaxy(tmp_path / "nonexistent.txt")

    def test_too_few_columns_raises(self, tmp_path):
        fp = tmp_path / "bad.txt"
        fp.write_text("1.0 2.0\n3.0 4.0\n")
        with pytest.raises(ValueError):
            read_galaxy(fp)


class TestReadBatch:
    def test_reads_multiple(self, tmp_path):
        fps = [_make_galaxy_txt(tmp_path, f"G{i}") for i in range(3)]
        result = read_batch(fps)
        assert len(result) == 3

    def test_bad_file_skipped(self, tmp_path):
        good = _make_galaxy_txt(tmp_path, "Good")
        bad = tmp_path / "Bad_rotmod.txt"
        bad.write_text("only two columns\n1 2\n")
        result = read_batch([good, bad])
        assert len(result) == 1
        assert "Good_rotmod" in result


class TestValidateHeader:
    def test_valid_df_passes(self):
        df = pd.DataFrame({c: [1.0] for c in ["R", "Vobs", "errV", "Vgas", "Vdisk", "Vbul"]})
        validate_header(df)  # should not raise

    def test_missing_column_raises(self):
        df = pd.DataFrame({"R": [1.0], "Vobs": [2.0], "errV": [0.5],
                           "Vgas": [0.1], "Vdisk": [1.0]})
        with pytest.raises(ValueError):
            validate_header(df)


# ---------------------------------------------------------------------------
# scm_analysis tests
# ---------------------------------------------------------------------------

class TestAnalyseGalaxy:
    def test_returns_expected_keys(self, tmp_path):
        fp = _make_galaxy_txt(tmp_path)
        df = read_galaxy(fp)
        row = analyse_galaxy("TestGal", df)
        for key in ["galaxy", "n_points", "scm_u_term", "veredicto", "fit_ok"]:
            assert key in row

    def test_scm_u_term_finite(self, tmp_path):
        fp = _make_galaxy_txt(tmp_path)
        df = read_galaxy(fp)
        row = analyse_galaxy("TestGal", df)
        assert np.isfinite(row["scm_u_term"])


class TestRunPipeline:
    def test_generates_csv(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir = tmp_path / "results"
        for i in range(3):
            _make_galaxy_txt(data_dir, f"G{i}")

        filepaths = sorted(data_dir.glob("*.txt"))
        results = run_pipeline(filepaths, out_dir=out_dir)

        assert len(results) == 3
        csv = out_dir / "universal_term_comparison_full.csv"
        assert csv.exists()
        df_out = pd.read_csv(csv)
        assert len(df_out) == 3
        assert not df_out.isna().any().any()


# ---------------------------------------------------------------------------
# sensitivity tests
# ---------------------------------------------------------------------------

class TestSensitivityUpsilon:
    def test_returns_expected_columns(self, tmp_path):
        fp = _make_galaxy_txt(tmp_path)
        df = read_galaxy(fp)
        result = sensitivity_upsilon(df, n_steps=5)
        assert "upsilon_disk" in result.columns
        assert "scm_u_term" in result.columns
        assert len(result) == 5


class TestRunSensitivity:
    def test_generates_per_galaxy_csv(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        out_dir = tmp_path / "sensitivity"
        for i in range(2):
            _make_galaxy_txt(data_dir, f"S{i}")

        filepaths = sorted(data_dir.glob("*.txt"))
        results = run_sensitivity(filepaths, out_dir=out_dir)

        assert len(results) == 2
        for name in results:
            csv = out_dir / f"{name}_sensitivity.csv"
            assert csv.exists()
            df = pd.read_csv(csv)
            assert len(df) > 0
            assert not df["scm_u_term"].isnull().any()
