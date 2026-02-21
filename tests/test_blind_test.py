"""Unit tests for scripts/run_blind_test_final.py."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_blind_test_final import (
    SCM_FIXED_PARAMS,
    load_lt_file,
    log_veredicto,
    predict_scm_blind,
    run_batch,
)

LT_DIR = REPO_ROOT / "data" / "little_things"
GALAXIES = ["DDO154", "DDO53", "NGC2366", "IC2574", "WLM"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_verdict(tmp_path):
    """Return a temporary path for veredicto_final.txt."""
    return str(tmp_path / "veredicto_final.txt")


# ---------------------------------------------------------------------------
# Tests: load_lt_file
# ---------------------------------------------------------------------------

class TestLoadLtFile:
    @pytest.mark.parametrize("galaxy", GALAXIES)
    def test_loads_each_galaxy(self, galaxy):
        """Each LITTLE THINGS CSV must load without errors."""
        df = load_lt_file(galaxy)
        assert len(df) >= 5

    @pytest.mark.parametrize("galaxy", GALAXIES)
    def test_required_columns_present(self, galaxy):
        """Loader must produce r_kpc, vobs, evobs, vbary."""
        df = load_lt_file(galaxy)
        for col in ["r_kpc", "vobs", "evobs", "vbary"]:
            assert col in df.columns, f"Missing column '{col}' for {galaxy}"

    @pytest.mark.parametrize("galaxy", GALAXIES)
    def test_all_finite(self, galaxy):
        df = load_lt_file(galaxy)
        for col in ["r_kpc", "vobs", "evobs", "vbary"]:
            assert np.all(np.isfinite(df[col].values)), f"Non-finite in '{col}' for {galaxy}"

    @pytest.mark.parametrize("galaxy", GALAXIES)
    def test_positive_radii(self, galaxy):
        df = load_lt_file(galaxy)
        assert (df["r_kpc"] > 0).all()

    @pytest.mark.parametrize("galaxy", GALAXIES)
    def test_positive_errors(self, galaxy):
        df = load_lt_file(galaxy)
        assert (df["evobs"] > 0).all()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_lt_file("NONEXISTENT_GALAXY_XYZ")

    def test_vbary_nonnegative(self):
        """vbary must be ≥ 0 (quadrature sum of velocities)."""
        df = load_lt_file("DDO154")
        assert (df["vbary"] >= 0).all()

    def test_csv_file_preferred_over_dat(self, tmp_path, monkeypatch):
        """Loader finds .csv when .dat is absent."""
        # Write a minimal CSV in a temp dir
        mini = pd.DataFrame({
            "rad": [1.0, 2.0],
            "Vrot": [30.0, 40.0],
            "e_Vrot": [5.0, 5.0],
            "vgas": [25.0, 30.0],
            "vdisk": [10.0, 12.0],
        })
        (tmp_path / "TESTGXY.csv").write_text("# comment\n" + mini.to_csv(index=False))
        monkeypatch.setattr("scripts.run_blind_test_final.DATA_DIR", str(tmp_path) + "/")
        df = load_lt_file("TESTGXY")
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Tests: predict_scm_blind
# ---------------------------------------------------------------------------

class TestPredictScmBlind:
    @pytest.mark.parametrize("galaxy", GALAXIES)
    def test_output_positive(self, galaxy):
        """Predicted velocity must be strictly positive."""
        df = load_lt_file(galaxy)
        v = predict_scm_blind(df, SCM_FIXED_PARAMS)
        assert np.all(v > 0), f"Non-positive prediction for {galaxy}"

    @pytest.mark.parametrize("galaxy", GALAXIES)
    def test_output_finite(self, galaxy):
        df = load_lt_file(galaxy)
        v = predict_scm_blind(df, SCM_FIXED_PARAMS)
        assert np.all(np.isfinite(v)), f"Non-finite prediction for {galaxy}"

    @pytest.mark.parametrize("galaxy", GALAXIES)
    def test_output_shape(self, galaxy):
        df = load_lt_file(galaxy)
        v = predict_scm_blind(df, SCM_FIXED_PARAMS)
        assert v.shape == (len(df),)

    def test_scm_exceeds_baryonic(self):
        """SCM velocity must be ≥ baryonic velocity (adds energy)."""
        df = load_lt_file("DDO154")
        v_scm = predict_scm_blind(df, SCM_FIXED_PARAMS)
        v_bar = df["vbary"].values
        assert np.all(v_scm >= v_bar - 1e-9)

    def test_hinge_activates_at_low_gbar(self):
        """Higher hinge_d → larger SCM prediction at low accelerations."""
        df = load_lt_file("DDO154")
        coeffs_strong = dict(SCM_FIXED_PARAMS, d=5.0)
        coeffs_weak   = dict(SCM_FIXED_PARAMS, d=0.1)
        v_strong = predict_scm_blind(df, coeffs_strong)
        v_weak   = predict_scm_blind(df, coeffs_weak)
        assert np.all(v_strong >= v_weak - 1e-9)


# ---------------------------------------------------------------------------
# Tests: log_veredicto
# ---------------------------------------------------------------------------

class TestLogVeredicto:
    def _make_res(self, galaxy="DDO154", delta=-10.0):
        return {"galaxy": galaxy, "rmse_bar": 15.0, "rmse_scm": 10.0,
                "delta_chi2": delta}

    def test_creates_file(self, tmp_verdict):
        assert not os.path.isfile(tmp_verdict)
        log_veredicto(self._make_res(), SCM_FIXED_PARAMS, tmp_verdict)
        assert os.path.isfile(tmp_verdict)

    def test_header_written_once(self, tmp_verdict):
        for _ in range(3):
            log_veredicto(self._make_res(), SCM_FIXED_PARAMS, tmp_verdict)
        with open(tmp_verdict) as f:
            content = f.read()
        assert content.count("VEREDICTO TEST CIEGO") == 1

    def test_mejora_si_when_delta_lt_minus_5(self, tmp_verdict):
        log_veredicto(self._make_res(delta=-10.0), SCM_FIXED_PARAMS, tmp_verdict)
        with open(tmp_verdict) as f:
            content = f.read()
        assert "SÍ" in content

    def test_mejora_no_when_delta_ge_minus_5(self, tmp_verdict):
        log_veredicto(self._make_res(delta=2.0), SCM_FIXED_PARAMS, tmp_verdict)
        with open(tmp_verdict) as f:
            content = f.read()
        assert "NO" in content

    def test_appends_multiple_galaxies(self, tmp_verdict):
        for g in ["DDO154", "DDO53", "NGC2366"]:
            log_veredicto(self._make_res(galaxy=g), SCM_FIXED_PARAMS, tmp_verdict)
        with open(tmp_verdict) as f:
            content = f.read()
        for g in ["DDO154", "DDO53", "NGC2366"]:
            assert g in content

    def test_coeffs_in_header(self, tmp_verdict):
        log_veredicto(self._make_res(), SCM_FIXED_PARAMS, tmp_verdict)
        with open(tmp_verdict) as f:
            content = f.read()
        assert "logg0" in content


# ---------------------------------------------------------------------------
# Tests: run_batch
# ---------------------------------------------------------------------------

class TestRunBatch:
    def test_returns_list_of_dicts(self, tmp_verdict):
        results = run_batch(["DDO154", "DDO53"], verdict_file=tmp_verdict)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_result_keys(self, tmp_verdict):
        results = run_batch(["DDO154"], verdict_file=tmp_verdict)
        assert results, "Expected at least one result"
        expected = {"galaxy", "rmse_bar", "rmse_scm", "delta_chi2"}
        assert expected.issubset(results[0].keys())

    def test_result_galaxy_name(self, tmp_verdict):
        results = run_batch(["NGC2366"], verdict_file=tmp_verdict)
        assert results[0]["galaxy"] == "NGC2366"

    def test_rmse_nonnegative(self, tmp_verdict):
        results = run_batch(["IC2574", "WLM"], verdict_file=tmp_verdict)
        for r in results:
            assert r["rmse_bar"] >= 0
            assert r["rmse_scm"] >= 0

    def test_skips_missing_galaxy(self, tmp_verdict, capsys):
        """A missing galaxy raises no exception; just logs an error."""
        results = run_batch(["NONEXISTENT_GALAXY"], verdict_file=tmp_verdict)
        assert results == []  # nothing returned for failures
        captured = capsys.readouterr()
        assert "Error" in captured.out or "Error" in captured.err

    def test_all_five_galaxies(self, tmp_verdict):
        results = run_batch(GALAXIES, verdict_file=tmp_verdict)
        assert len(results) == len(GALAXIES)
        returned_names = [r["galaxy"] for r in results]
        assert returned_names == GALAXIES

    def test_verdict_file_written(self, tmp_verdict):
        run_batch(["DDO154"], verdict_file=tmp_verdict)
        assert os.path.isfile(tmp_verdict)
        with open(tmp_verdict) as f:
            content = f.read()
        assert "DDO154" in content
