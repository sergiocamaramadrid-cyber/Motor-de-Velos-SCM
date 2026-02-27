"""
Unit tests for src/scm_analysis.py.

These tests exercise the full pipeline using synthetic (mock) galaxy data so
that no real SPARC download is required.
"""

import os
import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scm_analysis import (
    load_galaxy_table,
    load_rotation_curve,
    fit_galaxy,
    run_pipeline,
    load_custom_rotation_curve,
    run_custom_galaxy,
    _compute_kinematic_metrics,
    load_pressure_calibration,
    estimate_xi_from_sfr,
    _write_executive_summary,
    _write_audit_summary,
    _write_top10_latex,
)


# ---------------------------------------------------------------------------
# Fixtures — build a minimal synthetic SPARC dataset
# ---------------------------------------------------------------------------

@pytest.fixture()
def sparc_dir(tmp_path):
    """Create a tiny synthetic SPARC dataset in a temporary directory."""
    # Galaxy table
    galaxy_csv = tmp_path / "SPARC_Lelli2016c.csv"
    galaxy_table = pd.DataFrame({
        "Galaxy": ["NGC0000", "NGC0001", "NGC0002"],
        "D": [10.0, 20.0, 30.0],
        "Inc": [45.0, 60.0, 75.0],
        "L36": [1e9, 2e9, 3e9],
        "Vflat": [150.0, 200.0, 250.0],
        "e_Vflat": [5.0, 5.0, 5.0],
    })
    galaxy_table.to_csv(galaxy_csv, index=False)

    # Rotation curves
    rng = np.random.default_rng(42)
    for _, row in galaxy_table.iterrows():
        name = row["Galaxy"]
        v_flat = row["Vflat"]
        r = np.linspace(0.5, 15, 20)
        v_obs = np.full(20, v_flat) + rng.normal(0, 3, 20)
        v_obs_err = np.full(20, 5.0)
        v_gas = 0.3 * v_flat * np.ones(20)
        v_disk = 0.8 * v_flat * np.ones(20)
        v_bul = np.zeros(20)

        rc_df = pd.DataFrame({
            "r": r,
            "v_obs": v_obs,
            "v_obs_err": v_obs_err,
            "v_gas": v_gas,
            "v_disk": v_disk,
            "v_bul": v_bul,
            "SBdisk": np.zeros(20),
            "SBbul": np.zeros(20),
        })
        rc_path = tmp_path / f"{name}_rotmod.dat"
        rc_df.to_csv(rc_path, sep=" ", index=False, header=False)

    return tmp_path


# ---------------------------------------------------------------------------
# load_galaxy_table
# ---------------------------------------------------------------------------

class TestLoadGalaxyTable:
    def test_loads_csv(self, sparc_dir):
        df = load_galaxy_table(sparc_dir)
        assert "Galaxy" in df.columns
        assert len(df) == 3

    def test_missing_dir_raises(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            load_galaxy_table(missing)

    def test_no_table_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_galaxy_table(tmp_path)


# ---------------------------------------------------------------------------
# load_rotation_curve
# ---------------------------------------------------------------------------

class TestLoadRotationCurve:
    def test_returns_expected_columns(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0000")
        for col in ["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]:
            assert col in rc.columns

    def test_missing_galaxy_raises(self, sparc_dir):
        with pytest.raises(FileNotFoundError):
            load_rotation_curve(sparc_dir, "NOEXIST")

    def test_row_count(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0001")
        assert len(rc) == 20


# ---------------------------------------------------------------------------
# fit_galaxy
# ---------------------------------------------------------------------------

class TestFitGalaxy:
    def test_fit_returns_expected_keys(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0000")
        fit = fit_galaxy(rc)
        assert set(fit.keys()) == {"upsilon_disk", "chi2_reduced", "n_points"}

    def test_upsilon_disk_in_range(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0000")
        fit = fit_galaxy(rc)
        assert 0.1 <= fit["upsilon_disk"] <= 5.0

    def test_chi2_non_negative(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0000")
        fit = fit_galaxy(rc)
        assert fit["chi2_reduced"] >= 0.0

    def test_n_points_correct(self, sparc_dir):
        rc = load_rotation_curve(sparc_dir, "NGC0001")
        fit = fit_galaxy(rc)
        assert fit["n_points"] == 20


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_returns_dataframe(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        df = run_pipeline(sparc_dir, out_dir, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_processes_all_galaxies(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        df = run_pipeline(sparc_dir, out_dir, verbose=False)
        assert len(df) == 3

    def test_creates_csv(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        assert (out_dir / "universal_term_comparison_full.csv").exists()

    def test_creates_executive_summary(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        summary_path = out_dir / "executive_summary.txt"
        assert summary_path.exists()
        text = summary_path.read_text(encoding="utf-8")
        assert "SCM Pipeline" in text

    def test_creates_per_galaxy_csv(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        assert (out_dir / "per_galaxy_summary.csv").exists()

    def test_creates_latex_table(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        tex_path = out_dir / "top10_universal.tex"
        assert tex_path.exists()
        tex = tex_path.read_text(encoding="utf-8")
        assert r"\begin{table}" in tex

    def test_output_columns(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        df = run_pipeline(sparc_dir, out_dir, verbose=False)
        for col in ["galaxy", "chi2_reduced", "upsilon_disk", "n_points"]:
            assert col in df.columns

    def test_custom_a0(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results_a0"
        df = run_pipeline(sparc_dir, out_dir, a0=0.5e-10, verbose=False)
        assert len(df) == 3

    def test_creates_audit_summary(self, sparc_dir, tmp_path):
        out_dir = tmp_path / "results"
        run_pipeline(sparc_dir, out_dir, verbose=False)
        audit_path = out_dir / "audit_summary.json"
        assert audit_path.exists()
        import json
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        assert "xi_calibration" in data
        assert data["xi_calibration"]["version"] == "v0.6.1"
        assert "pipeline_stats" in data


# ---------------------------------------------------------------------------
# _write_executive_summary
# ---------------------------------------------------------------------------

class TestWriteExecutiveSummary:
    def test_empty_dataframe(self, tmp_path):
        df = pd.DataFrame()
        path = tmp_path / "summary.txt"
        _write_executive_summary(df, path)
        text = path.read_text(encoding="utf-8")
        assert "N_galaxies: 0" in text

    def test_non_empty_dataframe(self, tmp_path):
        df = pd.DataFrame({
            "chi2_reduced": [0.5, 1.0, 2.5, 3.0],
            "upsilon_disk": [1.0, 1.5, 2.0, 2.5],
        })
        path = tmp_path / "summary.txt"
        _write_executive_summary(df, path)
        text = path.read_text(encoding="utf-8")
        assert "N_galaxies: 4" in text
        assert "chi2_reduced median" in text
        assert "upsilon_disk median" in text


# ---------------------------------------------------------------------------
# _write_top10_latex
# ---------------------------------------------------------------------------

class TestWriteTop10Latex:
    def test_empty_dataframe(self, tmp_path):
        df = pd.DataFrame()
        path = tmp_path / "top10.tex"
        _write_top10_latex(df, path)
        text = path.read_text(encoding="utf-8")
        assert "% No data" in text

    def test_contains_galaxy_names(self, tmp_path):
        df = pd.DataFrame({
            "galaxy": [f"NGC{i:04d}" for i in range(15)],
            "chi2_reduced": np.linspace(0.5, 3.0, 15),
            "upsilon_disk": np.ones(15),
            "Vflat_kms": np.full(15, 150.0),
        })
        path = tmp_path / "top10.tex"
        _write_top10_latex(df, path)
        text = path.read_text(encoding="utf-8")
        assert "NGC0000" in text
        # Only top 10 should be present
        assert "NGC0014" not in text

    def test_tabular_structure(self, tmp_path):
        df = pd.DataFrame({
            "galaxy": ["G1", "G2"],
            "chi2_reduced": [0.8, 1.2],
            "upsilon_disk": [1.0, 1.5],
            "Vflat_kms": [150.0, float("nan")],
        })
        path = tmp_path / "top10.tex"
        _write_top10_latex(df, path)
        text = path.read_text(encoding="utf-8")
        assert r"\begin{tabular}" in text
        assert "---" in text  # nan Vflat renders as ---


# ---------------------------------------------------------------------------
# load_pressure_calibration
# ---------------------------------------------------------------------------

_CALIBRATION_PATH = Path(__file__).parent.parent / "data/calibration/local_group_xi_calibration.json"


class TestLoadPressureCalibration:
    def test_loads_from_default_path(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        assert data["calibration_id"] == "SCM_XI_LOCAL_GROUP_v0.6.1"

    def test_returns_dict(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        assert isinstance(data, dict)

    def test_required_top_level_keys(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        for key in ("calibration_id", "framework_version", "sample_size",
                    "xi_statistics", "sfr_model", "galaxies"):
            assert key in data, f"Missing key: {key}"

    def test_galaxies_list_length(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        assert len(data["galaxies"]) == 6

    def test_xi_statistics_values(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        stats = data["xi_statistics"]
        assert stats["mean"] == pytest.approx(1.36)
        assert stats["min"] == pytest.approx(1.28)
        assert stats["max"] == pytest.approx(1.42)

    def test_sfr_model_values(self):
        data = load_pressure_calibration(_CALIBRATION_PATH)
        sfr = data["sfr_model"]
        assert sfr["intercept"] == pytest.approx(1.33)
        assert sfr["slope"] == pytest.approx(0.21)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pressure_calibration(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# estimate_xi_from_sfr
# ---------------------------------------------------------------------------

class TestEstimateXiFromSfr:
    def test_baseline_sfr_gives_intercept(self):
        # log10(SFR) = 0 → xi = 1.33
        assert estimate_xi_from_sfr(0.0) == pytest.approx(1.33)

    def test_positive_log_sfr_increases_xi(self):
        assert estimate_xi_from_sfr(1.0) > estimate_xi_from_sfr(0.0)

    def test_negative_log_sfr_decreases_xi(self):
        assert estimate_xi_from_sfr(-1.0) < estimate_xi_from_sfr(0.0)

    def test_clamped_at_maximum(self):
        # Very high SFR → xi clamped at 1.42
        assert estimate_xi_from_sfr(100.0) == pytest.approx(1.42)

    def test_clamped_at_minimum(self):
        # Very low SFR → xi clamped at 1.28
        assert estimate_xi_from_sfr(-100.0) == pytest.approx(1.28)

    def test_lmc_log_sfr(self):
        # LMC: log_sfr = 0.30 → xi ≈ 1.33 + 0.21*0.30 = 1.393 → clamped within [1.28, 1.42]
        xi = estimate_xi_from_sfr(0.30)
        assert 1.28 <= xi <= 1.42

    def test_returns_float(self):
        assert isinstance(estimate_xi_from_sfr(0.0), float)


# ---------------------------------------------------------------------------
# _write_audit_summary
# ---------------------------------------------------------------------------

import json as _json


class TestWriteAuditSummary:
    def _make_df(self):
        return pd.DataFrame({
            "galaxy": ["G1", "G2"],
            "chi2_reduced": [0.8, 1.2],
            "upsilon_disk": [1.0, 1.5],
        })

    def test_creates_json_file(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        _write_audit_summary(self._make_df(), {}, path)
        assert path.exists()

    def test_xi_calibration_key_preserved(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        audit = {
            "xi_calibration": {
                "version": "v0.6.1",
                "model": "xi = 1.33 + 0.21 log10(SFR)",
                "range": [1.28, 1.42],
            }
        }
        _write_audit_summary(self._make_df(), audit, path)
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert data["xi_calibration"]["version"] == "v0.6.1"
        assert data["xi_calibration"]["range"] == [1.28, 1.42]

    def test_pipeline_stats_present(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        _write_audit_summary(self._make_df(), {}, path)
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert "pipeline_stats" in data
        assert data["pipeline_stats"]["n_galaxies"] == 2

    def test_chi2_median_correct(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        _write_audit_summary(self._make_df(), {}, path)
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert data["pipeline_stats"]["chi2_reduced_median"] == pytest.approx(1.0)

    def test_empty_dataframe(self, tmp_path):
        path = tmp_path / "audit_summary.json"
        _write_audit_summary(pd.DataFrame(), {}, path)
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert data["pipeline_stats"]["n_galaxies"] == 0
        assert data["pipeline_stats"]["chi2_reduced_median"] is None


# ---------------------------------------------------------------------------
# load_custom_rotation_curve
# ---------------------------------------------------------------------------

class TestLoadCustomRotationCurve:
    def _write_rotcurve(self, tmp_path, content):
        p = tmp_path / "test_rotcurve.txt"
        p.write_text(content, encoding="utf-8")
        return p

    def test_loads_three_column_file(self, tmp_path):
        p = self._write_rotcurve(tmp_path,
            "# radius_kpc velocity_kms error_kms\n"
            "0.5  55.1  2.2\n"
            "1.0  78.4  2.1\n"
        )
        df = load_custom_rotation_curve(p)
        assert len(df) == 2
        assert list(df.columns) == ["r", "v_obs", "v_obs_err", "v_gas", "v_disk", "v_bul"]

    def test_radii_values(self, tmp_path):
        p = self._write_rotcurve(tmp_path,
            "0.5  55.1  2.2\n1.0  78.4  2.1\n"
        )
        df = load_custom_rotation_curve(p)
        assert df["r"].tolist() == pytest.approx([0.5, 1.0])

    def test_v_obs_values(self, tmp_path):
        p = self._write_rotcurve(tmp_path, "2.0  102.3  2.0\n")
        df = load_custom_rotation_curve(p)
        assert df["v_obs"].iloc[0] == pytest.approx(102.3)

    def test_v_gas_zero(self, tmp_path):
        p = self._write_rotcurve(tmp_path, "1.0  78.4  2.1\n")
        df = load_custom_rotation_curve(p)
        assert df["v_gas"].iloc[0] == pytest.approx(0.0)

    def test_v_disk_equals_v_obs(self, tmp_path):
        p = self._write_rotcurve(tmp_path, "1.0  78.4  2.1\n")
        df = load_custom_rotation_curve(p)
        assert df["v_disk"].iloc[0] == pytest.approx(df["v_obs"].iloc[0])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_custom_rotation_curve(tmp_path / "nonexistent.txt")

    def test_tab_separated(self, tmp_path):
        p = self._write_rotcurve(tmp_path, "0.5\t55.1\t2.2\n1.0\t78.4\t2.1\n")
        df = load_custom_rotation_curve(p)
        assert len(df) == 2

    def test_m81_data_file(self):
        """Real m81_rotcurve.txt file in repo."""
        from pathlib import Path as _Path
        repo_root = _Path(__file__).parent.parent
        data_file = repo_root / "data" / "m81_group" / "m81_rotcurve.txt"
        if data_file.exists():
            df = load_custom_rotation_curve(data_file)
            assert len(df) >= 5
            assert (df["r"] > 0).all()
            assert (df["v_obs"] > 0).all()


# ---------------------------------------------------------------------------
# run_custom_galaxy
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_rotcurve(tmp_path):
    """Write a minimal 3-column rotation curve."""
    p = tmp_path / "g_rotcurve.txt"
    rows = "\n".join(f"{r:.1f}  {100.0 + r:.1f}  2.5" for r in [0.5, 1.0, 2.0, 3.0, 4.0])
    p.write_text("# radius_kpc velocity_kms error_kms\n" + rows + "\n", encoding="utf-8")
    return p


class TestRunCustomGalaxy:
    def test_returns_dict_with_expected_keys(self, simple_rotcurve, tmp_path):
        result = run_custom_galaxy("TestGal", simple_rotcurve, tmp_path / "out",
                                   verbose=False)
        for key in ("galaxy", "upsilon_disk", "chi2_reduced", "n_points",
                    "xi", "VIF_hinge", "deltaV_reduction_percent",
                    "condition_number_kappa", "pressure_injectors_detected",
                    "pressure_injector_detected", "audit_mode"):
            assert key in result

    def test_galaxy_name_in_result(self, simple_rotcurve, tmp_path):
        result = run_custom_galaxy("MyGalaxy", simple_rotcurve, tmp_path / "out",
                                   verbose=False)
        assert result["galaxy"] == "MyGalaxy"

    def test_xi_in_valid_range(self, simple_rotcurve, tmp_path):
        result = run_custom_galaxy("G", simple_rotcurve, tmp_path / "out", verbose=False)
        assert 1.28 <= result["xi"] <= 1.50

    def test_audit_mode_stored(self, simple_rotcurve, tmp_path):
        result = run_custom_galaxy("G", simple_rotcurve, tmp_path / "out",
                                   audit_mode="high-pressure", verbose=False)
        assert result["audit_mode"] == "high-pressure"

    def test_creates_result_csv_in_galaxy_subdir(self, simple_rotcurve, tmp_path):
        out = tmp_path / "out"
        run_custom_galaxy("G", simple_rotcurve, out, verbose=False)
        assert (out / "G" / "G_result.csv").exists()

    def test_creates_audit_summary_json_in_galaxy_subdir(self, simple_rotcurve, tmp_path):
        import json as _json
        out = tmp_path / "out"
        run_custom_galaxy("G", simple_rotcurve, out, verbose=False)
        p = out / "G" / "audit_summary.json"
        assert p.exists()
        data = _json.loads(p.read_text(encoding="utf-8"))
        assert "xi_calibration" in data
        assert "custom_run" in data

    def test_detect_pressure_injectors_flag(self, simple_rotcurve, tmp_path):
        result = run_custom_galaxy("G", simple_rotcurve, tmp_path / "out",
                                   detect_pressure_injectors=True, verbose=False)
        assert isinstance(result["pressure_injector_detected"], bool)

    def test_outdir_and_galaxy_subdir_created(self, simple_rotcurve, tmp_path):
        out = tmp_path / "new_dir" / "subdir"
        run_custom_galaxy("MyGal", simple_rotcurve, out, verbose=False)
        assert out.exists()
        assert (out / "MyGal").exists()



# ---------------------------------------------------------------------------
# CLI — _parse_args new flags
# ---------------------------------------------------------------------------

class TestCLINewFlags:
    def _parse(self, *args):
        from src.scm_analysis import _parse_args
        return _parse_args(list(args))

    def test_custom_data_flag(self, tmp_path):
        p = tmp_path / "g.txt"
        p.write_text("1.0 100.0 2.0\n")
        ns = self._parse("--custom-data", str(p), "--data-dir", "x")
        assert ns.custom_data == str(p)

    def test_target_galaxy_flag(self):
        ns = self._parse("--data-dir", "d", "--target-galaxy", "M82")
        assert ns.target_galaxy == "M82"

    def test_detect_pressure_injectors_flag(self):
        ns = self._parse("--data-dir", "d", "--detect-pressure-injectors")
        assert ns.detect_pressure_injectors is True

    def test_detect_pressure_injectors_default_false(self):
        ns = self._parse("--data-dir", "d")
        assert ns.detect_pressure_injectors is False

    def test_audit_mode_flag(self):
        ns = self._parse("--data-dir", "d", "--audit-mode", "high-pressure")
        assert ns.audit_mode == "high-pressure"

    def test_outdir_alias(self):
        ns = self._parse("--data-dir", "d", "--outdir", "my/out")
        assert ns.out == "my/out"


# ---------------------------------------------------------------------------
# _compute_kinematic_metrics
# ---------------------------------------------------------------------------

@pytest.fixture()
def m82_rotcurve():
    """Load the real M82 rotation curve from the repository data files."""
    from pathlib import Path as _Path
    return _Path(__file__).parent.parent / "data" / "m81_group" / "m82_rotcurve.txt"


class TestComputeKinematicMetrics:
    def _make_rc(self, r_values, v_values, err=2.0):
        """Build a minimal rotation-curve DataFrame."""
        import pandas as _pd
        return _pd.DataFrame({
            "r": r_values,
            "v_obs": v_values,
            "v_obs_err": [err] * len(r_values),
            "v_gas": [0.0] * len(r_values),
            "v_disk": v_values,
            "v_bul": [0.0] * len(r_values),
        })

    def test_returns_all_keys(self, m82_rotcurve):
        rc = load_custom_rotation_curve(m82_rotcurve)
        km = _compute_kinematic_metrics(rc)
        for key in ("xi", "VIF_hinge", "deltaV_reduction_percent",
                    "condition_number_kappa", "pressure_injectors_detected"):
            assert key in km

    def test_m82_xi_in_starburst_range(self, m82_rotcurve):
        """M82 xi should fall in the expected starburst range 1.44–1.50."""
        rc = load_custom_rotation_curve(m82_rotcurve)
        km = _compute_kinematic_metrics(rc)
        assert 1.44 <= km["xi"] <= 1.50, f"xi={km['xi']:.4f} outside starburst range"

    def test_xi_clamped_low(self):
        """Curves with very low inner velocity relative to flat clamp to xi_min=1.28.

        S = (V_inner/V_flat)² × (r_flat/r_inner).  With V_inner << V_flat the
        steepness index S ≪ 1 and the formula gives xi < 1.28, which is clamped.
        Example: V rises from 5 km/s at r=1 kpc to a flat ~200 km/s at the outer points.
        """
        rc = self._make_rc([1.0, 2.0, 4.0, 8.0], [5.0, 100.0, 200.0, 200.0])
        km = _compute_kinematic_metrics(rc)
        assert km["xi"] == pytest.approx(1.28)

    def test_xi_clamped_high(self):
        """Curves with very high inner velocity relative to flat clamp to xi_max=1.50.

        When V_inner ≈ V_flat AND r_flat/r_inner is large, S >> 6.57 and the
        formula gives xi > 1.50, which is clamped.
        Example: V ≈ 99 km/s at r=0.1 kpc, flat at 100 km/s to r=100 kpc.
        """
        rc = self._make_rc([0.1, 1.0, 10.0, 100.0], [99.0, 100.0, 100.0, 100.0])
        km = _compute_kinematic_metrics(rc)
        assert km["xi"] == pytest.approx(1.50)

    def test_vif_hinge_formula(self):
        """VIF_hinge = mean(last 3 v_obs) / first v_obs."""
        rc = self._make_rc([0.5, 2.0, 5.0, 10.0], [50.0, 100.0, 150.0, 200.0])
        km = _compute_kinematic_metrics(rc)
        # V_flat = mean of last 3 points (indices 1, 2, 3) = mean(100, 150, 200) = 150
        expected_vif = float((100.0 + 150.0 + 200.0) / 3.0 / 50.0)
        assert km["VIF_hinge"] == pytest.approx(expected_vif, rel=0.05)

    def test_deltav_reduction_positive(self, m82_rotcurve):
        from src.scm_analysis import _DV_LOW
        rc = load_custom_rotation_curve(m82_rotcurve)
        km = _compute_kinematic_metrics(rc)
        assert km["deltaV_reduction_percent"] >= _DV_LOW

    def test_deltav_reduction_increases_with_xi(self):
        """Higher xi → higher DeltaV_reduction.

        rc_low: V_inner=5 at r=1 (S→low, xi clamped at 1.28).
        rc_high: V_inner=100, V_flat≈200 at r_flat=10 with r_inner=0.5 (S≈5, xi≈1.45).
        """
        rc_low = self._make_rc([1.0, 2.0, 4.0, 8.0], [5.0, 100.0, 200.0, 200.0])
        rc_high = self._make_rc([0.5, 2.0, 5.0, 10.0], [100.0, 160.0, 200.0, 200.0])
        km_low = _compute_kinematic_metrics(rc_low)
        km_high = _compute_kinematic_metrics(rc_high)
        assert km_high["deltaV_reduction_percent"] >= km_low["deltaV_reduction_percent"]

    def test_condition_number_positive(self, m82_rotcurve):
        rc = load_custom_rotation_curve(m82_rotcurve)
        km = _compute_kinematic_metrics(rc)
        assert km["condition_number_kappa"] > 0

    def test_pressure_injectors_minimum_one(self, m82_rotcurve):
        rc = load_custom_rotation_curve(m82_rotcurve)
        km = _compute_kinematic_metrics(rc)
        assert km["pressure_injectors_detected"] >= 1

    def test_pressure_injectors_m82_high(self, m82_rotcurve):
        """M82 (starburst) should have more regions than a normal spiral."""
        rc_m82 = load_custom_rotation_curve(m82_rotcurve)
        km_m82 = _compute_kinematic_metrics(rc_m82)
        # M82 xi≈1.47 → regions=4; a flat galaxy (xi=1.28) → regions=1
        assert km_m82["pressure_injectors_detected"] >= 3
