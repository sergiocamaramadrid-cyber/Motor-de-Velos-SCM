"""tests/test_entry_points.py — Tests for the 4 named entry-point scripts and
run_full_pipeline.py."""

from __future__ import annotations

import sys
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from scripts.compute_slope_tail import main as compute_slope_tail_main
from scripts.build_master_catalog import main as build_master_catalog_main
from scripts.mass_split_analysis import main as mass_split_analysis_main
from scripts.env_mass_scan import main as env_mass_scan_main
from scripts.run_full_pipeline import (
    PIPELINE_STEPS,
    STEP_KEYS,
    main as pipeline_main,
    run_pipeline,
    run_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_completed(returncode: int = 0) -> MagicMock:
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    return cp


# ---------------------------------------------------------------------------
# Entry-point delegation tests
# ---------------------------------------------------------------------------

class TestComputeSlopeTail:
    def test_delegates_to_sparc_slope_tail(self):
        sentinel = {"slopes": [], "n": 0, "out_path": "x"}
        with patch("scripts.compute_slope_tail._main", return_value=sentinel) as mock:
            result = compute_slope_tail_main([])
        mock.assert_called_once_with([])
        assert result is sentinel

    def test_passes_argv_none(self):
        sentinel = {"slopes": [], "n": 0, "out_path": "x"}
        with patch("scripts.compute_slope_tail._main", return_value=sentinel) as mock:
            compute_slope_tail_main(None)
        mock.assert_called_once_with(None)

    def test_returns_dict(self):
        with patch("scripts.compute_slope_tail._main", return_value={}):
            result = compute_slope_tail_main([])
        assert isinstance(result, dict)

    def test_passes_arbitrary_argv(self):
        sentinel = {}
        with patch("scripts.compute_slope_tail._main", return_value=sentinel) as mock:
            compute_slope_tail_main(["--out", "results/custom.csv"])
        mock.assert_called_once_with(["--out", "results/custom.csv"])


class TestBuildMasterCatalog:
    def test_delegates_to_build_galaxy_catalog_env(self):
        sentinel = {"catalog": None, "n": 0, "out_path": "x"}
        with patch("scripts.build_master_catalog._main", return_value=sentinel) as mock:
            result = build_master_catalog_main([])
        mock.assert_called_once_with([])
        assert result is sentinel

    def test_passes_argv_none(self):
        with patch("scripts.build_master_catalog._main", return_value={}) as mock:
            build_master_catalog_main(None)
        mock.assert_called_once_with(None)

    def test_returns_dict(self):
        with patch("scripts.build_master_catalog._main", return_value={}):
            result = build_master_catalog_main([])
        assert isinstance(result, dict)

    def test_passes_arbitrary_argv(self):
        with patch("scripts.build_master_catalog._main", return_value={}) as mock:
            build_master_catalog_main(["--sparc", "data/sparc_basic.csv"])
        mock.assert_called_once_with(["--sparc", "data/sparc_basic.csv"])


class TestMassSplitAnalysis:
    def test_delegates_to_plot_sparc_slope_tail_hist(self):
        sentinel = {"figure_path": "x", "n": 0}
        with patch("scripts.mass_split_analysis._main", return_value=sentinel) as mock:
            result = mass_split_analysis_main([])
        mock.assert_called_once_with([])
        assert result is sentinel

    def test_passes_argv_none(self):
        with patch("scripts.mass_split_analysis._main", return_value={}) as mock:
            mass_split_analysis_main(None)
        mock.assert_called_once_with(None)

    def test_returns_dict(self):
        with patch("scripts.mass_split_analysis._main", return_value={}):
            result = mass_split_analysis_main([])
        assert isinstance(result, dict)

    def test_passes_arbitrary_argv(self):
        with patch("scripts.mass_split_analysis._main", return_value={}) as mock:
            mass_split_analysis_main(["--logm-cut", "10.1"])
        mock.assert_called_once_with(["--logm-cut", "10.1"])


class TestEnvMassScan:
    def test_delegates_to_plot_env_mass_scan(self):
        sentinel = {"scan_df": None, "figure_path": "x"}
        with patch("scripts.env_mass_scan._main", return_value=sentinel) as mock:
            result = env_mass_scan_main([])
        mock.assert_called_once_with([])
        assert result is sentinel

    def test_passes_argv_none(self):
        with patch("scripts.env_mass_scan._main", return_value={}) as mock:
            env_mass_scan_main(None)
        mock.assert_called_once_with(None)

    def test_returns_dict(self):
        with patch("scripts.env_mass_scan._main", return_value={}):
            result = env_mass_scan_main([])
        assert isinstance(result, dict)

    def test_passes_arbitrary_argv(self):
        with patch("scripts.env_mass_scan._main", return_value={}) as mock:
            env_mass_scan_main(["--n-min", "5"])
        mock.assert_called_once_with(["--n-min", "5"])


# ---------------------------------------------------------------------------
# run_full_pipeline — PIPELINE_STEPS / STEP_KEYS constants
# ---------------------------------------------------------------------------

class TestFullPipelineConstants:
    def test_pipeline_steps_is_list(self):
        assert isinstance(PIPELINE_STEPS, list)

    def test_pipeline_steps_has_four_entries(self):
        assert len(PIPELINE_STEPS) == 4

    def test_each_step_is_three_tuple(self):
        for step in PIPELINE_STEPS:
            assert len(step) == 3

    def test_step_scripts_reference_named_entry_points(self):
        scripts = {script for _, _, script in PIPELINE_STEPS}
        assert "scripts/compute_slope_tail.py" in scripts
        assert "scripts/build_master_catalog.py" in scripts
        assert "scripts/mass_split_analysis.py" in scripts
        assert "scripts/env_mass_scan.py" in scripts

    def test_step_keys_list_matches_pipeline_steps(self):
        assert STEP_KEYS == [key for key, _, _ in PIPELINE_STEPS]

    def test_compute_slope_tail_key_present(self):
        assert "compute_slope_tail" in STEP_KEYS

    def test_build_master_catalog_key_present(self):
        assert "build_master_catalog" in STEP_KEYS

    def test_mass_split_analysis_key_present(self):
        assert "mass_split_analysis" in STEP_KEYS

    def test_env_mass_scan_key_present(self):
        assert "env_mass_scan" in STEP_KEYS

    def test_pipeline_order(self):
        keys = STEP_KEYS
        assert keys.index("compute_slope_tail") < keys.index("build_master_catalog")
        assert keys.index("build_master_catalog") < keys.index("mass_split_analysis")
        assert keys.index("mass_split_analysis") < keys.index("env_mass_scan")


# ---------------------------------------------------------------------------
# run_full_pipeline — run_step
# ---------------------------------------------------------------------------

class TestFullPipelineRunStep:
    def test_dry_run_returns_success(self):
        result = run_step("echo test", dry_run=True)
        assert result["success"] is True
        assert result["returncode"] == 0

    def test_dry_run_cmd_stored(self):
        result = run_step("echo test", dry_run=True)
        assert result["cmd"] == "echo test"

    def test_list_cmd_dry_run(self):
        result = run_step(["echo", "hello"], dry_run=True)
        assert result["success"] is True

    def test_successful_subprocess(self):
        with patch("subprocess.run", return_value=_fake_completed(0)):
            result = run_step("echo ok", check=True, dry_run=False)
        assert result["success"] is True

    def test_failed_check_true_raises(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            with pytest.raises(RuntimeError):
                run_step("bad", check=True, dry_run=False)

    def test_failed_check_false_no_raise(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            result = run_step("bad", check=False, dry_run=False)
        assert result["success"] is False
        assert result["returncode"] == 1


# ---------------------------------------------------------------------------
# run_full_pipeline — run_pipeline
# ---------------------------------------------------------------------------

class TestFullPipelineRunPipeline:
    def test_returns_list(self):
        assert isinstance(run_pipeline(dry_run=True), list)

    def test_default_four_steps(self):
        assert len(run_pipeline(dry_run=True)) == 4

    def test_all_dry_run_succeed(self):
        assert all(r["success"] for r in run_pipeline(dry_run=True))

    def test_each_result_has_key(self):
        for r in run_pipeline(dry_run=True):
            assert "key" in r

    def test_result_keys_match_step_keys(self):
        results = run_pipeline(dry_run=True)
        assert [r["key"] for r in results] == STEP_KEYS

    def test_skip_one_step(self):
        results = run_pipeline(skip={"compute_slope_tail"}, dry_run=True)
        assert len(results) == 3
        assert all(r["key"] != "compute_slope_tail" for r in results)

    def test_skip_all_steps(self):
        assert run_pipeline(skip=set(STEP_KEYS), dry_run=True) == []

    def test_check_false_continues_on_failure(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            results = run_pipeline(check=False, dry_run=False)
        assert len(results) == 4
        assert all(not r["success"] for r in results)

    def test_all_steps_succeed_with_mock(self):
        with patch("subprocess.run", return_value=_fake_completed(0)):
            results = run_pipeline(dry_run=False)
        assert all(r["success"] for r in results)

    def test_default_python_is_sys_executable(self):
        captured = []

        def cap(cmd, **kw):
            captured.append(cmd)
            return _fake_completed(0)

        with patch("subprocess.run", side_effect=cap):
            run_pipeline(dry_run=False)

        assert all(c[0] == sys.executable for c in captured)


# ---------------------------------------------------------------------------
# run_full_pipeline — main
# ---------------------------------------------------------------------------

class TestFullPipelineMain:
    def test_returns_dict(self):
        assert isinstance(pipeline_main(["--dry-run"]), dict)

    def test_required_keys(self):
        result = pipeline_main(["--dry-run"])
        assert {"steps", "n_run", "n_ok", "success"}.issubset(result)

    def test_dry_run_all_ok(self):
        result = pipeline_main(["--dry-run"])
        assert result["success"] is True
        assert result["n_ok"] == 4
        assert result["n_run"] == 4

    def test_skip_one(self):
        result = pipeline_main(["--dry-run", "--skip", "compute_slope_tail"])
        assert result["n_run"] == 3

    def test_no_check_continues_on_failure(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            result = pipeline_main(["--no-check"])
        assert result["n_run"] == 4
        assert result["n_ok"] == 0
        assert result["success"] is False

    def test_success_true_on_all_passes(self):
        with patch("subprocess.run", return_value=_fake_completed(0)):
            result = pipeline_main([])
        assert result["success"] is True

    def test_custom_python_arg(self):
        captured = []

        def cap(cmd, **kw):
            captured.append(cmd)
            return _fake_completed(0)

        with patch("subprocess.run", side_effect=cap):
            pipeline_main(["--python", "/usr/bin/python3.11"])

        assert all(c[0] == "/usr/bin/python3.11" for c in captured)
