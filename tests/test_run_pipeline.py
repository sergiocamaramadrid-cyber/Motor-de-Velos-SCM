"""tests/test_run_pipeline.py — Tests for scripts/run_pipeline.py."""

from __future__ import annotations

import sys
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from scripts.run_pipeline import (
    PIPELINE_STEPS,
    STEP_KEYS,
    main,
    run_pipeline,
    run_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_completed(returncode: int = 0) -> MagicMock:
    """Return a mock CompletedProcess with the given returncode."""
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    return cp


# ---------------------------------------------------------------------------
# PIPELINE_STEPS / STEP_KEYS constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_pipeline_steps_is_list(self):
        assert isinstance(PIPELINE_STEPS, list)

    def test_pipeline_steps_not_empty(self):
        assert len(PIPELINE_STEPS) > 0

    def test_pipeline_steps_has_four_entries(self):
        assert len(PIPELINE_STEPS) == 4

    def test_each_step_is_three_tuple(self):
        for step in PIPELINE_STEPS:
            assert len(step) == 3, f"Step should be 3-tuple: {step}"

    def test_step_keys_are_strings(self):
        for key, _, _ in PIPELINE_STEPS:
            assert isinstance(key, str)

    def test_step_labels_are_strings(self):
        for _, label, _ in PIPELINE_STEPS:
            assert isinstance(label, str)

    def test_step_scripts_are_strings(self):
        for _, _, script in PIPELINE_STEPS:
            assert isinstance(script, str)

    def test_step_scripts_end_in_py(self):
        for _, _, script in PIPELINE_STEPS:
            assert script.endswith(".py"), f"Script should end in .py: {script}"

    def test_step_keys_list_matches_pipeline_steps(self):
        expected = [key for key, _, _ in PIPELINE_STEPS]
        assert STEP_KEYS == expected

    def test_slope_tail_step_present(self):
        keys = [key for key, _, _ in PIPELINE_STEPS]
        assert "slope_tail" in keys

    def test_master_catalog_step_present(self):
        keys = [key for key, _, _ in PIPELINE_STEPS]
        assert "master_catalog" in keys

    def test_slope_hist_step_present(self):
        keys = [key for key, _, _ in PIPELINE_STEPS]
        assert "slope_hist" in keys

    def test_env_scan_step_present(self):
        keys = [key for key, _, _ in PIPELINE_STEPS]
        assert "env_scan" in keys

    def test_pipeline_order(self):
        """slope_tail must precede master_catalog, which must precede the rest."""
        keys = STEP_KEYS
        assert keys.index("slope_tail") < keys.index("master_catalog")
        assert keys.index("master_catalog") < keys.index("slope_hist")
        assert keys.index("slope_hist") < keys.index("env_scan")


# ---------------------------------------------------------------------------
# run_step
# ---------------------------------------------------------------------------

class TestRunStep:
    def test_returns_dict(self):
        result = run_step("echo hello", dry_run=True)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = run_step("echo hello", dry_run=True)
        assert {"cmd", "returncode", "success"}.issubset(result)

    def test_dry_run_returncode_zero(self):
        result = run_step("echo hello", dry_run=True)
        assert result["returncode"] == 0

    def test_dry_run_success_true(self):
        result = run_step("echo hello", dry_run=True)
        assert result["success"] is True

    def test_dry_run_cmd_stored(self):
        result = run_step("echo hello world", dry_run=True)
        assert result["cmd"] == "echo hello world"

    def test_list_cmd_dry_run(self):
        result = run_step(["echo", "hello"], dry_run=True)
        assert result["success"] is True

    def test_successful_subprocess(self):
        with patch("subprocess.run", return_value=_fake_completed(0)) as mock_run:
            result = run_step("echo ok", check=True, dry_run=False)
        assert result["success"] is True
        assert result["returncode"] == 0
        mock_run.assert_called_once()

    def test_failed_subprocess_check_true(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            with pytest.raises(RuntimeError, match="Pipeline step failed"):
                run_step("python bad_script.py", check=True, dry_run=False)

    def test_failed_subprocess_check_false(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            result = run_step("python bad_script.py", check=False, dry_run=False)
        assert result["success"] is False
        assert result["returncode"] == 1

    def test_cmd_stored_as_string(self):
        with patch("subprocess.run", return_value=_fake_completed(0)):
            result = run_step(["python", "scripts/sparc_slope_tail.py"], dry_run=False)
        assert isinstance(result["cmd"], str)

    def test_list_cmd_joined_in_result(self):
        with patch("subprocess.run", return_value=_fake_completed(0)):
            result = run_step(["python", "myscript.py"], dry_run=False)
        assert "python" in result["cmd"]
        assert "myscript.py" in result["cmd"]


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_returns_list(self):
        results = run_pipeline(dry_run=True)
        assert isinstance(results, list)

    def test_default_four_steps(self):
        results = run_pipeline(dry_run=True)
        assert len(results) == 4

    def test_all_succeed_dry_run(self):
        results = run_pipeline(dry_run=True)
        assert all(r["success"] for r in results)

    def test_each_result_has_key(self):
        results = run_pipeline(dry_run=True)
        for r in results:
            assert "key" in r

    def test_each_result_has_label(self):
        results = run_pipeline(dry_run=True)
        for r in results:
            assert "label" in r

    def test_each_result_has_cmd(self):
        results = run_pipeline(dry_run=True)
        for r in results:
            assert "cmd" in r

    def test_result_keys_match_step_keys(self):
        results = run_pipeline(dry_run=True)
        assert [r["key"] for r in results] == STEP_KEYS

    def test_skip_one_step(self):
        results = run_pipeline(skip={"slope_tail"}, dry_run=True)
        assert len(results) == 3
        assert all(r["key"] != "slope_tail" for r in results)

    def test_skip_two_steps(self):
        results = run_pipeline(skip={"slope_tail", "env_scan"}, dry_run=True)
        assert len(results) == 2

    def test_skip_all_steps(self):
        results = run_pipeline(skip=set(STEP_KEYS), dry_run=True)
        assert len(results) == 0

    def test_custom_steps(self):
        custom = [("my_step", "My label", "scripts/sparc_slope_tail.py")]
        results = run_pipeline(steps=custom, dry_run=True)
        assert len(results) == 1
        assert results[0]["key"] == "my_step"

    def test_check_true_aborts_on_failure(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            with pytest.raises(RuntimeError):
                run_pipeline(check=True, dry_run=False)

    def test_check_false_continues_on_failure(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            results = run_pipeline(check=False, dry_run=False)
        assert len(results) == 4
        assert all(not r["success"] for r in results)

    def test_all_steps_succeed_with_mock(self):
        with patch("subprocess.run", return_value=_fake_completed(0)):
            results = run_pipeline(check=True, dry_run=False)
        assert len(results) == 4
        assert all(r["success"] for r in results)

    def test_custom_python_interpreter(self):
        captured = []

        def capture_run(cmd, **kwargs):
            captured.append(cmd)
            return _fake_completed(0)

        with patch("subprocess.run", side_effect=capture_run):
            run_pipeline(python="/usr/bin/python3.10", dry_run=False)

        for cmd in captured:
            assert cmd[0] == "/usr/bin/python3.10"

    def test_default_python_is_sys_executable(self):
        captured = []

        def capture_run(cmd, **kwargs):
            captured.append(cmd)
            return _fake_completed(0)

        with patch("subprocess.run", side_effect=capture_run):
            run_pipeline(dry_run=False)

        for cmd in captured:
            assert cmd[0] == sys.executable

    def test_result_order_matches_pipeline(self):
        results = run_pipeline(dry_run=True)
        for result, (key, _, _) in zip(results, PIPELINE_STEPS):
            assert result["key"] == key


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self):
        result = main(["--dry-run"])
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = main(["--dry-run"])
        assert {"steps", "n_run", "n_ok", "success"}.issubset(result)

    def test_dry_run_all_ok(self):
        result = main(["--dry-run"])
        assert result["success"] is True
        assert result["n_ok"] == 4
        assert result["n_run"] == 4

    def test_steps_is_list(self):
        result = main(["--dry-run"])
        assert isinstance(result["steps"], list)

    def test_n_run_equals_len_steps(self):
        result = main(["--dry-run"])
        assert result["n_run"] == len(result["steps"])

    def test_skip_one(self):
        result = main(["--dry-run", "--skip", "slope_tail"])
        assert result["n_run"] == 3
        assert all(s["key"] != "slope_tail" for s in result["steps"])

    def test_skip_two(self):
        result = main(["--dry-run", "--skip", "slope_tail", "--skip", "env_scan"])
        assert result["n_run"] == 2

    def test_no_check_continues_on_failure(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            result = main(["--no-check"])
        assert result["n_run"] == 4
        assert result["n_ok"] == 0
        assert result["success"] is False

    def test_success_false_on_all_failures(self):
        with patch("subprocess.run", return_value=_fake_completed(1)):
            result = main(["--no-check"])
        assert result["success"] is False

    def test_success_true_on_all_passes(self):
        with patch("subprocess.run", return_value=_fake_completed(0)):
            result = main([])
        assert result["success"] is True

    def test_custom_python_arg(self):
        captured = []

        def cap(cmd, **kw):
            captured.append(cmd)
            return _fake_completed(0)

        with patch("subprocess.run", side_effect=cap):
            main(["--python", "/usr/bin/python3.11"])

        assert all(c[0] == "/usr/bin/python3.11" for c in captured)

    def test_n_ok_counts_successes(self):
        call_count = [0]

        def alternating(cmd, **kw):
            call_count[0] += 1
            rc = 0 if call_count[0] % 2 == 0 else 1
            return _fake_completed(rc)

        with patch("subprocess.run", side_effect=alternating):
            result = main(["--no-check"])

        # 4 steps: alternating 1,0,1,0 → 2 ok
        assert result["n_ok"] == 2
        assert result["n_run"] == 4
