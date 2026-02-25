"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Covers both audit modes (global and radial) using synthetic datasets so that
no real SPARC download is required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_scm import (
    audit_global,
    audit_radial,
    main,
    CHI2_MEDIAN_PASS,
    CHI2_FRAC_GOOD_THRESHOLD,
    CHI2_FRAC_PASS,
    SLOPE_EXPECTED,
    MIN_DEEP_POINTS,
    SLOPE_SIGMA_TOL,
    DEEP_THRESHOLD_DEFAULT,
    A0_DEFAULT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_global_csv(tmp_path: Path, n: int = 30,
                     chi2_values: np.ndarray | None = None,
                     ud_values: np.ndarray | None = None) -> Path:
    """Write a synthetic per_galaxy_summary.csv."""
    rng = np.random.default_rng(0)
    if chi2_values is None:
        chi2_values = rng.uniform(0.5, 2.5, n)
    else:
        n = len(chi2_values)
    if ud_values is None:
        ud_values = rng.uniform(0.5, 3.0, n)
    df = pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "upsilon_disk": ud_values,
        "chi2_reduced": chi2_values,
        "n_points": np.full(n, 20, dtype=int),
        "Vflat_kms": np.linspace(80.0, 300.0, n),
        "M_bar_BTFR_Msun": np.linspace(1e9, 1e11, n),
    })
    p = tmp_path / "per_galaxy_summary.csv"
    df.to_csv(p, index=False)
    return p


def _make_radial_csv(tmp_path: Path,
                     n_deep: int = 100, n_shallow: int = 50,
                     planted_slope: float = 0.5,
                     noise_std: float = 0.005,
                     g0: float = A0_DEFAULT) -> Path:
    """Write a synthetic universal_term_comparison_full.csv."""
    rng = np.random.default_rng(42)
    g_bar_deep = rng.uniform(0.001 * g0, 0.29 * g0, n_deep)
    g_bar_shallow = rng.uniform(g0, 10.0 * g0, n_shallow)
    g_bar = np.concatenate([g_bar_deep, g_bar_shallow])
    log_gbar = np.log10(g_bar)
    log_gobs = (planted_slope * log_gbar
                + 0.5 * np.log10(g0)
                + rng.normal(0, noise_std, len(g_bar)))
    df = pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(len(g_bar))],
        "r_kpc": rng.uniform(1.0, 20.0, len(g_bar)),
        "g_bar": g_bar,
        "g_obs": 10.0 ** log_gobs,
        "log_g_bar": log_gbar,
        "log_g_obs": log_gobs,
    })
    p = tmp_path / "universal_term_comparison_full.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# audit_global unit tests
# ---------------------------------------------------------------------------

class TestAuditGlobal:
    def test_returns_verdict_key(self, tmp_path):
        csv = _make_global_csv(tmp_path)
        result = audit_global(csv)
        assert "verdict" in result
        assert result["verdict"] in ("PASS", "FAIL")

    def test_returns_required_metric_keys(self, tmp_path):
        csv = _make_global_csv(tmp_path)
        result = audit_global(csv)
        for key in ("n_galaxies", "chi2_median", "frac_good_fit",
                    "upsilon_disk_median", "verdict", "verdict_reason"):
            assert key in result, f"Missing key: {key}"

    def test_pass_when_low_chi2(self, tmp_path):
        """All galaxies with chi2 well below threshold → PASS."""
        chi2 = np.full(40, 1.0)  # median = 1.0 < 3.0, all < 2.0
        csv = _make_global_csv(tmp_path, chi2_values=chi2)
        result = audit_global(csv)
        assert result["verdict"] == "PASS"
        assert result["crit_chi2_median_pass"] is True
        assert result["crit_frac_good_pass"] is True

    def test_fail_when_high_chi2_median(self, tmp_path):
        """Median chi2 above threshold → FAIL."""
        chi2 = np.full(40, 5.0)  # median = 5.0 > 3.0
        csv = _make_global_csv(tmp_path, chi2_values=chi2)
        result = audit_global(csv)
        assert result["verdict"] == "FAIL"
        assert result["crit_chi2_median_pass"] is False

    def test_fail_when_frac_good_too_low(self, tmp_path):
        """Median chi2 OK but almost no galaxies with chi2 < 2 → FAIL."""
        # 10 galaxies with chi2=1.0, 90 with chi2=2.5
        chi2 = np.concatenate([np.full(10, 1.0), np.full(90, 2.5)])
        # median ≈ 2.5 which is < 3.0 → criterion 1 passes
        # but fraction below 2.0 = 10/100 = 0.10 < 0.30 → criterion 2 fails
        csv = _make_global_csv(tmp_path, chi2_values=chi2)
        result = audit_global(csv)
        assert result["crit_frac_good_pass"] is False
        assert result["verdict"] == "FAIL"

    def test_n_galaxies_correct(self, tmp_path):
        csv = _make_global_csv(tmp_path, n=25)
        result = audit_global(csv)
        assert result["n_galaxies"] == 25

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "upsilon_disk": [1.0]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            audit_global(p)

    def test_mode_is_global(self, tmp_path):
        csv = _make_global_csv(tmp_path)
        result = audit_global(csv)
        assert result["mode"] == "global"

    def test_frac_good_between_0_and_1(self, tmp_path):
        csv = _make_global_csv(tmp_path)
        result = audit_global(csv)
        assert 0.0 <= result["frac_good_fit"] <= 1.0


# ---------------------------------------------------------------------------
# audit_radial unit tests
# ---------------------------------------------------------------------------

class TestAuditRadial:
    def test_returns_verdict_key(self, tmp_path):
        csv = _make_radial_csv(tmp_path)
        result = audit_radial(csv)
        assert "verdict" in result
        assert result["verdict"] in ("PASS", "FAIL")

    def test_returns_required_metric_keys(self, tmp_path):
        csv = _make_radial_csv(tmp_path)
        result = audit_radial(csv)
        for key in ("n_total", "n_deep", "slope", "stderr",
                    "verdict", "verdict_reason", "crit_deep_points_pass",
                    "crit_slope_pass"):
            assert key in result, f"Missing key: {key}"

    def test_pass_when_mond_slope(self, tmp_path):
        """Data with planted β = 0.5 and many deep points → PASS."""
        csv = _make_radial_csv(tmp_path, n_deep=200, n_shallow=50,
                                planted_slope=0.5, noise_std=0.002)
        result = audit_radial(csv)
        assert result["crit_deep_points_pass"] is True
        assert result["verdict"] == "PASS"

    def test_fail_when_slope_deviates(self, tmp_path):
        """Strongly deviated slope → FAIL."""
        # β = 1.0 (Newtonian rather than deep-MOND) with tiny noise
        csv = _make_radial_csv(tmp_path, n_deep=300, n_shallow=50,
                                planted_slope=1.0, noise_std=0.001)
        result = audit_radial(csv)
        assert result["crit_slope_pass"] is False
        assert result["verdict"] == "FAIL"

    def test_fail_when_too_few_deep_points(self, tmp_path):
        """Only 5 deep points → FAIL (below MIN_DEEP_POINTS=10)."""
        csv = _make_radial_csv(tmp_path, n_deep=5, n_shallow=100)
        result = audit_radial(csv)
        assert result["n_deep"] == 5
        assert result["crit_deep_points_pass"] is False
        assert result["verdict"] == "FAIL"

    def test_fail_when_zero_deep_points(self, tmp_path):
        """No deep points at all → FAIL (n_deep < 2)."""
        # All points in Newtonian regime
        g0 = A0_DEFAULT
        g_bar = np.linspace(2.0 * g0, 10.0 * g0, 100)
        log_gbar = np.log10(g_bar)
        log_gobs = log_gbar
        df = pd.DataFrame({
            "galaxy": "X", "r_kpc": 5.0,
            "g_bar": g_bar, "g_obs": 10**log_gobs,
            "log_g_bar": log_gbar, "log_g_obs": log_gobs,
        })
        p = tmp_path / "no_deep.csv"
        df.to_csv(p, index=False)
        result = audit_radial(p)
        assert result["n_deep"] == 0
        assert result["verdict"] == "FAIL"

    def test_slope_approximately_recovered(self, tmp_path):
        """Slope value in result should be close to planted value."""
        csv = _make_radial_csv(tmp_path, n_deep=300, n_shallow=50,
                                planted_slope=0.5, noise_std=0.002)
        result = audit_radial(csv)
        assert result["slope"] == pytest.approx(0.5, abs=0.05)

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "g_bar": [1e-11]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            audit_radial(p)

    def test_mode_is_radial(self, tmp_path):
        csv = _make_radial_csv(tmp_path)
        result = audit_radial(csv)
        assert result["mode"] == "radial"

    def test_deep_frac_in_range(self, tmp_path):
        csv = _make_radial_csv(tmp_path)
        result = audit_radial(csv)
        assert 0.0 <= result["deep_frac"] <= 1.0

    def test_custom_deep_threshold(self, tmp_path):
        """Higher deep_threshold should include more points."""
        csv = _make_radial_csv(tmp_path, n_deep=100, n_shallow=100)
        r_low = audit_radial(csv, deep_threshold=0.1)
        r_high = audit_radial(csv, deep_threshold=0.5)
        assert r_high["n_deep"] >= r_low["n_deep"]


# ---------------------------------------------------------------------------
# main() CLI tests
# ---------------------------------------------------------------------------

class TestMainGlobalMode:
    def test_pass_returns_0(self, tmp_path):
        csv = _make_global_csv(tmp_path, chi2_values=np.full(40, 1.0))
        out = tmp_path / "out_pass"
        rc = main(["--global-csv", str(csv), "--out", str(out)])
        assert rc == 0

    def test_fail_returns_1(self, tmp_path):
        csv = _make_global_csv(tmp_path, chi2_values=np.full(40, 6.0))
        out = tmp_path / "out_fail"
        rc = main(["--global-csv", str(csv), "--out", str(out)])
        assert rc == 1

    def test_writes_verdict_file(self, tmp_path):
        csv = _make_global_csv(tmp_path)
        out = tmp_path / "out_verd"
        main(["--global-csv", str(csv), "--out", str(out)])
        assert (out / "audit_verdict.txt").exists()

    def test_writes_metrics_csv(self, tmp_path):
        csv = _make_global_csv(tmp_path)
        out = tmp_path / "out_met"
        main(["--global-csv", str(csv), "--out", str(out)])
        assert (out / "audit_metrics.csv").exists()

    def test_metrics_csv_has_verdict_column(self, tmp_path):
        csv = _make_global_csv(tmp_path)
        out = tmp_path / "out_vcol"
        main(["--global-csv", str(csv), "--out", str(out)])
        df = pd.read_csv(out / "audit_metrics.csv")
        assert "verdict" in df.columns

    def test_verdict_txt_contains_verdict_word(self, tmp_path):
        csv = _make_global_csv(tmp_path)
        out = tmp_path / "out_vtxt"
        main(["--global-csv", str(csv), "--out", str(out)])
        text = (out / "audit_verdict.txt").read_text(encoding="utf-8")
        assert "VERDICT" in text

    def test_missing_file_returns_2(self, tmp_path):
        rc = main(["--global-csv", str(tmp_path / "nonexistent.csv"),
                   "--out", str(tmp_path / "out")])
        assert rc == 2

    def test_no_source_raises_systemexit(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--out", str(tmp_path)])


class TestMainRadialMode:
    def test_pass_returns_0(self, tmp_path):
        csv = _make_radial_csv(tmp_path, n_deep=300, n_shallow=50,
                                planted_slope=0.5, noise_std=0.002)
        out = tmp_path / "out_pass"
        rc = main(["--radial-csv", str(csv), "--out", str(out)])
        assert rc == 0

    def test_fail_returns_1(self, tmp_path):
        csv = _make_radial_csv(tmp_path, n_deep=300, n_shallow=50,
                                planted_slope=1.0, noise_std=0.001)
        out = tmp_path / "out_fail"
        rc = main(["--radial-csv", str(csv), "--out", str(out)])
        assert rc == 1

    def test_writes_verdict_file(self, tmp_path):
        csv = _make_radial_csv(tmp_path)
        out = tmp_path / "out_verd"
        main(["--radial-csv", str(csv), "--out", str(out)])
        assert (out / "audit_verdict.txt").exists()

    def test_writes_metrics_csv(self, tmp_path):
        csv = _make_radial_csv(tmp_path)
        out = tmp_path / "out_met"
        main(["--radial-csv", str(csv), "--out", str(out)])
        assert (out / "audit_metrics.csv").exists()

    def test_verdict_txt_contains_sha256(self, tmp_path):
        csv = _make_radial_csv(tmp_path)
        out = tmp_path / "out_sha"
        main(["--radial-csv", str(csv), "--out", str(out)])
        text = (out / "audit_verdict.txt").read_text(encoding="utf-8")
        assert "SHA-256" in text

    def test_deep_threshold_flag(self, tmp_path):
        csv = _make_radial_csv(tmp_path, n_deep=200, n_shallow=50)
        out = tmp_path / "out_thr"
        main(["--radial-csv", str(csv), "--out", str(out),
              "--deep-threshold", "0.5"])
        df = pd.read_csv(out / "audit_metrics.csv")
        assert float(df["deep_threshold"].iloc[0]) == pytest.approx(0.5)

    def test_missing_file_returns_2(self, tmp_path):
        rc = main(["--radial-csv", str(tmp_path / "nonexistent.csv"),
                   "--out", str(tmp_path / "out")])
        assert rc == 2

    def test_mutually_exclusive_both_raises(self, tmp_path):
        csv = _make_radial_csv(tmp_path)
        with pytest.raises(SystemExit):
            main(["--global-csv", str(csv), "--radial-csv", str(csv),
                  "--out", str(tmp_path / "out")])


# ---------------------------------------------------------------------------
# Integration: run_pipeline → audit_scm (both modes)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Verify that outputs from run_pipeline() can be fed directly to
    audit_scm without error."""

    @pytest.fixture(scope="class")
    def pipeline_outputs(self, tmp_path_factory):
        from src.scm_analysis import run_pipeline

        root = tmp_path_factory.mktemp("audit_intg")
        rng = np.random.default_rng(9)
        n_gal, n_pts = 15, 20
        names = [f"AUD{i:02d}" for i in range(n_gal)]
        v_flats = np.linspace(100.0, 280.0, n_gal)

        pd.DataFrame({
            "Galaxy": names,
            "D": np.linspace(5, 60, n_gal),
            "Inc": np.linspace(30, 80, n_gal),
            "L36": 1e9 * np.arange(1, n_gal + 1, dtype=float),
            "Vflat": v_flats,
            "e_Vflat": np.full(n_gal, 5.0),
        }).to_csv(root / "SPARC_Lelli2016c.csv", index=False)

        for name, vf in zip(names, v_flats):
            r = np.linspace(0.5, 15.0, n_pts)
            rc = pd.DataFrame({
                "r": r,
                "v_obs": np.full(n_pts, vf) + rng.normal(0, 3, n_pts),
                "v_obs_err": np.full(n_pts, 5.0),
                "v_gas": 0.3 * vf * np.ones(n_pts),
                "v_disk": 0.75 * vf * np.ones(n_pts),
                "v_bul": np.zeros(n_pts),
                "SBdisk": np.zeros(n_pts),
                "SBbul": np.zeros(n_pts),
            })
            rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)

        out = tmp_path_factory.mktemp("audit_pipe_out")
        run_pipeline(root, out, verbose=False)
        return out

    def test_global_audit_runs_without_error(self, pipeline_outputs, tmp_path):
        global_csv = pipeline_outputs / "per_galaxy_summary.csv"
        rc = main(["--global-csv", str(global_csv), "--out", str(tmp_path / "g")])
        assert rc in (0, 1)  # either verdict is acceptable; key is no crash

    def test_radial_audit_runs_without_error(self, pipeline_outputs, tmp_path):
        radial_csv = pipeline_outputs / "universal_term_comparison_full.csv"
        rc = main(["--radial-csv", str(radial_csv), "--out", str(tmp_path / "r")])
        assert rc in (0, 1)

    def test_global_metrics_csv_has_n_galaxies(self, pipeline_outputs, tmp_path):
        global_csv = pipeline_outputs / "per_galaxy_summary.csv"
        out = tmp_path / "gout"
        main(["--global-csv", str(global_csv), "--out", str(out)])
        df = pd.read_csv(out / "audit_metrics.csv")
        assert int(df["n_galaxies"].iloc[0]) == 15

    def test_radial_metrics_csv_has_n_total(self, pipeline_outputs, tmp_path):
        radial_csv = pipeline_outputs / "universal_term_comparison_full.csv"
        out = tmp_path / "rout"
        main(["--radial-csv", str(radial_csv), "--out", str(out)])
        df = pd.read_csv(out / "audit_metrics.csv")
        assert int(df["n_total"].iloc[0]) > 0
