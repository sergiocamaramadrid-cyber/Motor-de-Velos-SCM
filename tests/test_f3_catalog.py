"""
tests/test_f3_catalog.py — Tests for scripts/f3_catalog.py.

Validates per-galaxy friction slope computation, Velo Inerte anomaly flagging,
catalog column contract, and the CLI entry-point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.f3_catalog import (
    per_galaxy_friction_slope,
    build_f3_catalog,
    format_summary,
    main,
    A0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
    EXPECTED_SLOPE,
    MIN_DEEP_POINTS,
    ANOMALY_SIGMA,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_per_point_csv(
    tmp_path: Path,
    n_galaxies: int = 5,
    n_deep_per_galaxy: int = 20,
    n_shallow_per_galaxy: int = 20,
    planted_slope: float = 0.5,
    noise_std: float = 0.005,
    g0: float = A0_DEFAULT,
    anomaly_galaxy: str | None = None,
    anomaly_slope: float = 0.3,
) -> Path:
    """Build a synthetic per-radial-point CSV with a known planted slope.

    Parameters
    ----------
    anomaly_galaxy : str or None
        If provided and it matches one of the galaxy names, that galaxy
        gets ``anomaly_slope`` planted instead of ``planted_slope``.
    """
    rng = np.random.default_rng(42)
    rows = []
    galaxy_names = [f"G{i:03d}" for i in range(n_galaxies)]
    for name in galaxy_names:
        slope = anomaly_slope if name == anomaly_galaxy else planted_slope

        g_bar_deep = rng.uniform(0.001 * g0, 0.28 * g0, n_deep_per_galaxy)
        g_bar_shallow = rng.uniform(g0, 5.0 * g0, n_shallow_per_galaxy)
        g_bar_all = np.concatenate([g_bar_deep, g_bar_shallow])

        log_gbar = np.log10(g_bar_all)
        log_gobs = (slope * log_gbar
                    + 0.5 * np.log10(g0)
                    + rng.normal(0, noise_std, len(g_bar_all)))

        for i in range(len(g_bar_all)):
            rows.append({
                "galaxy": name,
                "r_kpc": float(i + 1),
                "g_bar": float(g_bar_all[i]),
                "g_obs": float(10.0 ** log_gobs[i]),
                "log_g_bar": float(log_gbar[i]),
                "log_g_obs": float(log_gobs[i]),
            })

    df = pd.DataFrame(rows)
    p = tmp_path / "universal_term_comparison_full.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# per_galaxy_friction_slope
# ---------------------------------------------------------------------------

class TestPerGalaxyFrictionSlope:
    def _make_group(self, g0=A0_DEFAULT, n_deep=30, n_shallow=20,
                    slope=0.5, noise=0.002):
        rng = np.random.default_rng(7)
        g_deep = rng.uniform(0.001 * g0, 0.28 * g0, n_deep)
        g_shallow = rng.uniform(g0, 5.0 * g0, n_shallow)
        g_bar = np.concatenate([g_deep, g_shallow])
        log_gbar = np.log10(g_bar)
        log_gobs = slope * log_gbar + 0.5 * np.log10(g0) + rng.normal(0, noise, len(g_bar))
        return pd.DataFrame({
            "g_bar": g_bar,
            "log_g_bar": log_gbar,
            "log_g_obs": log_gobs,
        })

    def test_returns_required_keys(self):
        group = self._make_group()
        result = per_galaxy_friction_slope(group)
        required = {
            "n_points", "n_deep", "deep_frac",
            "friction_slope", "friction_slope_err", "delta_from_mond",
            "r2_deep", "velo_inerte_flag",
        }
        assert required.issubset(set(result.keys()))

    def test_recovers_planted_slope(self):
        group = self._make_group(slope=0.5, n_deep=100, noise=0.001)
        result = per_galaxy_friction_slope(group)
        assert result["friction_slope"] == pytest.approx(0.5, abs=0.03)

    def test_delta_from_mond_consistent(self):
        group = self._make_group(slope=0.4, n_deep=50, noise=0.001)
        result = per_galaxy_friction_slope(group)
        assert result["delta_from_mond"] == pytest.approx(
            result["friction_slope"] - EXPECTED_SLOPE, abs=1e-6
        )

    def test_nan_when_no_deep_points(self):
        g0 = A0_DEFAULT
        # All points in Newtonian regime
        g_bar = np.linspace(2.0 * g0, 10.0 * g0, 30)
        group = pd.DataFrame({
            "g_bar": g_bar,
            "log_g_bar": np.log10(g_bar),
            "log_g_obs": np.log10(g_bar),
        })
        result = per_galaxy_friction_slope(group)
        assert np.isnan(result["friction_slope"])
        assert result["n_deep"] == 0
        assert result["velo_inerte_flag"] is False

    def test_no_anomaly_for_mond_slope(self):
        """Pure β=0.5 data with many deep points must NOT be flagged."""
        group = self._make_group(slope=0.5, n_deep=MIN_DEEP_POINTS + 5, noise=0.001)
        result = per_galaxy_friction_slope(group)
        assert result["velo_inerte_flag"] is False

    def test_anomaly_flag_for_deviant_slope(self):
        """A significantly deviant slope (0.2) with many deep points MUST be flagged."""
        group = self._make_group(slope=0.2, n_deep=MIN_DEEP_POINTS + 20, noise=1e-5)
        result = per_galaxy_friction_slope(group)
        assert result["velo_inerte_flag"] is True

    def test_r2_between_zero_and_one(self):
        group = self._make_group()
        result = per_galaxy_friction_slope(group)
        if not np.isnan(result["r2_deep"]):
            assert 0.0 <= result["r2_deep"] <= 1.0

    def test_deep_frac_sums_correctly(self):
        g0 = A0_DEFAULT
        threshold = DEEP_THRESHOLD_DEFAULT
        rng = np.random.default_rng(3)
        g_bar = np.concatenate([
            rng.uniform(0.001 * g0, 0.1 * g0, 10),  # deep (distinct values)
            np.full(20, 2.0 * g0),                   # shallow
        ])
        log_gbar = np.log10(g_bar)
        group = pd.DataFrame({
            "g_bar": g_bar,
            "log_g_bar": log_gbar,
            "log_g_obs": log_gbar,
        })
        result = per_galaxy_friction_slope(group, a0=g0, deep_threshold=threshold)
        assert result["n_deep"] == 10
        assert result["deep_frac"] == pytest.approx(10 / 30)


# ---------------------------------------------------------------------------
# build_f3_catalog
# ---------------------------------------------------------------------------

class TestBuildF3Catalog:
    CATALOG_COLS = [
        "galaxy", "n_points", "n_deep", "deep_frac",
        "friction_slope", "friction_slope_err", "delta_from_mond",
        "r2_deep", "velo_inerte_flag",
    ]

    def test_returns_dataframe(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        cat = build_f3_catalog(csv)
        assert isinstance(cat, pd.DataFrame)

    def test_column_contract(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        cat = build_f3_catalog(csv)
        assert cat.columns.tolist() == self.CATALOG_COLS

    def test_one_row_per_galaxy(self, tmp_path):
        n = 7
        csv = _make_per_point_csv(tmp_path, n_galaxies=n)
        cat = build_f3_catalog(csv)
        assert len(cat) == n

    def test_sorted_by_galaxy(self, tmp_path):
        csv = _make_per_point_csv(tmp_path, n_galaxies=5)
        cat = build_f3_catalog(csv)
        assert list(cat["galaxy"]) == sorted(cat["galaxy"].tolist())

    def test_n_points_dtype_int(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        cat = build_f3_catalog(csv)
        assert pd.api.types.is_integer_dtype(cat["n_points"])

    def test_velo_inerte_flag_dtype_bool(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        cat = build_f3_catalog(csv)
        assert cat["velo_inerte_flag"].dtype == bool

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_f3_catalog(tmp_path / "nonexistent.csv")

    def test_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "g_bar": [1e-11]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            build_f3_catalog(bad)

    def test_anomaly_galaxy_flagged(self, tmp_path):
        """Galaxy G000 with slope 0.2 should be flagged; others should not."""
        csv = _make_per_point_csv(
            tmp_path,
            n_galaxies=5,
            n_deep_per_galaxy=MIN_DEEP_POINTS + 10,
            planted_slope=0.5,
            noise_std=1e-5,
            anomaly_galaxy="G000",
            anomaly_slope=0.2,
        )
        cat = build_f3_catalog(csv)
        assert cat.loc[cat["galaxy"] == "G000", "velo_inerte_flag"].iloc[0]  # truthy
        # Other galaxies should not be flagged (slope ≈ 0.5)
        others = cat.loc[cat["galaxy"] != "G000", "velo_inerte_flag"]
        assert not others.any()

    def test_friction_slope_near_planted_value(self, tmp_path):
        csv = _make_per_point_csv(
            tmp_path, n_galaxies=3, n_deep_per_galaxy=50,
            planted_slope=0.5, noise_std=0.002,
        )
        cat = build_f3_catalog(csv)
        valid = cat["friction_slope"].dropna()
        assert len(valid) == 3
        assert (valid - 0.5).abs().max() < 0.1


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def _make_catalog(self):
        return pd.DataFrame({
            "galaxy": ["G000", "G001", "G002"],
            "n_points": [40, 40, 40],
            "n_deep": [20, 20, 0],
            "deep_frac": [0.5, 0.5, 0.0],
            "friction_slope": [0.50, 0.51, float("nan")],
            "friction_slope_err": [0.02, 0.02, float("nan")],
            "delta_from_mond": [0.00, 0.01, float("nan")],
            "r2_deep": [0.95, 0.96, float("nan")],
            "velo_inerte_flag": [False, False, False],
        })

    def test_contains_mean_friction_slope(self):
        cat = self._make_catalog()
        lines = format_summary(cat, A0_DEFAULT, DEEP_THRESHOLD_DEFAULT,
                               "in.csv", "out.csv")
        combined = "\n".join(lines)
        assert "Mean" in combined

    def test_contains_anomaly_count(self):
        cat = self._make_catalog()
        lines = format_summary(cat, A0_DEFAULT, DEEP_THRESHOLD_DEFAULT,
                               "in.csv", "out.csv")
        combined = "\n".join(lines)
        assert "Velo Inerte anomalies" in combined

    def test_flagged_galaxies_listed(self):
        cat = self._make_catalog()
        cat.loc[0, "velo_inerte_flag"] = True
        lines = format_summary(cat, A0_DEFAULT, DEEP_THRESHOLD_DEFAULT,
                               "in.csv", "out.csv")
        combined = "\n".join(lines)
        assert "G000" in combined


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_writes_catalog_csv(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        out = tmp_path / "f3_catalog.csv"
        main(["--csv", str(csv), "--out", str(out)])
        assert out.exists()

    def test_catalog_has_correct_columns(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        out = tmp_path / "f3_catalog.csv"
        main(["--csv", str(csv), "--out", str(out)])
        cat = pd.read_csv(out)
        expected_cols = [
            "galaxy", "n_points", "n_deep", "deep_frac",
            "friction_slope", "friction_slope_err", "delta_from_mond",
            "r2_deep", "velo_inerte_flag",
        ]
        assert cat.columns.tolist() == expected_cols

    def test_returns_dataframe(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        out = tmp_path / "f3_catalog.csv"
        result = main(["--csv", str(csv), "--out", str(out)])
        assert isinstance(result, pd.DataFrame)

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "nope.csv"),
                  "--out", str(tmp_path / "out.csv")])

    def test_custom_a0(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        out = tmp_path / "f3_a0.csv"
        cat = main(["--csv", str(csv), "--out", str(out), "--a0", "0.8e-10"])
        assert isinstance(cat, pd.DataFrame)

    def test_custom_deep_threshold(self, tmp_path):
        csv = _make_per_point_csv(tmp_path)
        out = tmp_path / "f3_thr.csv"
        cat = main(["--csv", str(csv), "--out", str(out), "--deep-threshold", "0.1"])
        assert isinstance(cat, pd.DataFrame)


# ---------------------------------------------------------------------------
# Integration: run_pipeline → f3_catalog
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Verify that run_pipeline() generates a CSV consumable by f3_catalog."""

    @pytest.fixture(scope="class")
    def pipeline_csv(self, tmp_path_factory):
        from src.scm_analysis import run_pipeline

        root = tmp_path_factory.mktemp("f3_pipe")
        rng = np.random.default_rng(11)
        n_gal, n_pts = 8, 20
        names = [f"F{i:02d}" for i in range(n_gal)]
        v_flats = np.linspace(100.0, 250.0, n_gal)

        pd.DataFrame({
            "Galaxy": names,
            "D": np.linspace(5, 40, n_gal),
            "Inc": np.linspace(30, 70, n_gal),
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

        out = tmp_path_factory.mktemp("f3_pipe_out")
        run_pipeline(root, out, verbose=False)
        return out / "universal_term_comparison_full.csv"

    def test_pipeline_csv_accepted(self, pipeline_csv, tmp_path):
        out = tmp_path / "f3.csv"
        cat = main(["--csv", str(pipeline_csv), "--out", str(out)])
        assert len(cat) == 8

    def test_one_row_per_pipeline_galaxy(self, pipeline_csv, tmp_path):
        out = tmp_path / "f3b.csv"
        cat = main(["--csv", str(pipeline_csv), "--out", str(out)])
        assert cat["galaxy"].nunique() == len(cat)
