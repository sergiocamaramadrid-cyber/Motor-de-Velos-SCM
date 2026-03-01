"""
tests/test_f3_catalog.py — Tests for the F3 catalog pipeline.

Covers:
  1. Synthetic CI fixture (results/f3_catalog_synthetic_flat.csv) — verifies
     β ≈ 1 as expected for flat-rotation-curve synthetic data.
  2. generate_f3_catalog.py — per-galaxy β measurement script.
  3. f3_catalog_analysis.py — statistical analysis of the catalog.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_f3_catalog import (
    generate_f3_catalog,
    measure_galaxy_beta,
    A0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
    MIN_DEEP_POINTS_DEFAULT,
    main as generate_main,
)
from scripts.f3_catalog_analysis import (
    analyze_f3_catalog,
    format_analysis_report,
    main as analysis_main,
)

# ---------------------------------------------------------------------------
# Path to the committed synthetic CI fixture
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
FIXTURE_PATH = _REPO_ROOT / "results" / "f3_catalog_synthetic_flat.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_sparc_dir(
    tmp_path: Path,
    n_gal: int = 5,
    v_flats: list[float] | None = None,
    n_pts: int = 30,
    seed: int = 42,
) -> Path:
    """Create a minimal synthetic SPARC-like dataset with flat rotation curves.

    Velocities are chosen small (≤ 3.5 km/s) so that deep-regime points
    (g_bar < 0.3 × a0) exist at large radii.  With flat rotation curves both
    g_obs and g_bar scale as V²/r, giving β = 1 by construction.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if v_flats is None:
        v_flats = np.linspace(1.5, 3.5, n_gal).tolist()
    names = [f"T{i:02d}" for i in range(n_gal)]

    pd.DataFrame({
        "Galaxy": names,
        "D": np.linspace(5.0, 30.0, n_gal),
        "Inc": np.linspace(35.0, 65.0, n_gal),
        "L36": 1e9 * np.arange(1, n_gal + 1, dtype=float),
        "Vflat": v_flats,
        "e_Vflat": np.full(n_gal, 0.1),
    }).to_csv(tmp_path / "SPARC_Lelli2016c.csv", index=False)

    r = np.linspace(0.2, 15.0, n_pts)
    for name, vf in zip(names, v_flats):
        v_obs = np.full(n_pts, vf) + rng.normal(0, 0.02, n_pts)
        rc = pd.DataFrame({
            "r": r,
            "v_obs": np.clip(v_obs, 0.01, None),
            "v_obs_err": np.full(n_pts, 0.05),
            "v_gas": 0.3 * vf * np.ones(n_pts),
            "v_disk": 0.7 * vf * np.ones(n_pts),
            "v_bul": np.zeros(n_pts),
            "SBdisk": np.zeros(n_pts),
            "SBbul": np.zeros(n_pts),
        })
        rc.to_csv(tmp_path / f"{name}_rotmod.dat", sep=" ", index=False, header=False)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Synthetic CI fixture verification
# ---------------------------------------------------------------------------

class TestSyntheticFixture:
    """Verify the committed CI fixture (f3_catalog_synthetic_flat.csv)."""

    def test_fixture_exists(self):
        assert FIXTURE_PATH.exists(), (
            f"CI fixture not found: {FIXTURE_PATH}"
        )

    def test_fixture_columns(self):
        df = pd.read_csv(FIXTURE_PATH)
        required = {
            "galaxy", "beta", "beta_err", "intercept",
            "r_value", "p_value", "n_deep", "n_total", "reliable",
        }
        assert required.issubset(set(df.columns)), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_fixture_scm_canonical_columns(self):
        """Fixture must also expose the SCM framework canonical column names."""
        df = pd.read_csv(FIXTURE_PATH)
        scm_cols = {"friction_slope", "friction_slope_err", "velo_inerte_flag"}
        assert scm_cols.issubset(set(df.columns)), (
            f"Missing SCM canonical columns: {scm_cols - set(df.columns)}"
        )

    def test_fixture_scm_aliases_match_originals(self):
        """friction_slope must equal beta, etc. (they are aliases)."""
        df = pd.read_csv(FIXTURE_PATH)
        pd.testing.assert_series_equal(
            df["friction_slope"], df["beta"], check_names=False
        )
        pd.testing.assert_series_equal(
            df["friction_slope_err"], df["beta_err"], check_names=False
        )
        pd.testing.assert_series_equal(
            df["velo_inerte_flag"], df["reliable"], check_names=False
        )

    def test_fixture_not_empty(self):
        df = pd.read_csv(FIXTURE_PATH)
        assert len(df) > 0, "Fixture catalog is empty"

    def test_fixture_has_reliable_galaxies(self):
        """All fixture galaxies should have reliable β fits."""
        df = pd.read_csv(FIXTURE_PATH)
        assert df["reliable"].all(), (
            "Some fixture galaxies are not marked reliable — "
            "regenerate the fixture with smaller velocities."
        )

    def test_fixture_beta_near_one(self):
        """Reliable β values must be ≈ 1.0 ± 0.10 (flat rotation curve signature).

        With flat rotation curves, g_obs and g_bar both scale as V²/r with
        constant V, so the deep-regime slope β = 1 by construction.
        This is the defining property that distinguishes the synthetic CI
        fixture from a physical SPARC measurement (β ≈ 0.5).
        """
        df = pd.read_csv(FIXTURE_PATH)
        reliable = df[df["reliable"]]["beta"].dropna()
        assert len(reliable) > 0, "No reliable β values in fixture"
        assert reliable.mean() == pytest.approx(1.0, abs=0.10), (
            f"Fixture mean β = {reliable.mean():.3f}; expected ≈ 1.0 "
            "(synthetic flat-curve signature)"
        )

    def test_fixture_beta_not_mond(self):
        """Fixture β must be clearly > 0.5 (MOND deep-regime value).

        This is the semantic separation: the fixture is NOT a physical
        measurement; it would be incorrect to interpret its β as MOND-
        consistent.
        """
        df = pd.read_csv(FIXTURE_PATH)
        reliable = df[df["reliable"]]["beta"].dropna()
        assert reliable.mean() > 0.7, (
            f"Fixture mean β = {reliable.mean():.3f}; must be > 0.7 to be "
            "distinct from the physical MOND expectation (≈ 0.5)"
        )

    def test_fixture_n_deep_positive(self):
        """Every reliable galaxy must have at least min_deep_points deep pts."""
        df = pd.read_csv(FIXTURE_PATH)
        reliable_rows = df[df["reliable"]]
        assert (reliable_rows["n_deep"] >= MIN_DEEP_POINTS_DEFAULT).all(), (
            "Some reliable rows have fewer deep points than min_deep_points"
        )


# ---------------------------------------------------------------------------
# 2. measure_galaxy_beta unit tests
# ---------------------------------------------------------------------------

class TestMeasureGalaxyBeta:
    def _make_rc(self, vf: float, n_pts: int = 30) -> tuple[pd.DataFrame, float]:
        """Create a synthetic rotation curve with flat profile."""
        rng = np.random.default_rng(7)
        r = np.linspace(0.2, 15.0, n_pts)
        rc = pd.DataFrame({
            "r": r,
            "v_obs": np.full(n_pts, vf) + rng.normal(0, 0.01, n_pts),
            "v_obs_err": np.full(n_pts, 0.05),
            "v_gas": 0.3 * vf * np.ones(n_pts),
            "v_disk": 0.7 * vf * np.ones(n_pts),
            "v_bul": np.zeros(n_pts),
        })
        return rc, vf

    def test_returns_required_keys(self):
        rc, vf = self._make_rc(2.0)
        result = measure_galaxy_beta(rc, upsilon_disk=1.0)
        required = {
            "beta", "beta_err", "intercept", "r_value", "p_value",
            "n_deep", "n_total", "reliable",
        }
        assert required.issubset(set(result.keys()))

    def test_flat_curve_beta_near_one(self):
        """Flat-rotation-curve data must produce β ≈ 1."""
        rc, vf = self._make_rc(2.0)
        result = measure_galaxy_beta(rc, upsilon_disk=1.86)
        assert not math.isnan(result["beta"]), "β is NaN — no deep-regime points"
        assert result["beta"] == pytest.approx(1.0, abs=0.15)

    def test_nan_when_no_deep_points(self):
        """High-velocity galaxy has no deep-regime points → β = NaN."""
        rng = np.random.default_rng(0)
        vf = 200.0  # km/s — no deep-regime points
        n = 20
        rc = pd.DataFrame({
            "r": np.linspace(0.5, 15.0, n),
            "v_obs": np.full(n, vf) + rng.normal(0, 3, n),
            "v_obs_err": np.full(n, 5.0),
            "v_gas": 0.3 * vf * np.ones(n),
            "v_disk": 0.7 * vf * np.ones(n),
            "v_bul": np.zeros(n),
        })
        result = measure_galaxy_beta(rc, upsilon_disk=1.0)
        assert math.isnan(result["beta"])
        assert result["n_deep"] == 0
        assert result["reliable"] is False

    def test_reliable_flag_requires_min_deep_points(self):
        """reliable=False when n_deep < min_deep_points."""
        rc, vf = self._make_rc(2.0)
        # Use tiny deep_threshold so very few points qualify
        result = measure_galaxy_beta(
            rc, upsilon_disk=1.0,
            min_deep_points=100,  # impossible to satisfy
        )
        assert result["reliable"] is False

    def test_n_total_equals_valid_points(self):
        rc, vf = self._make_rc(2.0)
        result = measure_galaxy_beta(rc, upsilon_disk=1.0)
        # All v_obs and v_gas>0, v_disk>0 so n_total should equal n_pts
        assert result["n_total"] == len(rc)


# ---------------------------------------------------------------------------
# 3. generate_f3_catalog integration tests
# ---------------------------------------------------------------------------

class TestGenerateF3Catalog:
    def test_output_columns(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=3)
        out = tmp_path / "catalog.csv"
        df = generate_f3_catalog(data_dir, out, verbose=False)
        required = {
            "galaxy", "beta", "beta_err", "intercept",
            "r_value", "p_value", "n_deep", "n_total", "reliable",
        }
        assert required.issubset(set(df.columns))

    def test_output_scm_canonical_columns(self, tmp_path):
        """Pipeline output must include SCM framework canonical column names."""
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=3)
        out = tmp_path / "catalog.csv"
        df = generate_f3_catalog(data_dir, out, verbose=False)
        scm_cols = {"friction_slope", "friction_slope_err", "velo_inerte_flag"}
        assert scm_cols.issubset(set(df.columns)), (
            f"Missing SCM canonical columns: {scm_cols - set(df.columns)}"
        )

    def test_output_file_created(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=3)
        out = tmp_path / "catalog.csv"
        generate_f3_catalog(data_dir, out, verbose=False)
        assert out.exists()

    def test_output_sorted_by_galaxy(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=5)
        out = tmp_path / "catalog.csv"
        df = generate_f3_catalog(data_dir, out, verbose=False)
        assert list(df["galaxy"]) == sorted(df["galaxy"].tolist())

    def test_flat_curves_beta_approx_one(self, tmp_path):
        """Full pipeline on flat-curve synthetic data → β ≈ 1."""
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=5)
        out = tmp_path / "catalog.csv"
        df = generate_f3_catalog(data_dir, out, verbose=False)
        reliable = df[df["reliable"]]["beta"].dropna()
        assert len(reliable) > 0, "No reliable galaxies produced"
        assert reliable.mean() == pytest.approx(1.0, abs=0.15)

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_f3_catalog(tmp_path / "nonexistent", tmp_path / "out.csv",
                                verbose=False)

    def test_cli_main_produces_file(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=3)
        out = tmp_path / "catalog.csv"
        df = generate_main(["--data-dir", str(data_dir), "--out", str(out),
                            "--quiet"])
        assert out.exists()
        assert len(df) == 3


# ---------------------------------------------------------------------------
# 4. analyze_f3_catalog and f3_catalog_analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeF3Catalog:
    def _make_catalog(self, n: int = 10, beta: float = 1.0) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(n)],
            "beta": beta + rng.normal(0, 0.02, n),
            "beta_err": np.full(n, 0.01),
            "intercept": np.zeros(n),
            "r_value": np.full(n, 0.99),
            "p_value": np.full(n, 1e-20),
            "n_deep": np.full(n, 15),
            "n_total": np.full(n, 30),
            "reliable": np.ones(n, dtype=bool),
        })

    def test_returns_required_keys(self):
        df = self._make_catalog()
        stats = analyze_f3_catalog(df)
        required = {
            "n_galaxies", "n_reliable", "beta_mean", "beta_median",
            "beta_std", "beta_mean_all", "delta_from_mond", "consistent_mond",
        }
        assert required.issubset(set(stats.keys()))

    def test_beta_near_one_not_mond_consistent(self):
        """β ≈ 1 (synthetic) is NOT MOND-consistent (which expects β ≈ 0.5)."""
        df = self._make_catalog(beta=1.0)
        stats = analyze_f3_catalog(df)
        assert not stats["consistent_mond"]

    def test_beta_near_half_mond_consistent(self):
        """β ≈ 0.5 (physical SPARC) IS MOND-consistent."""
        df = self._make_catalog(beta=0.5)
        stats = analyze_f3_catalog(df)
        assert stats["consistent_mond"]

    def test_n_reliable_counts_only_reliable(self):
        df = self._make_catalog(n=10)
        df.loc[0:4, "reliable"] = False
        stats = analyze_f3_catalog(df)
        assert stats["n_reliable"] == 5

    def test_missing_columns_raises(self):
        bad = pd.DataFrame({"galaxy": ["A"], "beta": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            analyze_f3_catalog(bad)

    def test_format_report_contains_verdict(self):
        df = self._make_catalog()
        stats = analyze_f3_catalog(df)
        lines = format_analysis_report(stats, "test.csv")
        combined = "\n".join(lines)
        assert "Verdict" in combined
        assert "β" in combined

    def test_analysis_cli_missing_catalog_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analysis_main(["--catalog", str(tmp_path / "missing.csv")])

    def test_analysis_cli_writes_output_files(self, tmp_path):
        df = self._make_catalog()
        catalog = tmp_path / "catalog.csv"
        df.to_csv(catalog, index=False)
        out_dir = tmp_path / "out"
        analysis_main(["--catalog", str(catalog), "--out", str(out_dir)])
        assert (out_dir / "f3_analysis.csv").exists()
        assert (out_dir / "f3_analysis.log").exists()

    def test_analysis_cli_returns_stats_dict(self, tmp_path):
        df = self._make_catalog()
        catalog = tmp_path / "catalog.csv"
        df.to_csv(catalog, index=False)
        stats = analysis_main(["--catalog", str(catalog)])
        assert "beta_median" in stats
        assert stats["n_galaxies"] == 10


# ---------------------------------------------------------------------------
# 5. Semantic separation test
# ---------------------------------------------------------------------------

class TestSemanticSeparation:
    """Verify the three-level separation defined in the SCM framework:
       1. CI fixture  (synthetic flat)  → β ≈ 1
       2. Physical pipeline output      → β ≈ 0.5 (MOND)
       3. Statistical analysis          → reads the physical output
    """

    def test_fixture_labeled_synthetic_flat(self):
        """The fixture file name must contain 'synthetic_flat'."""
        assert "synthetic_flat" in FIXTURE_PATH.name, (
            "CI fixture must be named '*synthetic_flat*' to distinguish it "
            "from the physical measurement output."
        )

    def test_fixture_beta_differs_from_mond(self):
        """Fixture β must be clearly separated from the MOND value (0.5)."""
        df = pd.read_csv(FIXTURE_PATH)
        reliable = df[df["reliable"]]["beta"].dropna()
        # |mean_beta - 0.5| must be > 0.3 (i.e. fixture β is not MOND-like)
        assert abs(reliable.mean() - 0.5) > 0.3, (
            f"Fixture β = {reliable.mean():.3f} is too close to MOND value "
            f"0.5 — the semantic separation is violated."
        )

    def test_generate_script_default_output_is_real(self):
        """generate_f3_catalog.py must default to f3_catalog_real.csv."""
        import argparse
        from scripts.generate_f3_catalog import _parse_args
        args = _parse_args(["--data-dir", "/tmp/data"])
        assert "f3_catalog_real.csv" in args.out, (
            "Physical pipeline default output must be f3_catalog_real.csv"
        )

    def test_analysis_script_default_input_is_real(self):
        """f3_catalog_analysis.py must default to f3_catalog_real.csv."""
        from scripts.f3_catalog_analysis import _parse_args
        args = _parse_args([])
        assert "f3_catalog_real.csv" in args.catalog, (
            "Analysis script default input must be f3_catalog_real.csv"
        )
