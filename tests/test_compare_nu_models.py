"""
tests/test_compare_nu_models.py — Tests for scripts/compare_nu_models.py.

Uses synthetic SPARC-like data (no real download required).
"""

import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.compare_nu_models import (
    nu_simple,
    nu_standard,
    nu_exp_linear,
    v_pred_nu,
    v_pred_velos,
    log_likelihood,
    aicc,
    run_data_dir_comparison,
    run_csv_comparison,
    main,
    A0_DEFAULT,
    NU_MODELS,
)


# ---------------------------------------------------------------------------
# ν function unit tests
# ---------------------------------------------------------------------------

class TestNuFunctions:
    """Each ν model must satisfy ν(x) → 1/√x as x → 0 (deep MOND limit)."""

    @pytest.mark.parametrize("nu_fn", list(NU_MODELS.values()))
    def test_deep_mond_asymptote(self, nu_fn):
        """ν(x) ≈ 1/√x  for x << 1."""
        x = np.array([1e-4, 1e-5, 1e-6])
        ratio = nu_fn(x) * np.sqrt(x)  # should → 1
        np.testing.assert_allclose(ratio, np.ones(3), rtol=0.05)

    @pytest.mark.parametrize("nu_fn", list(NU_MODELS.values()))
    def test_newtonian_limit(self, nu_fn):
        """ν(x) → 1 as x → ∞ (Newtonian regime)."""
        x = np.array([100.0, 1000.0])
        nu = nu_fn(x)
        np.testing.assert_allclose(nu, np.ones(2), rtol=0.01)

    @pytest.mark.parametrize("nu_fn", list(NU_MODELS.values()))
    def test_positive_and_finite(self, nu_fn):
        """ν(x) must be positive and finite for x ∈ [0.001, 1000]."""
        x = np.logspace(-3, 3, 100)
        nu = nu_fn(x)
        assert np.all(np.isfinite(nu)), "ν has non-finite values"
        assert np.all(nu > 0), "ν has non-positive values"

    @pytest.mark.parametrize("nu_fn", list(NU_MODELS.values()))
    def test_monotone_decreasing(self, nu_fn):
        """ν(x) should be non-increasing (monotonically decreasing or flat at Newtonian plateau)."""
        x = np.logspace(-3, 3, 200)
        nu = nu_fn(x)
        assert np.all(np.diff(nu) <= 0), "ν is not non-increasing"


# ---------------------------------------------------------------------------
# Log-likelihood and AICc
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def test_perfect_prediction(self):
        """Perfect prediction: residuals = 0, LL = -½ Σ ln(2π σ²)."""
        v = np.array([100.0, 150.0, 200.0])
        err = np.array([5.0, 5.0, 5.0])
        ll = log_likelihood(v, err, v)
        expected = -0.5 * np.sum(np.log(2 * np.pi * err ** 2))
        assert ll == pytest.approx(expected, rel=1e-6)

    def test_worse_prediction_lower_ll(self):
        v = np.array([100.0, 150.0])
        err = np.array([10.0, 10.0])
        ll_good = log_likelihood(v, err, v)
        ll_bad = log_likelihood(v, err, v + 20.0)
        assert ll_good > ll_bad

    def test_zero_error_handled(self):
        v = np.array([100.0])
        err = np.array([0.0])  # should use σ=1 fallback
        ll = log_likelihood(v, err, v)
        assert np.isfinite(ll)


class TestAICc:
    def test_increases_with_k(self):
        ll = -100.0
        n = 50
        aicc1 = aicc(ll, 1, n)
        aicc2 = aicc(ll, 5, n)
        assert aicc2 > aicc1

    def test_correction_shrinks_for_large_n(self):
        """For large n, AICc → AIC."""
        ll = -500.0
        k = 5
        aic_c_small = aicc(ll, k, 20)
        aic_c_large = aicc(ll, k, 10000)
        # With large n, correction → 0, AICc → AIC = -2*LL + 2k = 1010
        assert aic_c_large == pytest.approx(-2 * ll + 2 * k, rel=0.01)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sparc_dir(tmp_path_factory):
    """Synthetic 20-galaxy SPARC-like dataset."""
    root = tmp_path_factory.mktemp("sparc20")
    rng = np.random.default_rng(7)
    n_gal = 20
    names = [f"T{i:03d}" for i in range(n_gal)]
    v_flats = np.linspace(80.0, 300.0, n_gal)

    pd.DataFrame({
        "Galaxy": names,
        "D": np.linspace(5, 60, n_gal),
        "Inc": np.linspace(30, 80, n_gal),
        "L36": 1e9 * np.arange(1, n_gal + 1, dtype=float),
        "Vflat": v_flats,
        "e_Vflat": np.full(n_gal, 5.0),
    }).to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    n_pts = 15
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

    return root


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory, sparc_dir):
    """Pre-computed per-galaxy CSV (from the sparc_dir synthetic dataset)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.scm_analysis import run_pipeline

    out = tmp_path_factory.mktemp("csv_out")
    run_pipeline(sparc_dir, out, verbose=False)
    return out / "per_galaxy_summary.csv"


# ---------------------------------------------------------------------------
# run_data_dir_comparison
# ---------------------------------------------------------------------------

class TestRunDataDirComparison:
    def test_returns_dataframe(self, sparc_dir, tmp_path):
        df, winner, _ = run_data_dir_comparison(sparc_dir, tmp_path / "out")
        assert isinstance(df, pd.DataFrame)

    def test_all_models_present(self, sparc_dir, tmp_path):
        df, _, _ = run_data_dir_comparison(sparc_dir, tmp_path / "out")
        expected = {"velos", "simple", "standard", "exp_linear"}
        assert set(df["model"]) == expected

    def test_delta_aicc_winner_is_zero(self, sparc_dir, tmp_path):
        df, winner, _ = run_data_dir_comparison(sparc_dir, tmp_path / "out")
        assert df.loc[df["model"] == winner, "delta_AICc"].iloc[0] == pytest.approx(0.0)

    def test_ll_finite(self, sparc_dir, tmp_path):
        df, _, _ = run_data_dir_comparison(sparc_dir, tmp_path / "out")
        assert df["LL"].apply(np.isfinite).all()

    def test_output_csv_written(self, sparc_dir, tmp_path):
        out = tmp_path / "out2"
        run_data_dir_comparison(sparc_dir, out)
        assert (out / "compare_nu_models.csv").exists()

    def test_n_galaxies_matches(self, sparc_dir, tmp_path):
        df, _, _ = run_data_dir_comparison(sparc_dir, tmp_path / "out3")
        assert int(df["N_galaxies"].iloc[0]) == 20


# ---------------------------------------------------------------------------
# run_csv_comparison
# ---------------------------------------------------------------------------

class TestRunCsvComparison:
    def test_returns_dataframe(self, csv_path, tmp_path):
        df, winner = run_csv_comparison(csv_path, tmp_path / "out")
        assert isinstance(df, pd.DataFrame)

    def test_velos_has_ll(self, csv_path, tmp_path):
        df, _ = run_csv_comparison(csv_path, tmp_path / "out")
        velos_ll = df.loc[df["model"] == "velos", "LL"].iloc[0]
        assert np.isfinite(velos_ll)

    def test_nu_models_na(self, csv_path, tmp_path):
        df, _ = run_csv_comparison(csv_path, tmp_path / "out")
        for mname in NU_MODELS:
            row_ll = df.loc[df["model"] == mname, "LL"].iloc[0]
            assert np.isnan(row_ll), f"{mname} LL should be NaN in csv mode"

    def test_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "upsilon_disk": [1.0]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            run_csv_comparison(bad_csv, tmp_path / "out")


# ---------------------------------------------------------------------------
# CLI (main) integration tests
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_data_dir_creates_log(self, sparc_dir, tmp_path):
        out = tmp_path / "cli_out"
        main(["--data-dir", str(sparc_dir), "--out", str(out)])
        assert (out / "compare_nu_models.log").exists()
        assert (out / "compare_nu_models.csv").exists()

    def test_csv_mode_creates_log(self, csv_path, tmp_path):
        out = tmp_path / "cli_csv"
        main(["--csv", str(csv_path), "--out", str(out)])
        assert (out / "compare_nu_models.log").exists()
        assert (out / "compare_nu_models.csv").exists()

    def test_no_source_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--out", str(tmp_path)])
