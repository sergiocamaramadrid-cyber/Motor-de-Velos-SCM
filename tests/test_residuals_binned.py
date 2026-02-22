"""Tests for scripts/compute_residuals_binned.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the scripts package importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.compute_residuals_binned import (
    A0_SI,
    bin_residuals,
    compute_log_residuals,
    g_pred_rar,
    load_dataset,
    nu_rar,
    write_csv,
)


# ---------------------------------------------------------------------------
# nu_rar
# ---------------------------------------------------------------------------

class TestNuRar:
    def test_deep_mond_limit(self):
        """In deep-MOND (x → 0), ν ≈ 1/√x.

        For small x: 1 − exp(−√x) ≈ √x, so ν(x) = 1/(1−exp(−√x)) ≈ 1/√x.
        """
        x = np.array([1e-6])
        nu = nu_rar(x)
        expected = 1.0 / np.sqrt(x)
        np.testing.assert_allclose(nu, expected, rtol=1e-3)

    def test_newtonian_limit(self):
        """In Newtonian regime (x → ∞), ν → 1."""
        x = np.array([1e6])
        nu = nu_rar(x)
        np.testing.assert_allclose(nu, 1.0, rtol=1e-4)

    def test_always_geq_one(self):
        x = np.logspace(-4, 4, 50)
        assert np.all(nu_rar(x) >= 1.0)


# ---------------------------------------------------------------------------
# g_pred_rar
# ---------------------------------------------------------------------------

class TestGPredRar:
    def test_g_pred_geq_g_bar(self):
        """Predicted acceleration must be ≥ g_bar (ν ≥ 1)."""
        g_bar = np.logspace(-14, -9, 100)
        g_pred = g_pred_rar(g_bar, a0=A0_SI)
        assert np.all(g_pred >= g_bar)

    def test_newtonian_limit(self):
        """When g_bar >> a0, g_pred ≈ g_bar."""
        g_bar = np.array([1e-5])  # >> a0 = 1.2e-10
        g_pred = g_pred_rar(g_bar, a0=A0_SI)
        np.testing.assert_allclose(g_pred, g_bar, rtol=1e-4)


# ---------------------------------------------------------------------------
# compute_log_residuals
# ---------------------------------------------------------------------------

class TestComputeLogResiduals:
    def test_zero_residual_when_obs_equals_pred(self):
        g_bar = np.logspace(-13, -10, 20)
        g_obs = g_pred_rar(g_bar)
        delta = compute_log_residuals(g_bar, g_obs)
        np.testing.assert_allclose(delta, 0.0, atol=1e-10)

    def test_shape_preserved(self):
        g_bar = np.logspace(-13, -10, 50)
        g_obs = g_bar * 1.5
        delta = compute_log_residuals(g_bar, g_obs)
        assert delta.shape == g_bar.shape


# ---------------------------------------------------------------------------
# bin_residuals
# ---------------------------------------------------------------------------

class TestBinResiduals:
    def _make_data(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        g_bar = np.logspace(-14, -9, n)
        delta = rng.normal(0.0, 0.05, n)
        return g_bar, delta

    def test_returns_dataframe(self):
        g_bar, delta = self._make_data()
        result = bin_residuals(g_bar, delta, n_bins=10, min_count=3)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        g_bar, delta = self._make_data()
        result = bin_residuals(g_bar, delta, n_bins=10)
        assert set(result.columns) == {
            "g_bar_center", "median_residual", "mad_residual", "count"
        }

    def test_counts_positive(self):
        g_bar, delta = self._make_data()
        result = bin_residuals(g_bar, delta, n_bins=10)
        assert (result["count"] > 0).all()

    def test_mad_nonnegative(self):
        g_bar, delta = self._make_data()
        result = bin_residuals(g_bar, delta, n_bins=10)
        assert (result["mad_residual"] >= 0).all()

    def test_min_count_filter(self):
        """Bins with fewer than min_count points must be dropped."""
        g_bar, delta = self._make_data(n=20)
        # With min_count=100, no bin should survive
        result = bin_residuals(g_bar, delta, n_bins=10, min_count=100)
        assert len(result) == 0

    def test_g_bar_center_monotone_increasing(self):
        g_bar, delta = self._make_data()
        result = bin_residuals(g_bar, delta, n_bins=10)
        centers = result["g_bar_center"].values
        assert np.all(np.diff(centers) > 0)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_loads_sparc_sample(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("g_bar,g_obs,g_err\n1e-12,2e-12,1e-13\n2e-12,3e-12,2e-13\n")
        df = load_dataset(csv)
        assert "g_bar" in df.columns
        assert "g_obs" in df.columns
        assert len(df) == 2

    def test_missing_column_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("g_obs,g_err\n1e-12,1e-13\n")
        with pytest.raises(ValueError, match="g_bar"):
            load_dataset(csv)

    def test_filters_nonpositive_rows(self, tmp_path):
        csv = tmp_path / "nonpos.csv"
        csv.write_text("g_bar,g_obs\n1e-12,2e-12\n-1e-12,2e-12\n1e-12,-2e-12\n")
        df = load_dataset(csv)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def test_metadata_in_header(self, tmp_path):
        out = tmp_path / "test_out.csv"
        df = pd.DataFrame(
            {"g_bar_center": [1e-12], "median_residual": [0.0],
             "mad_residual": [0.01], "count": [5]}
        )
        meta = {"bins_effective": 1, "g_bar_center_min": "1.0e-12",
                "g_bar_center_max": "1.0e-12"}
        write_csv(df, out, meta)
        content = out.read_text()
        assert "# bins_effective: 1" in content
        assert "# g_bar_center_min:" in content
        assert "# g_bar_center_max:" in content

    def test_csv_readable_back(self, tmp_path):
        out = tmp_path / "roundtrip.csv"
        df = pd.DataFrame(
            {"g_bar_center": [1e-12, 2e-12],
             "median_residual": [0.1, -0.2],
             "mad_residual": [0.05, 0.04],
             "count": [10, 12]}
        )
        write_csv(df, out, {})
        df2 = pd.read_csv(out, comment="#")
        assert list(df2.columns) == list(df.columns)
        assert len(df2) == 2


# ---------------------------------------------------------------------------
# Integration: residuals_binned_v02.csv exists and is well-formed
# ---------------------------------------------------------------------------

class TestResidualsBinnedCsvExists:
    """Validate the committed artefact residuals_binned_v02.csv."""

    _CSV = Path(__file__).parent.parent / "results" / "residuals_binned_v02.csv"

    def test_file_exists(self):
        assert self._CSV.exists(), f"Missing: {self._CSV}"

    def test_metadata_comments(self):
        lines = self._CSV.read_text().splitlines()
        comment_lines = [l for l in lines if l.startswith("#")]
        keys = [l.split(":")[0].lstrip("# ") for l in comment_lines]
        assert "bins_effective" in keys
        assert "g_bar_center_min" in keys
        assert "g_bar_center_max" in keys

    def test_loadable_with_pandas(self):
        df = pd.read_csv(self._CSV, comment="#")
        assert set(df.columns) == {
            "g_bar_center", "median_residual", "mad_residual", "count"
        }
        assert len(df) > 0

    def test_bins_positive_count(self):
        df = pd.read_csv(self._CSV, comment="#")
        assert (df["count"] > 0).all()

    def test_mad_nonnegative(self):
        df = pd.read_csv(self._CSV, comment="#")
        assert (df["mad_residual"] >= 0).all()

    def test_g_bar_center_monotone(self):
        df = pd.read_csv(self._CSV, comment="#")
        centers = df["g_bar_center"].values
        assert np.all(np.diff(centers) > 0), "g_bar_center must be monotonically increasing"
