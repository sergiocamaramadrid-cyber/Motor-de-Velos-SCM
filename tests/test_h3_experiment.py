"""
tests/test_h3_experiment.py — Tests for the H3 radial diffusion experiment.

Covers:
  1. bootstrap_beta — correctness, NaN handling, reproducibility.
  2. run_h3_experiment — end-to-end with synthetic radial profiles and
     environmental delta data.
  3. generate_sparc_radial_profiles — _compute_delta_f3 helper.
  4. generate_sparc_delta_env — _from_radial_csv aggregation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_h3_experiment import bootstrap_beta, run_h3_experiment

# ---------------------------------------------------------------------------
# Helpers – synthetic data builders
# ---------------------------------------------------------------------------

def _make_radial_profiles(tmp_path: Path, n_galaxies: int = 20, n_pts: int = 15) -> Path:
    """Write a synthetic sparc_full_radial.csv with a clear autocorrelation signal."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_galaxies):
        gal = f"galaxy_{i:03d}"
        # AR(1) signal: phi ~ U(0.3, 0.9) so rho_lag1 is positive and large
        phi = rng.uniform(0.3, 0.9)
        z = 0.0
        for _ in range(n_pts):
            z = phi * z + rng.normal(0, 0.1)
            rows.append({'galaxy': gal, 'r': rng.uniform(0.5, 15.0), 'delta_F3': z})
    df = pd.DataFrame(rows)
    out = tmp_path / 'sparc_full_radial.csv'
    df.to_csv(out, index=False)
    return out


def _make_env_data(tmp_path: Path, n_galaxies: int = 20) -> Path:
    """Write a synthetic SPARC_with_delta_real.csv."""
    rng = np.random.default_rng(1)
    records = [
        {'galaxy': f'galaxy_{i:03d}', 'delta_mass': rng.normal(0, 1), 'delta_mode': 'mass'}
        for i in range(n_galaxies)
    ]
    df = pd.DataFrame(records)
    out = tmp_path / 'SPARC_with_delta_real.csv'
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# bootstrap_beta tests
# ---------------------------------------------------------------------------

class TestBootstrapBeta:
    def test_returns_negative_beta_for_negative_slope(self, tmp_path):
        """β should be close to -1 for perfectly anti-correlated data."""
        rng = np.random.default_rng(99)
        n = 50
        x = rng.normal(0, 1, n)
        y = -1.5 * x + rng.normal(0, 0.1, n)
        df = pd.DataFrame({'x': x, 'y': y})
        result = bootstrap_beta(df, 'x', 'y', n_boot=200, seed=7)
        assert result['mean'] < 0
        assert result['ci_high'] < 0   # CI entirely below zero

    def test_returns_nan_for_tiny_dataset(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = bootstrap_beta(df, 'x', 'y', n_boot=50, seed=0)
        assert result['valid_iters'] == 0
        assert math.isnan(result['mean'])

    def test_reproducible_with_same_seed(self, tmp_path):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({'x': rng.normal(size=30), 'y': rng.normal(size=30)})
        r1 = bootstrap_beta(df, 'x', 'y', n_boot=100, seed=77)
        r2 = bootstrap_beta(df, 'x', 'y', n_boot=100, seed=77)
        assert r1['mean'] == r2['mean']
        assert r1['ci_low'] == r2['ci_low']
        assert r1['ci_high'] == r2['ci_high']

    def test_nan_values_in_columns_are_dropped(self):
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'y': [2.0, 4.0, np.nan, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        })
        result = bootstrap_beta(df, 'x', 'y', n_boot=100, seed=0)
        # Should not raise and should return a valid beta
        assert result['valid_iters'] > 0
        assert math.isfinite(result['mean'])

    def test_positive_beta_for_positive_slope(self):
        rng = np.random.default_rng(5)
        n = 40
        x = rng.normal(0, 1, n)
        y = 2.0 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({'x': x, 'y': y})
        result = bootstrap_beta(df, 'x', 'y', n_boot=200, seed=0)
        assert result['mean'] > 0
        assert result['ci_low'] > 0


# ---------------------------------------------------------------------------
# run_h3_experiment end-to-end tests
# ---------------------------------------------------------------------------

class TestRunH3Experiment:
    def test_produces_final_report_csv(self, tmp_path):
        f3_path = _make_radial_profiles(tmp_path, n_galaxies=25, n_pts=15)
        env_path = _make_env_data(tmp_path, n_galaxies=25)
        out_dir = tmp_path / 'H3_out'

        summary = run_h3_experiment(
            f3_path=f3_path,
            delta_path=env_path,
            base_out_dir=out_dir,
            thresholds=(8, 10),
            n_boot=50,
            seed=0,
            min_galaxies=5,
        )

        assert (out_dir / 'final_report.csv').exists()
        assert (out_dir / 'all_rejected.csv').exists()
        assert isinstance(summary, pd.DataFrame)

    def test_creates_per_threshold_subdirs(self, tmp_path):
        f3_path = _make_radial_profiles(tmp_path, n_galaxies=25, n_pts=15)
        env_path = _make_env_data(tmp_path, n_galaxies=25)
        out_dir = tmp_path / 'H3_out2'

        run_h3_experiment(
            f3_path=f3_path,
            delta_path=env_path,
            base_out_dir=out_dir,
            thresholds=(8,),
            n_boot=30,
            seed=0,
            min_galaxies=5,
        )

        sub = out_dir / 'min_points_8'
        assert sub.is_dir()
        assert (sub / 'data_merged.csv').exists()
        assert (sub / 'rejected_galaxies.csv').exists()

    def test_summary_report_columns(self, tmp_path):
        f3_path = _make_radial_profiles(tmp_path, n_galaxies=25, n_pts=15)
        env_path = _make_env_data(tmp_path, n_galaxies=25)
        out_dir = tmp_path / 'H3_cols'

        summary = run_h3_experiment(
            f3_path=f3_path,
            delta_path=env_path,
            base_out_dir=out_dir,
            thresholds=(10,),
            n_boot=30,
            seed=0,
            min_galaxies=5,
        )

        expected_cols = {
            'threshold', 'metric', 'spearman_rho', 'spearman_p',
            'boot_mean', 'boot_ci_low', 'boot_ci_high',
            'boot_valid_iters', 'n_galaxies',
        }
        assert expected_cols.issubset(set(summary.columns))

    def test_empty_result_when_all_below_threshold(self, tmp_path):
        """If all galaxies have fewer points than the threshold, report is empty."""
        f3_path = _make_radial_profiles(tmp_path, n_galaxies=20, n_pts=5)
        env_path = _make_env_data(tmp_path, n_galaxies=20)
        out_dir = tmp_path / 'H3_empty'

        summary = run_h3_experiment(
            f3_path=f3_path,
            delta_path=env_path,
            base_out_dir=out_dir,
            thresholds=(20,),   # requires 20 pts but only 5 generated
            n_boot=30,
            seed=0,
            min_galaxies=5,
        )

        assert summary.empty

    def test_missing_delta_F3_column_is_rejected(self, tmp_path):
        """Rows with galaxy but no delta_F3 column are counted as rejected."""
        bad_f3 = pd.DataFrame({'galaxy': ['g1', 'g2'], 'r': [1.0, 2.0]})
        f3_path = tmp_path / 'bad.csv'
        bad_f3.to_csv(f3_path, index=False)

        env_path = _make_env_data(tmp_path, n_galaxies=5)
        out_dir = tmp_path / 'H3_bad'

        summary = run_h3_experiment(
            f3_path=f3_path,
            delta_path=env_path,
            base_out_dir=out_dir,
            thresholds=(5,),
            n_boot=20,
            seed=0,
            min_galaxies=1,
        )

        all_rej = pd.read_csv(out_dir / 'all_rejected.csv')
        assert len(all_rej) == 2
        assert all(all_rej['reason'] == 'no_delta_F3')

    def test_reproducible_with_same_seed(self, tmp_path):
        f3_path = _make_radial_profiles(tmp_path, n_galaxies=20, n_pts=12)
        env_path = _make_env_data(tmp_path, n_galaxies=20)

        s1 = run_h3_experiment(
            f3_path=f3_path, delta_path=env_path,
            base_out_dir=tmp_path / 'run1',
            thresholds=(8,), n_boot=50, seed=42, min_galaxies=5,
        )
        s2 = run_h3_experiment(
            f3_path=f3_path, delta_path=env_path,
            base_out_dir=tmp_path / 'run2',
            thresholds=(8,), n_boot=50, seed=42, min_galaxies=5,
        )

        if not s1.empty and not s2.empty:
            pd.testing.assert_frame_equal(
                s1.reset_index(drop=True),
                s2.reset_index(drop=True),
            )

    def test_delta_mode_filter(self, tmp_path):
        """Only rows with delta_mode == 'mass' should be used."""
        f3_path = _make_radial_profiles(tmp_path, n_galaxies=20, n_pts=12)

        # Create env_data with mixed modes
        records = []
        for i in range(20):
            records.append({
                'galaxy': f'galaxy_{i:03d}',
                'delta_mass': float(i) * 0.1,
                'delta_mode': 'mass' if i % 2 == 0 else 'velocity',
            })
        env_path = tmp_path / 'mixed_env.csv'
        pd.DataFrame(records).to_csv(env_path, index=False)

        out_dir = tmp_path / 'H3_mode'
        run_h3_experiment(
            f3_path=f3_path,
            delta_path=env_path,
            base_out_dir=out_dir,
            thresholds=(8,),
            n_boot=30,
            seed=0,
            min_galaxies=1,
        )

        # After merge, only 'mass' mode galaxies (even indices) should be present
        merged = pd.read_csv(out_dir / 'min_points_8' / 'data_merged.csv')
        # Even-indexed galaxies are in 'mass' mode
        even_galaxies = {f'galaxy_{i:03d}' for i in range(0, 20, 2)}
        assert set(merged['galaxy']).issubset(even_galaxies)


# ---------------------------------------------------------------------------
# generate_sparc_radial_profiles helper test
# ---------------------------------------------------------------------------

class TestComputeDeltaF3:
    def test_returns_expected_columns(self):
        """_compute_delta_f3 must return columns ['r', 'delta_F3']."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.generate_sparc_radial_profiles import _compute_delta_f3

        rc = pd.DataFrame({
            'r':       [1.0, 2.0, 5.0, 10.0],
            'v_obs':   [100.0, 110.0, 120.0, 130.0],
            'v_gas':   [20.0,  25.0,  30.0,  35.0],
            'v_disk':  [80.0,  85.0,  90.0,  95.0],
            'v_bul':   [0.0,   0.0,   0.0,   0.0],
        })
        result = _compute_delta_f3(rc, upsilon_disk=1.0)
        assert set(result.columns) == {'r', 'delta_F3'}
        assert len(result) == 4
        assert result['delta_F3'].notna().all()

    def test_skips_invalid_points(self):
        """Points with v_obs = 0 should be excluded (g_obs = 0 → not valid)."""
        from scripts.generate_sparc_radial_profiles import _compute_delta_f3

        rc = pd.DataFrame({
            'r':       [1.0, 2.0],
            'v_obs':   [0.0, 100.0],   # first point invalid
            'v_gas':   [10.0, 10.0],
            'v_disk':  [50.0, 50.0],
            'v_bul':   [0.0,  0.0],
        })
        result = _compute_delta_f3(rc, upsilon_disk=1.0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# generate_sparc_delta_env helper test
# ---------------------------------------------------------------------------

class TestFromRadialCsv:
    def test_aggregates_to_median(self, tmp_path):
        """delta_mass must be the per-galaxy median of delta_F3."""
        from scripts.generate_sparc_delta_env import _from_radial_csv

        df = pd.DataFrame({
            'galaxy':   ['gA', 'gA', 'gA', 'gB', 'gB'],
            'r':        [1, 2, 3, 1, 2],
            'delta_F3': [0.1, 0.3, 0.5, -0.2, -0.4],
        })
        radial_csv = tmp_path / 'profiles.csv'
        df.to_csv(radial_csv, index=False)

        result = _from_radial_csv(radial_csv, verbose=False)
        assert set(result.columns) == {'galaxy', 'delta_mass', 'delta_mode'}
        assert (result['delta_mode'] == 'mass').all()

        gA_row = result[result['galaxy'] == 'gA']
        assert pytest.approx(float(gA_row['delta_mass'].iloc[0]), abs=1e-9) == 0.3

        gB_row = result[result['galaxy'] == 'gB']
        assert pytest.approx(float(gB_row['delta_mass'].iloc[0]), abs=1e-9) == -0.3

    def test_raises_without_delta_f3_column(self, tmp_path):
        from scripts.generate_sparc_delta_env import _from_radial_csv

        df = pd.DataFrame({'galaxy': ['g1'], 'r': [1.0]})
        csv_path = tmp_path / 'bad.csv'
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="delta_F3"):
            _from_radial_csv(csv_path, verbose=False)
