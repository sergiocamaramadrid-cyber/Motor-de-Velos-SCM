"""
tests/test_generate_scm_residual_catalog.py — Tests for the residual catalog
generation pipeline (generate_scm_residual_catalog.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_scm_residual_catalog import (
    compute_galaxy_residual,
    generate_residual_catalog,
    OUTPUT_COLS,
    MIN_POINTS_DEFAULT,
    A0_DEFAULT,
    main as generate_main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_sparc_dir(
    tmp_path: Path,
    n_gal: int = 5,
    v_flats: list[float] | None = None,
    n_pts: int = 20,
    seed: int = 42,
) -> Path:
    """Create a minimal synthetic SPARC-like dataset.

    Velocities are chosen small (≤ 3.5 km/s) so that deep-regime points
    exist at large radii.  Flat rotation curves give g_obs ∝ g_bar, so
    f3_residual ≈ 0 and v_last equals the flat velocity.
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


def _make_rc(vf: float = 2.0, n_pts: int = 20, seed: int = 7) -> pd.DataFrame:
    """Create a single synthetic rotation curve DataFrame."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.2, 15.0, n_pts)
    return pd.DataFrame({
        "r": r,
        "v_obs": np.full(n_pts, vf) + rng.normal(0, 0.01, n_pts),
        "v_obs_err": np.full(n_pts, 0.05),
        "v_gas": 0.3 * vf * np.ones(n_pts),
        "v_disk": 0.7 * vf * np.ones(n_pts),
        "v_bul": np.zeros(n_pts),
    })


# ---------------------------------------------------------------------------
# compute_galaxy_residual unit tests
# ---------------------------------------------------------------------------

class TestComputeGalaxyResidual:
    def test_returns_dict_with_required_keys(self):
        rc = _make_rc()
        result = compute_galaxy_residual(rc, upsilon_disk=1.0)
        assert result is not None
        assert {"f3_residual", "v_last"}.issubset(result.keys())

    def test_f3_residual_is_finite(self):
        rc = _make_rc()
        result = compute_galaxy_residual(rc, upsilon_disk=1.0)
        assert result is not None
        assert np.isfinite(result["f3_residual"])

    def test_v_last_equals_last_radius_v_obs(self):
        """v_last must be the v_obs at the outermost radial point."""
        rc = _make_rc()
        result = compute_galaxy_residual(rc, upsilon_disk=1.0)
        assert result is not None
        expected = float(rc.loc[rc["r"].idxmax(), "v_obs"])
        assert result["v_last"] == pytest.approx(expected, rel=1e-9)

    def test_flat_curve_f3_residual_near_zero(self):
        """For a flat rotation curve with upsilon_disk ≈ 1, the log ratio
        g_obs/g_bar should be close to zero (both scale as V²/r)."""
        rc = _make_rc(vf=2.0)
        result = compute_galaxy_residual(rc, upsilon_disk=1.0)
        assert result is not None
        assert abs(result["f3_residual"]) < 1.0

    def test_returns_none_when_insufficient_points(self):
        """When fewer valid points than min_points, return None."""
        rc = _make_rc(n_pts=1)
        result = compute_galaxy_residual(rc, upsilon_disk=1.0, min_points=5)
        assert result is None

    def test_returns_none_for_zero_velocities(self):
        """All-zero v_obs produces g_obs = 0 → no valid points → None."""
        rc = _make_rc()
        rc["v_obs"] = 0.0
        result = compute_galaxy_residual(rc, upsilon_disk=1.0, min_points=1)
        assert result is None

    def test_v_last_is_positive(self):
        rc = _make_rc()
        result = compute_galaxy_residual(rc, upsilon_disk=1.0)
        assert result is not None
        assert result["v_last"] > 0.0


# ---------------------------------------------------------------------------
# generate_residual_catalog integration tests
# ---------------------------------------------------------------------------

class TestGenerateResidualCatalog:
    def test_output_columns(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=3)
        out = tmp_path / "catalog.csv"
        df = generate_residual_catalog(data_dir, out, verbose=False)
        assert set(OUTPUT_COLS).issubset(set(df.columns))

    def test_output_file_created(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=3)
        out = tmp_path / "catalog.csv"
        generate_residual_catalog(data_dir, out, verbose=False)
        assert out.exists()

    def test_output_sorted_by_galaxy(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=5)
        out = tmp_path / "catalog.csv"
        df = generate_residual_catalog(data_dir, out, verbose=False)
        assert list(df["galaxy"]) == sorted(df["galaxy"].tolist())

    def test_n_galaxies_matches_input(self, tmp_path):
        n = 4
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=n)
        out = tmp_path / "catalog.csv"
        df = generate_residual_catalog(data_dir, out, verbose=False)
        assert len(df) == n

    def test_all_f3_residuals_finite(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=4)
        out = tmp_path / "catalog.csv"
        df = generate_residual_catalog(data_dir, out, verbose=False)
        assert df["f3_residual"].notna().all()
        assert np.isfinite(df["f3_residual"].values).all()

    def test_all_v_last_positive(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=4)
        out = tmp_path / "catalog.csv"
        df = generate_residual_catalog(data_dir, out, verbose=False)
        assert (df["v_last"] > 0).all()

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_residual_catalog(tmp_path / "nonexistent",
                                      tmp_path / "out.csv", verbose=False)

    def test_output_csv_readable(self, tmp_path):
        """The written CSV must be readable and contain the expected columns."""
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=3)
        out = tmp_path / "catalog.csv"
        generate_residual_catalog(data_dir, out, verbose=False)
        df_read = pd.read_csv(out)
        assert set(OUTPUT_COLS).issubset(set(df_read.columns))

    def test_v_last_differs_between_galaxies(self, tmp_path):
        """Different v_flat inputs → different v_last values."""
        v_flats = [1.5, 2.0, 2.5, 3.0]
        data_dir = _make_synthetic_sparc_dir(
            tmp_path / "data", n_gal=4, v_flats=v_flats
        )
        out = tmp_path / "catalog.csv"
        df = generate_residual_catalog(data_dir, out, verbose=False)
        assert df["v_last"].nunique() > 1

    def test_cli_main_produces_file(self, tmp_path):
        data_dir = _make_synthetic_sparc_dir(tmp_path / "data", n_gal=3)
        out = tmp_path / "catalog.csv"
        df = generate_main([
            "--data-dir", str(data_dir),
            "--out", str(out),
            "--quiet",
        ])
        assert out.exists()
        assert len(df) == 3
