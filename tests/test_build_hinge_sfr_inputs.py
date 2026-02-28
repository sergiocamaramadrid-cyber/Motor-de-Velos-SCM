"""
tests/test_build_hinge_sfr_inputs.py — Unit tests for
scripts/build_hinge_sfr_inputs_from_sparc.py.

All tests use synthetic SPARC-like fixtures written to a tmp_path;
no real SPARC download is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_hinge_sfr_inputs_from_sparc import (
    DEFAULT_MIN_PTS,
    HE_FACTOR,
    MS_INTERCEPT,
    MS_SLOPE,
    _SPARC_MASS_UNIT,
    build_inputs,
    compute_vbar,
    load_galaxy_table,
    load_rotmod,
    main,
    main_sequence_log_sfr,
    morph_bin,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_galaxy_table(tmp_path: Path, n: int = 5, quality: bool = True) -> Path:
    """Write a minimal synthetic SPARC_Lelli2016c.csv to *tmp_path*."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append({
            "Galaxy": f"NGC{1000 + i}",
            "Mstar": round(float(rng.uniform(0.5, 10.0)), 3),
            "MHI": round(float(rng.uniform(0.1, 3.0)), 3),
            "T": int(rng.integers(-3, 11)),
            "Q": 1 if quality else 3,
        })
    df = pd.DataFrame(rows)
    p = tmp_path / "SPARC_Lelli2016c.csv"
    df.to_csv(p, index=False)
    return p


def _make_rotmod(tmp_path: Path, name: str, n_pts: int = 15) -> Path:
    """Write a synthetic rotmod file for galaxy *name* to *tmp_path*/raw/."""
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    r = np.linspace(0.5, 20.0, n_pts)
    v_obs = 100.0 + 20.0 * rng.standard_normal(n_pts)
    v_gas = 30.0 + 10.0 * rng.standard_normal(n_pts)
    v_disk = 80.0 + 10.0 * rng.standard_normal(n_pts)
    v_bul = 20.0 + 5.0 * rng.standard_normal(n_pts)
    sb_disk = rng.uniform(0.1, 5.0, n_pts)
    sb_bul = rng.uniform(0.0, 2.0, n_pts)
    rows = []
    for i in range(n_pts):
        rows.append(f"{r[i]:.4f} {v_obs[i]:.4f} {abs(v_obs[i]) * 0.05:.4f} "
                    f"{v_gas[i]:.4f} {v_disk[i]:.4f} {v_bul[i]:.4f} "
                    f"{sb_disk[i]:.4f} {sb_bul[i]:.4f}")
    p = raw_dir / f"{name}_rotmod.dat"
    p.write_text("\n".join(rows) + "\n")
    return p


def _make_sparc_fixture(tmp_path: Path, n_gal: int = 5, n_pts: int = 15) -> Path:
    """Create a full synthetic SPARC directory fixture."""
    _make_galaxy_table(tmp_path, n=n_gal)
    for i in range(n_gal):
        _make_rotmod(tmp_path, f"NGC{1000 + i}", n_pts=n_pts)
    return tmp_path


# ---------------------------------------------------------------------------
# morph_bin
# ---------------------------------------------------------------------------

class TestMorphBin:
    def test_late_high_t(self):
        assert morph_bin(7) == "late"

    def test_late_boundary(self):
        assert morph_bin(5) == "late"

    def test_inter(self):
        assert morph_bin(3) == "inter"

    def test_inter_boundary(self):
        assert morph_bin(0) == "inter"

    def test_early(self):
        assert morph_bin(-2) == "early"

    def test_early_boundary(self):
        assert morph_bin(-0.5) == "early"


# ---------------------------------------------------------------------------
# main_sequence_log_sfr
# ---------------------------------------------------------------------------

class TestMainSequenceLogSfr:
    def test_formula(self):
        log_mbar = 10.0
        expected = MS_SLOPE * log_mbar + MS_INTERCEPT
        assert main_sequence_log_sfr(log_mbar) == pytest.approx(expected, rel=1e-9)

    def test_monotone_increasing(self):
        sfr_low = main_sequence_log_sfr(9.0)
        sfr_high = main_sequence_log_sfr(11.0)
        assert sfr_high > sfr_low


# ---------------------------------------------------------------------------
# compute_vbar
# ---------------------------------------------------------------------------

class TestComputeVbar:
    def test_pythagorean(self):
        df = pd.DataFrame({
            "v_gas": [3.0],
            "v_disk": [4.0],
            "v_bul": [0.0],
        })
        vbar = compute_vbar(df)
        np.testing.assert_allclose(vbar, [5.0], rtol=1e-9)

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "v_gas": rng.uniform(-50, 50, 20),
            "v_disk": rng.uniform(50, 200, 20),
            "v_bul": rng.uniform(-20, 20, 20),
        })
        assert np.all(compute_vbar(df) >= 0)

    def test_zero_components(self):
        df = pd.DataFrame({
            "v_gas": [0.0],
            "v_disk": [0.0],
            "v_bul": [0.0],
        })
        np.testing.assert_allclose(compute_vbar(df), [0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# load_rotmod
# ---------------------------------------------------------------------------

class TestLoadRotmod:
    def test_returns_expected_columns(self, tmp_path):
        p = _make_rotmod(tmp_path, "NGC1000")
        df = load_rotmod(p)
        for col in ("r", "v_gas", "v_disk", "v_bul"):
            assert col in df.columns

    def test_row_count_matches(self, tmp_path):
        n = 12
        p = _make_rotmod(tmp_path, "NGC2000", n_pts=n)
        df = load_rotmod(p)
        assert len(df) == n

    def test_radii_positive(self, tmp_path):
        p = _make_rotmod(tmp_path, "NGC3000")
        df = load_rotmod(p)
        assert (df["r"] > 0).all()


# ---------------------------------------------------------------------------
# load_galaxy_table
# ---------------------------------------------------------------------------

class TestLoadGalaxyTable:
    def test_loads_csv(self, tmp_path):
        _make_galaxy_table(tmp_path, n=3)
        gt = load_galaxy_table(tmp_path)
        assert len(gt) == 3
        assert "Galaxy" in gt.columns

    def test_raises_when_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SPARC galaxy table"):
            load_galaxy_table(tmp_path)


# ---------------------------------------------------------------------------
# build_inputs — main builder function
# ---------------------------------------------------------------------------

class TestBuildInputs:
    def test_returns_two_dataframes(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=3)
        profiles, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert isinstance(profiles, pd.DataFrame)
        assert isinstance(gal, pd.DataFrame)

    def test_profiles_has_expected_columns(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=3)
        profiles, _ = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        for col in ("galaxy", "r_kpc", "vbar_kms", "rmax_kpc"):
            assert col in profiles.columns

    def test_galaxy_table_has_expected_columns(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=3)
        _, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        for col in ("galaxy", "log_mbar", "log_sfr", "morph_bin"):
            assert col in gal.columns

    def test_output_files_written(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=3)
        out = tmp_path / "out"
        build_inputs(tmp_path, out, verbose=False)
        assert (out / "profiles.csv").exists()
        assert (out / "galaxy_table.csv").exists()

    def test_n_galaxies_matches(self, tmp_path):
        n = 4
        _make_sparc_fixture(tmp_path, n_gal=n)
        _, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert len(gal) == n

    def test_no_duplicate_galaxy_in_table(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=5)
        _, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert gal["galaxy"].nunique() == len(gal)

    def test_radii_positive(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=3)
        profiles, _ = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert (profiles["r_kpc"] > 0).all()

    def test_vbar_non_negative(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=3)
        profiles, _ = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert (profiles["vbar_kms"] >= 0).all()

    def test_rmax_equals_max_r(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=3)
        profiles, _ = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        for gname, gdf in profiles.groupby("galaxy"):
            assert pytest.approx(float(gdf["rmax_kpc"].iloc[0])) == float(
                gdf["r_kpc"].max()
            )

    def test_log_mbar_finite(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=5)
        _, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert gal["log_mbar"].notna().all()

    def test_log_sfr_finite(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=5)
        _, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert gal["log_sfr"].notna().all()

    def test_morph_bin_values(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=5)
        _, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert set(gal["morph_bin"]).issubset({"late", "inter", "early"})

    def test_min_pts_filter(self, tmp_path):
        """Galaxies with fewer than min_pts radial points must be excluded."""
        _make_galaxy_table(tmp_path, n=3)
        # Write galaxy 0 with only 5 points, others with 15
        _make_rotmod(tmp_path, "NGC1000", n_pts=5)   # will be filtered
        _make_rotmod(tmp_path, "NGC1001", n_pts=15)
        _make_rotmod(tmp_path, "NGC1002", n_pts=15)
        _, gal = build_inputs(
            tmp_path, tmp_path / "out", min_pts=10, verbose=False
        )
        assert "NGC1000" not in gal["galaxy"].values
        assert len(gal) == 2

    def test_missing_rotmod_skipped(self, tmp_path):
        """Galaxies whose rotmod file is absent must be skipped gracefully."""
        _make_galaxy_table(tmp_path, n=3)
        # Only provide rotmod for the first two
        _make_rotmod(tmp_path, "NGC1000")
        _make_rotmod(tmp_path, "NGC1001")
        # NGC1002 has no rotmod
        _, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert "NGC1002" not in gal["galaxy"].values
        assert len(gal) == 2

    def test_raises_when_galaxy_table_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_inputs(tmp_path, tmp_path / "out", verbose=False)

    def test_mass_formula(self, tmp_path):
        """log_mbar = log10((HE_FACTOR*MHI + Mstar) * 1e9)."""
        _make_galaxy_table(tmp_path, n=1)
        _make_rotmod(tmp_path, "NGC1000")
        gt = load_galaxy_table(tmp_path)
        row = gt.iloc[0]
        mbar_expected = (HE_FACTOR * float(row["MHI"]) + float(row["Mstar"])) * _SPARC_MASS_UNIT
        log_mbar_expected = np.log10(mbar_expected)
        _, gal = build_inputs(tmp_path, tmp_path / "out", verbose=False)
        assert gal.loc[gal["galaxy"] == "NGC1000", "log_mbar"].iloc[0] == pytest.approx(
            log_mbar_expected, rel=1e-6
        )


# ---------------------------------------------------------------------------
# External SFR table
# ---------------------------------------------------------------------------

class TestSfrTable:
    def test_uses_real_sfr_when_provided(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=2)
        # Write an SFR table for the first galaxy only
        sfr_table = tmp_path / "sfr.csv"
        sfr_table.write_text("galaxy,log_sfr\nNGC1000,-0.5\n")
        _, gal = build_inputs(
            tmp_path, tmp_path / "out", sfr_table=sfr_table, verbose=False
        )
        row = gal.loc[gal["galaxy"] == "NGC1000"]
        assert row["log_sfr"].iloc[0] == pytest.approx(-0.5, rel=1e-9)

    def test_falls_back_to_proxy_when_galaxy_missing(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=2)
        sfr_table = tmp_path / "sfr.csv"
        sfr_table.write_text("galaxy,log_sfr\nNGC1000,-0.5\n")
        _, gal = build_inputs(
            tmp_path, tmp_path / "out", sfr_table=sfr_table, verbose=False
        )
        row = gal.loc[gal["galaxy"] == "NGC1001"]
        log_mbar = float(row["log_mbar"].iloc[0])
        expected = main_sequence_log_sfr(log_mbar)
        assert row["log_sfr"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_linear_sfr_column_accepted(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=1)
        sfr_table = tmp_path / "sfr.csv"
        # Linear SFR = 0.5 M_sun/yr → log_sfr ≈ -0.301
        sfr_table.write_text("galaxy,sfr\nNGC1000,0.5\n")
        _, gal = build_inputs(
            tmp_path, tmp_path / "out", sfr_table=sfr_table, verbose=False
        )
        expected = np.log10(0.5)
        assert gal.loc[gal["galaxy"] == "NGC1000", "log_sfr"].iloc[0] == pytest.approx(
            expected, rel=1e-6
        )

    def test_invalid_sfr_table_raises(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=1)
        sfr_table = tmp_path / "sfr_bad.csv"
        sfr_table.write_text("galaxy,other_col\nNGC1000,1.0\n")
        with pytest.raises(ValueError, match="log_sfr.*sfr"):
            build_inputs(
                tmp_path, tmp_path / "out", sfr_table=sfr_table, verbose=False
            )


# ---------------------------------------------------------------------------
# main() CLI — smoke test
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_runs_end_to_end(self, tmp_path):
        _make_sparc_fixture(tmp_path, n_gal=3)
        out = tmp_path / "cli_out"
        main([
            "--data-dir", str(tmp_path),
            "--out-dir", str(out),
            "--quiet",
        ])
        assert (out / "profiles.csv").exists()
        assert (out / "galaxy_table.csv").exists()

    def test_quality_filter(self, tmp_path):
        """--quality 1 should reject all galaxies written with Q=3."""
        _make_galaxy_table(tmp_path, n=3, quality=False)   # all Q=3
        for i in range(3):
            _make_rotmod(tmp_path, f"NGC{1000 + i}")
        out = tmp_path / "cli_out"
        main([
            "--data-dir", str(tmp_path),
            "--out-dir", str(out),
            "--quality", "1",
            "--quiet",
        ])
        gal = pd.read_csv(out / "galaxy_table.csv")
        assert len(gal) == 0
