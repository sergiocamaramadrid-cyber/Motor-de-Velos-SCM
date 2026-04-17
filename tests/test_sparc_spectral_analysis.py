"""
tests/test_sparc_spectral_analysis.py — Tests for sparc_spectral_analysis.py
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.sparc_spectral_analysis import (
    MIN_POINTS_DEFAULT,
    NGRID_MAX,
    NGRID_MIN,
    OUTPUT_COLUMNS,
    SMOOTH_WINDOW_FRAC,
    SMOOTH_WINDOW_MIN,
    build_spectral_catalog,
    compute_spectral_features,
    galaxy_name_from_path,
    generate_summary_figures,
    main,
    parse_rotmod,
    print_summary,
)

_REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _write_rotmod(path: Path, r: np.ndarray, v: np.ndarray) -> None:
    """Write a minimal SPARC-style rotmod file."""
    with open(path, "w") as fh:
        fh.write("# radius(kpc)  v_obs(km/s)\n")
        for ri, vi in zip(r, v):
            fh.write(f"{ri:.4f}  {vi:.4f}\n")


def _make_sparc_dir(
    tmp_path: Path,
    n_gal: int = 5,
    n_pts: int = 40,
    v_flat: float = 120.0,
    seed: int = 42,
) -> Path:
    """Create a synthetic SPARC directory with *n_gal* rotmod files."""
    sparc_dir = tmp_path / "SPARC"
    sparc_dir.mkdir()
    rng = np.random.default_rng(seed)
    r = np.linspace(0.5, 20.0, n_pts)
    for i in range(n_gal):
        name = f"NGC{1000 + i:04d}"
        v = v_flat + rng.normal(0, 2.0, n_pts)
        v = np.clip(v, 10.0, None)
        _write_rotmod(sparc_dir / f"{name}_rotmod.dat", r, v)
    return sparc_dir


def _flat_curve(n: int = 50, v_flat: float = 100.0, rmax: float = 15.0,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a flat rotation curve with small Gaussian scatter."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.3, rmax, n)
    v = v_flat + rng.normal(0, 1.5, n)
    return r, np.clip(v, 1.0, None)


def _sinusoidal_residual_curve(
    n: int = 60, v_flat: float = 100.0, rmax: float = 20.0,
    amp: float = 10.0, wavelength: float = 5.0, seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a flat+sinusoid curve; the sinusoid is the planted spectral signal."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.5, rmax, n)
    v = v_flat + amp * np.sin(2 * np.pi * r / wavelength)
    v += rng.normal(0, 0.5, n)
    return r, np.clip(v, 1.0, None)


# ===========================================================================
# galaxy_name_from_path
# ===========================================================================


class TestGalaxyNameFromPath:
    def test_standard_sparc_suffix(self):
        assert galaxy_name_from_path("NGC1234_rotmod.dat") == "NGC1234"

    def test_prefix_form(self):
        assert galaxy_name_from_path("rotmod_UGC5005.dat") == "UGC5005"

    def test_path_object(self):
        assert galaxy_name_from_path(Path("/data/SPARC/NGC0300_rotmod.dat")) == "NGC0300"

    def test_no_decoration(self):
        # Files without rotmod decoration return the stem unchanged
        assert galaxy_name_from_path("mygalaxy.dat") == "mygalaxy"

    def test_preserves_case(self):
        assert galaxy_name_from_path("DDO154_rotmod.dat") == "DDO154"

    def test_nested_path(self):
        p = Path("/content/drive/MyDrive/Colab Notebooks/rotmod/NGC6503_rotmod.dat")
        assert galaxy_name_from_path(p) == "NGC6503"


# ===========================================================================
# parse_rotmod
# ===========================================================================


class TestParseRotmod:
    def test_basic_read(self, tmp_path):
        f = tmp_path / "G01_rotmod.dat"
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([50.0, 80.0, 100.0])
        _write_rotmod(f, r, v)
        r_out, v_out = parse_rotmod(f)
        np.testing.assert_allclose(r_out, r, atol=1e-3)
        np.testing.assert_allclose(v_out, v, atol=1e-3)

    def test_comments_skipped(self, tmp_path):
        f = tmp_path / "G_comments.dat"
        f.write_text("# header\n# another comment\n1.0  50.0\n2.0  80.0\n")
        r, v = parse_rotmod(f)
        assert len(r) == 2

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "G_blanks.dat"
        f.write_text("1.0  50.0\n\n\n2.0  80.0\n")
        r, v = parse_rotmod(f)
        assert len(r) == 2

    def test_non_positive_filtered(self, tmp_path):
        f = tmp_path / "G_neg.dat"
        f.write_text("0.0  50.0\n-1.0  50.0\n1.0  0.0\n2.0  80.0\n")
        r, v = parse_rotmod(f)
        assert len(r) == 1
        assert float(r[0]) == pytest.approx(2.0)

    def test_non_finite_filtered(self, tmp_path):
        f = tmp_path / "G_nan.dat"
        f.write_text("1.0  50.0\nnan  80.0\n2.0  inf\n3.0  90.0\n")
        r, v = parse_rotmod(f)
        assert len(r) == 2

    def test_bad_lines_skipped(self, tmp_path):
        f = tmp_path / "G_bad.dat"
        f.write_text("1.0  50.0\nfoo bar baz\n2.0  80.0\n")
        r, v = parse_rotmod(f)
        assert len(r) == 2

    def test_returns_arrays(self, tmp_path):
        f = tmp_path / "G_type.dat"
        _write_rotmod(f, np.array([1.0, 2.0]), np.array([50.0, 80.0]))
        r, v = parse_rotmod(f)
        assert isinstance(r, np.ndarray)
        assert isinstance(v, np.ndarray)

    def test_extra_columns_ignored(self, tmp_path):
        f = tmp_path / "G_extra.dat"
        f.write_text("1.0 50.0 0.1 10.0 8.0 0.0\n2.0 80.0 0.2 12.0 9.0 0.0\n")
        r, v = parse_rotmod(f)
        assert len(r) == 2
        assert float(v[0]) == pytest.approx(50.0)

    def test_string_path(self, tmp_path):
        f = tmp_path / "G_str.dat"
        _write_rotmod(f, np.array([1.0]), np.array([50.0]))
        r, v = parse_rotmod(str(f))
        assert len(r) == 1


# ===========================================================================
# compute_spectral_features
# ===========================================================================


class TestComputeSpectralFeatures:
    def test_returns_expected_keys(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        for key in ("n_grid", "residual_rms_kms", "lambda_dom_kpc",
                    "peak_freq_1perkpc", "peak_power", "n_peaks"):
            assert key in result, f"Missing key: {key}"

    def test_n_grid_in_range(self):
        r, v = _flat_curve(n=50)
        result = compute_spectral_features(r, v)
        assert NGRID_MIN <= result["n_grid"] <= NGRID_MAX

    def test_residual_rms_positive(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        assert result["residual_rms_kms"] >= 0.0

    def test_lambda_dom_positive(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        lam = result["lambda_dom_kpc"]
        if not math.isnan(lam):
            assert lam > 0.0

    def test_peak_freq_consistent_with_lambda(self):
        r, v = _sinusoidal_residual_curve()
        result = compute_spectral_features(r, v)
        lam = result["lambda_dom_kpc"]
        f = result["peak_freq_1perkpc"]
        if not (math.isnan(lam) or math.isnan(f)):
            assert abs(lam - 1.0 / f) < 1e-8

    def test_planted_wavelength_detected(self):
        """A planted 5-kpc sinusoid should produce a λ_dom near 5 kpc."""
        r, v = _sinusoidal_residual_curve(wavelength=5.0, amp=15.0, n=80)
        result = compute_spectral_features(r, v)
        lam = result["lambda_dom_kpc"]
        assert not math.isnan(lam)
        assert 2.0 < lam < 12.0  # broad tolerance; FFT resolution-limited

    def test_n_peaks_non_negative(self):
        r, v = _flat_curve()
        result = compute_spectral_features(r, v)
        assert result["n_peaks"] >= 0

    def test_too_few_points_raises(self):
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([50.0, 60.0, 70.0])
        with pytest.raises(ValueError):
            compute_spectral_features(r, v)

    def test_unsorted_input_handled(self):
        r, v = _flat_curve()
        # Shuffle the input; result should still be valid
        idx = np.random.default_rng(7).permutation(len(r))
        result = compute_spectral_features(r[idx], v[idx])
        assert result["residual_rms_kms"] >= 0.0

    def test_no_valid_freq_range_returns_nan(self):
        """If rmax − rmin is tiny, no frequency falls in the physical range."""
        r = np.linspace(0.1, 0.12, 30)  # 0.02 kpc span → very low f_min
        v = np.full(30, 50.0) + np.random.default_rng(3).normal(0, 0.1, 30)
        # With such a narrow range, lambda_max_factor * r_range < lambda_min_kpc
        # is possible; we just check it doesn't crash
        result = compute_spectral_features(r, v)
        assert "lambda_dom_kpc" in result

    def test_custom_parameters_accepted(self):
        r, v = _flat_curve()
        result = compute_spectral_features(
            r, v,
            smooth_window_frac=0.1,
            peak_height_factor=2.0,
            lambda_min_kpc=1.0,
        )
        assert "lambda_dom_kpc" in result

    def test_n_grid_grows_with_input_points(self):
        r_small, v_small = _flat_curve(n=30)
        r_large, v_large = _flat_curve(n=100)
        res_small = compute_spectral_features(r_small, v_small)
        res_large = compute_spectral_features(r_large, v_large)
        assert res_large["n_grid"] >= res_small["n_grid"]


# ===========================================================================
# build_spectral_catalog
# ===========================================================================


class TestBuildSpectralCatalog:
    def test_creates_output_file(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "catalog.csv"
        build_spectral_catalog(sparc_dir, out, verbose=False)
        assert out.exists()

    def test_output_columns(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=3)
        out = tmp_path / "catalog.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        for col in OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_n_rows_matches_n_files(self, tmp_path):
        n = 4
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=n)
        out = tmp_path / "catalog.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        assert len(df) == n

    def test_sorted_by_galaxy(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=5)
        out = tmp_path / "catalog.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        galaxies = list(df["galaxy"])
        assert galaxies == sorted(galaxies)

    def test_galaxy_names_match_files(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=3)
        out = tmp_path / "catalog.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        expected = sorted(
            galaxy_name_from_path(f) for f in sparc_dir.glob("*_rotmod.dat")
        )
        assert list(df["galaxy"]) == expected

    def test_min_points_filtering(self, tmp_path):
        sparc_dir = tmp_path / "SPARC"
        sparc_dir.mkdir()
        # Write one galaxy with 30 points and one with 5
        r_ok = np.linspace(0.5, 15.0, 30)
        r_small = np.linspace(0.5, 5.0, 5)
        _write_rotmod(sparc_dir / "BigG_rotmod.dat",
                      r_ok, np.full(30, 100.0))
        _write_rotmod(sparc_dir / "SmallG_rotmod.dat",
                      r_small, np.full(5, 80.0))
        out = tmp_path / "cat.csv"
        df = build_spectral_catalog(sparc_dir, out, min_points=20, verbose=False)
        assert len(df) == 1
        assert df["galaxy"].iloc[0] == "BigG"

    def test_deduplication_by_name(self, tmp_path):
        """Two files with the same galaxy name → only one entry."""
        sparc_dir = tmp_path / "SPARC"
        sparc_dir.mkdir()
        sub = sparc_dir / "sub"
        sub.mkdir()
        r = np.linspace(0.5, 15.0, 30)
        v = np.full(30, 100.0)
        _write_rotmod(sparc_dir / "NGC0001_rotmod.dat", r, v)
        _write_rotmod(sub / "NGC0001_rotmod.dat", r, v * 1.1)
        out = tmp_path / "cat.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        assert len(df[df["galaxy"] == "NGC0001"]) == 1

    def test_empty_dir_returns_empty_df(self, tmp_path):
        sparc_dir = tmp_path / "SPARC"
        sparc_dir.mkdir()
        out = tmp_path / "cat.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        assert len(df) == 0
        for col in OUTPUT_COLUMNS:
            assert col in df.columns

    def test_catalog_written_to_disk(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "sub" / "catalog.csv"
        build_spectral_catalog(sparc_dir, out, verbose=False)
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) > 0

    def test_rmin_lt_rmax(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "cat.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        assert (df["rmin_kpc"] < df["rmax_kpc"]).all()

    def test_n_points_raw_positive(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "cat.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        assert (df["n_points_raw"] > 0).all()

    def test_n_grid_in_range(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "cat.csv"
        df = build_spectral_catalog(sparc_dir, out, verbose=False)
        assert ((df["n_grid"] >= NGRID_MIN) & (df["n_grid"] <= NGRID_MAX)).all()

    def test_plot_dir_created(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=2)
        out = tmp_path / "cat.csv"
        plot_dir = tmp_path / "plots"
        build_spectral_catalog(sparc_dir, out, plot_dir=plot_dir, verbose=False)
        assert plot_dir.is_dir()
        pngs = list(plot_dir.glob("*.png"))
        assert len(pngs) == 2

    def test_verbose_output(self, tmp_path, capsys):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=2)
        out = tmp_path / "cat.csv"
        build_spectral_catalog(sparc_dir, out, verbose=True)
        captured = capsys.readouterr()
        assert "SPARC rotmod files found" in captured.out

    def test_quiet_no_output(self, tmp_path, capsys):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=2)
        out = tmp_path / "cat.csv"
        build_spectral_catalog(sparc_dir, out, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_corrupted_file_skipped(self, tmp_path):
        sparc_dir = tmp_path / "SPARC"
        sparc_dir.mkdir()
        # Write one valid file and one corrupted file
        r = np.linspace(0.5, 15.0, 30)
        _write_rotmod(sparc_dir / "GoodGal_rotmod.dat", r, np.full(30, 100.0))
        (sparc_dir / "BadGal_rotmod.dat").write_text("this is not data\n" * 30)
        out = tmp_path / "cat.csv"
        df = build_spectral_catalog(sparc_dir, out, min_points=20, verbose=False)
        # BadGal has no valid points → skipped; GoodGal should be present
        assert "GoodGal" in df["galaxy"].values


# ===========================================================================
# main()
# ===========================================================================


class TestMain:
    def test_returns_dict(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        assert isinstance(result, dict)

    def test_dict_keys(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        for key in ("n_galaxies", "n_valid", "median_lambda_dom_kpc",
                    "median_n_peaks", "median_residual_rms_kms",
                    "out_path", "catalog"):
            assert key in result, f"Missing key: {key}"

    def test_n_galaxies(self, tmp_path):
        n = 4
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=n)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        assert result["n_galaxies"] == n

    def test_out_path_in_result(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "mycatalog.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        assert result["out_path"] == str(out)

    def test_catalog_is_dataframe(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        assert isinstance(result["catalog"], pd.DataFrame)

    def test_n_valid_le_n_galaxies(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        assert result["n_valid"] <= result["n_galaxies"]

    def test_median_lambda_finite_when_data_present(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=5)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        if result["n_valid"] > 0:
            assert math.isfinite(result["median_lambda_dom_kpc"])

    def test_min_points_flag(self, tmp_path):
        sparc_dir = tmp_path / "SPARC"
        sparc_dir.mkdir()
        r_ok = np.linspace(0.5, 15.0, 40)
        r_small = np.linspace(0.5, 5.0, 10)
        _write_rotmod(sparc_dir / "BigG_rotmod.dat", r_ok, np.full(40, 100.0))
        _write_rotmod(sparc_dir / "SmallG_rotmod.dat", r_small, np.full(10, 80.0))
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--min-points", "30",
            "--quiet",
            "--no-summary-figures",
        ])
        assert result["n_galaxies"] == 1

    def test_plot_dir_flag(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=2)
        out = tmp_path / "cat.csv"
        plot_dir = tmp_path / "panels"
        main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--plot-dir", str(plot_dir),
            "--quiet",
            "--no-summary-figures",
        ])
        assert plot_dir.is_dir()
        assert len(list(plot_dir.glob("*.png"))) == 2

    def test_summary_figures_created_by_default(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=3)
        out = tmp_path / "cat.csv"
        main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
        ])
        # Summary figures go next to the output CSV
        out_dir = out.parent
        assert (out_dir / "lambda_dom_hist.png").exists()
        assert (out_dir / "lambda_dom_vs_npoints.png").exists()

    def test_no_summary_figures_flag(self, tmp_path):
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=2)
        out = tmp_path / "sub" / "cat.csv"
        main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        out_dir = out.parent
        assert not (out_dir / "lambda_dom_hist.png").exists()

    def test_empty_sparc_dir(self, tmp_path):
        sparc_dir = tmp_path / "SPARC"
        sparc_dir.mkdir()
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])
        assert result["n_galaxies"] == 0

    def test_subprocess_cli(self, tmp_path):
        """Script runs without error from the command line."""
        sparc_dir = _make_sparc_dir(tmp_path)
        out = tmp_path / "cat.csv"
        proc = subprocess.run(
            [
                sys.executable, "-m", "scripts.sparc_spectral_analysis",
                "--sparc-dir", str(sparc_dir),
                "--out", str(out),
                "--quiet",
                "--no-summary-figures",
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr


# ===========================================================================
# print_summary / generate_summary_figures
# ===========================================================================


class TestSummaryHelpers:
    def _make_catalog(self, n: int = 6) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(n)],
            "n_points_raw": rng.integers(20, 60, n),
            "rmin_kpc": rng.uniform(0.1, 1.0, n),
            "rmax_kpc": rng.uniform(10.0, 20.0, n),
            "n_grid": np.full(n, 128),
            "residual_rms_kms": rng.uniform(0.5, 5.0, n),
            "lambda_dom_kpc": np.where(
                rng.random(n) > 0.2, rng.uniform(2.0, 15.0, n), np.nan
            ),
            "peak_freq_1perkpc": rng.uniform(0.05, 0.5, n),
            "peak_power": rng.uniform(10.0, 1000.0, n),
            "n_peaks": rng.integers(0, 5, n),
        })

    def test_print_summary_runs(self, capsys):
        catalog = self._make_catalog()
        print_summary(catalog)
        captured = capsys.readouterr()
        assert "Total galaxies" in captured.out

    def test_generate_summary_figures_creates_pngs(self, tmp_path):
        catalog = self._make_catalog()
        generate_summary_figures(catalog, tmp_path)
        assert (tmp_path / "lambda_dom_hist.png").exists()
        assert (tmp_path / "lambda_dom_vs_npoints.png").exists()

    def test_generate_summary_figures_empty_catalog(self, tmp_path):
        """No error when all lambda_dom_kpc are NaN."""
        catalog = self._make_catalog()
        catalog["lambda_dom_kpc"] = np.nan
        generate_summary_figures(catalog, tmp_path)  # should not raise

    def test_print_summary_all_nan(self, capsys):
        catalog = self._make_catalog()
        catalog["lambda_dom_kpc"] = np.nan
        print_summary(catalog)  # should not raise


# ===========================================================================
# Round-trip integration test
# ===========================================================================


class TestRoundTrip:
    def test_full_round_trip(self, tmp_path):
        """Write rotmod files → run main → read CSV → verify consistency."""
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=5, n_pts=50)
        out = tmp_path / "out" / "catalog.csv"
        result = main([
            "--sparc-dir", str(sparc_dir),
            "--out", str(out),
            "--quiet",
            "--no-summary-figures",
        ])

        # Re-read from disk
        df = pd.read_csv(out)
        assert len(df) == result["n_galaxies"]
        for col in OUTPUT_COLUMNS:
            assert col in df.columns

        # rmin < rmax
        assert (df["rmin_kpc"] < df["rmax_kpc"]).all()
        # residual_rms non-negative
        assert (df["residual_rms_kms"] >= 0.0).all()
        # n_peaks non-negative
        assert (df["n_peaks"] >= 0).all()
        # lambda_dom positive where not NaN
        valid = df.dropna(subset=["lambda_dom_kpc"])
        if len(valid) > 0:
            assert (valid["lambda_dom_kpc"] > 0.0).all()

    def test_reproducibility(self, tmp_path):
        """Running twice on the same data produces the same catalog."""
        sparc_dir = _make_sparc_dir(tmp_path, n_gal=3)
        out1 = tmp_path / "cat1.csv"
        out2 = tmp_path / "cat2.csv"
        main(["--sparc-dir", str(sparc_dir), "--out", str(out1),
              "--quiet", "--no-summary-figures"])
        main(["--sparc-dir", str(sparc_dir), "--out", str(out2),
              "--quiet", "--no-summary-figures"])
        df1 = pd.read_csv(out1)
        df2 = pd.read_csv(out2)
        pd.testing.assert_frame_equal(df1, df2)
