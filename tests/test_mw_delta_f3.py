"""
tests/test_mw_delta_f3.py — Tests for scripts/mw_delta_f3.py.

Covers:
  1. compute_slope()   — log-log OLS, weighted and unweighted.
  2. scan_r_cuts()     — radial threshold scan.
  3. generate_figure() — figure creation and PNG/PDF output.
  4. main()            — end-to-end CLI and regression guard on mw_cepheids.csv.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.mw_delta_f3 import (
    BETA_REF,
    FIGURE_CAPTION,
    R_CUT_DEFAULT,
    _SCORE_EPS,
    compute_slope,
    find_best_r_cut,
    generate_figure,
    main,
    scan_r_cuts,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_curve(n: int = 20, R_min: float = 4.0, R_max: float = 22.0) -> pd.DataFrame:
    """Exactly flat rotation curve (slope = 0)."""
    R = np.linspace(R_min, R_max, n)
    V = np.full(n, 220.0)
    return pd.DataFrame({"R_kpc": R, "Vc_kms": V})


def _declining_curve(
    n: int = 20,
    R_min: float = 4.0,
    R_max: float = 22.0,
    slope: float = -0.1,
    V0: float = 220.0,
) -> pd.DataFrame:
    """Power-law rotation curve V ∝ R^slope (exact log-log relationship)."""
    R = np.linspace(R_min, R_max, n)
    V = V0 * (R / R_min) ** slope
    return pd.DataFrame({"R_kpc": R, "Vc_kms": V})


def _curve_with_errors(
    n: int = 20,
    R_min: float = 4.0,
    R_max: float = 22.0,
    slope: float = -0.1,
) -> pd.DataFrame:
    df = _declining_curve(n, R_min, R_max, slope)
    df["e_Vc"] = 5.0
    return df


def _write_csv(df: pd.DataFrame, tmp_path: Path) -> Path:
    p = tmp_path / "rc.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. compute_slope
# ---------------------------------------------------------------------------


class TestComputeSlope:
    def test_returns_expected_keys(self):
        R = np.linspace(5.0, 20.0, 10)
        V = np.full(10, 230.0)
        result = compute_slope(R, V)
        assert set(result.keys()) == {"slope_tail", "intercept", "delta_f3", "p_slope", "n"}

    def test_n_equals_input_length(self):
        R = np.linspace(5.0, 20.0, 15)
        V = np.full(15, 230.0)
        assert compute_slope(R, V)["n"] == 15

    def test_flat_curve_slope_near_zero(self):
        R = np.linspace(5.0, 20.0, 50)
        V = np.full(50, 230.0)
        result = compute_slope(R, V)
        assert abs(result["slope_tail"]) < 1e-6

    def test_flat_curve_delta_f3_near_minus_half(self):
        R = np.linspace(5.0, 20.0, 50)
        V = np.full(50, 230.0)
        result = compute_slope(R, V)
        assert abs(result["delta_f3"] - (-BETA_REF)) < 1e-6

    def test_power_law_slope_recovered(self):
        """For V ∝ R^(-0.1), the slope must be -0.1 within 0.01."""
        true_slope = -0.1
        R = np.linspace(5.0, 22.0, 100)
        V = 230.0 * (R / 5.0) ** true_slope
        result = compute_slope(R, V)
        assert abs(result["slope_tail"] - true_slope) < 0.01

    def test_delta_f3_equals_slope_minus_beta(self):
        R = np.linspace(5.0, 20.0, 20)
        V = np.random.default_rng(42).uniform(200, 240, 20)
        result = compute_slope(R, V)
        assert math.isclose(result["delta_f3"], result["slope_tail"] - BETA_REF)

    def test_weighted_returns_same_keys(self):
        R = np.linspace(5.0, 20.0, 10)
        V = np.full(10, 230.0)
        w = np.ones(10) * 0.04
        result = compute_slope(R, V, weights=w)
        assert set(result.keys()) == {"slope_tail", "intercept", "delta_f3", "p_slope", "n"}

    def test_weighted_flat_slope_near_zero(self):
        R = np.linspace(5.0, 20.0, 30)
        V = np.full(30, 230.0)
        w = np.ones(30)
        result = compute_slope(R, V, weights=w)
        assert abs(result["slope_tail"]) < 1e-6

    def test_weighted_power_law_slope_recovered(self):
        true_slope = -0.2
        R = np.linspace(5.0, 22.0, 100)
        V = 230.0 * (R / 5.0) ** true_slope
        w = np.ones(100)
        result = compute_slope(R, V, weights=w)
        assert abs(result["slope_tail"] - true_slope) < 0.01

    def test_raises_with_single_point(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_slope(np.array([10.0]), np.array([230.0]))

    def test_slope_sign_negative_for_declining(self):
        R = np.linspace(10.0, 22.0, 20)
        V = 230.0 * (R / 10.0) ** (-0.15)
        result = compute_slope(R, V)
        assert result["slope_tail"] < 0

    def test_delta_f3_negative_for_declining(self):
        R = np.linspace(10.0, 22.0, 20)
        V = 230.0 * (R / 10.0) ** (-0.15)
        result = compute_slope(R, V)
        assert result["delta_f3"] < 0


# ---------------------------------------------------------------------------
# 2. scan_r_cuts
# ---------------------------------------------------------------------------


class TestScanRCuts:
    def test_returns_dataframe(self):
        df = _flat_curve(30)
        result = scan_r_cuts(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        df = _flat_curve(30)
        result = scan_r_cuts(df)
        assert set(result.columns) >= {"r_cut", "slope_tail", "delta_f3", "p_slope", "n"}

    def test_r_cut_monotone(self):
        df = _flat_curve(40)
        result = scan_r_cuts(df, r_start=8.0, r_stop=14.0, r_step=1.0)
        if len(result) > 1:
            assert (result["r_cut"].diff().dropna() >= 0).all()

    def test_n_decreases_with_larger_r_cut(self):
        df = _flat_curve(40)
        result = scan_r_cuts(df, r_start=5.0, r_stop=15.0, r_step=2.0)
        if len(result) > 1:
            assert (result["n"].diff().dropna() <= 0).all()

    def test_empty_if_n_min_too_large(self):
        df = _flat_curve(5)
        result = scan_r_cuts(df, r_start=4.0, r_stop=20.0, r_step=1.0, n_min=10)
        assert len(result) == 0

    def test_flat_curve_delta_f3_near_minus_half(self):
        df = _flat_curve(40)
        result = scan_r_cuts(df, r_start=5.0, r_stop=10.0, r_step=1.0, n_min=3)
        assert (result["delta_f3"].abs() - abs(-BETA_REF)).abs().max() < 0.01

    def test_uses_weights_when_e_vc_present(self):
        df = _curve_with_errors(40)
        result = scan_r_cuts(df, r_start=5.0, r_stop=12.0, r_step=1.0, n_min=3)
        assert not result.empty

    def test_non_empty_for_valid_data(self):
        df = _declining_curve(40)
        result = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=1.0, n_min=3)
        assert not result.empty

    def test_scan_slope_negative_for_declining(self):
        df = _declining_curve(40, slope=-0.1)
        result = scan_r_cuts(df, r_start=5.0, r_stop=12.0, r_step=2.0, n_min=3)
        assert (result["slope_tail"] < 0).all()

    def test_p_slope_column_present(self):
        df = _declining_curve(40, slope=-0.1)
        result = scan_r_cuts(df, r_start=5.0, r_stop=12.0, r_step=2.0, n_min=3)
        assert "p_slope" in result.columns

    def test_p_slope_values_in_01(self):
        df = _declining_curve(40, slope=-0.1)
        result = scan_r_cuts(df, r_start=5.0, r_stop=12.0, r_step=2.0, n_min=3)
        assert (result["p_slope"] >= 0).all()
        assert (result["p_slope"] <= 1).all()


# ---------------------------------------------------------------------------
# 3. find_best_r_cut
# ---------------------------------------------------------------------------


class TestFindBestRCut:
    def test_returns_dict_with_expected_keys(self):
        df = _declining_curve(40, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=1.0, n_min=3)
        result = find_best_r_cut(scan_df)
        assert set(result.keys()) == {"r_crit", "slope_tail", "delta_f3", "p_slope", "n", "score"}

    def test_r_crit_in_scan_range(self):
        df = _declining_curve(40, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=1.0, n_min=3)
        result = find_best_r_cut(scan_df)
        assert scan_df["r_cut"].min() <= result["r_crit"] <= scan_df["r_cut"].max()

    def test_score_is_positive(self):
        df = _declining_curve(40, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=1.0, n_min=3)
        result = find_best_r_cut(scan_df)
        assert result["score"] > 0

    def test_slope_matches_row_in_scan_df(self):
        df = _declining_curve(40, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=1.0, n_min=3)
        result = find_best_r_cut(scan_df)
        row = scan_df[scan_df["r_cut"] == result["r_crit"]].iloc[0]
        assert math.isclose(result["slope_tail"], row["slope_tail"])

    def test_delta_f3_matches_row(self):
        df = _declining_curve(40, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=1.0, n_min=3)
        result = find_best_r_cut(scan_df)
        row = scan_df[scan_df["r_cut"] == result["r_crit"]].iloc[0]
        assert math.isclose(result["delta_f3"], row["delta_f3"])

    def test_p_slope_matches_row(self):
        df = _declining_curve(40, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=1.0, n_min=3)
        result = find_best_r_cut(scan_df)
        row = scan_df[scan_df["r_cut"] == result["r_crit"]].iloc[0]
        assert math.isclose(result["p_slope"], row["p_slope"])

    def test_raises_on_empty_scan_df(self):
        empty = pd.DataFrame(columns=["r_cut", "slope_tail", "delta_f3", "p_slope", "n"])
        with pytest.raises(ValueError, match="empty"):
            find_best_r_cut(empty)

    def test_score_eps_constant_positive(self):
        assert _SCORE_EPS > 0

    def test_n_matches_row(self):
        df = _declining_curve(40, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=1.0, n_min=3)
        result = find_best_r_cut(scan_df)
        row = scan_df[scan_df["r_cut"] == result["r_crit"]].iloc[0]
        assert result["n"] == int(row["n"])


# ---------------------------------------------------------------------------
# 4. generate_figure
# ---------------------------------------------------------------------------


class TestGenerateFigure:
    def test_creates_png(self, tmp_path):
        df = _flat_curve(30)
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert out.exists()

    def test_creates_sibling_pdf(self, tmp_path):
        df = _flat_curve(30)
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert (tmp_path / "fig.pdf").exists()

    def test_returns_figure(self, tmp_path):
        df = _flat_curve(30)
        out = tmp_path / "fig.png"
        fig = generate_figure(df, out)
        assert isinstance(fig, plt.Figure)

    def test_figure_has_two_axes(self, tmp_path):
        df = _flat_curve(30)
        out = tmp_path / "fig.png"
        fig = generate_figure(df, out)
        assert len(fig.axes) >= 2

    def test_custom_r_cut(self, tmp_path):
        df = _flat_curve(30)
        out = tmp_path / "fig.png"
        generate_figure(df, out, r_cut=10.0)
        assert out.exists()

    def test_with_error_column(self, tmp_path):
        df = _curve_with_errors(30)
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        df = _flat_curve(30)
        out = tmp_path / "sub" / "dir" / "fig.png"
        generate_figure(df, out)
        assert out.exists()

    def test_png_non_empty(self, tmp_path):
        df = _flat_curve(30)
        out = tmp_path / "fig.png"
        generate_figure(df, out)
        assert out.stat().st_size > 10_000

    def test_with_best_r_cut(self, tmp_path):
        df = _declining_curve(30, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=2.0, n_min=3)
        best = find_best_r_cut(scan_df)
        out = tmp_path / "fig_best.png"
        fig = generate_figure(df, out, best=best)
        assert out.exists()
        assert isinstance(fig, plt.Figure)

    def test_figure_three_axes_with_best(self, tmp_path):
        """Twin axis added when best is provided → 3 axes total."""
        df = _declining_curve(30, slope=-0.15)
        scan_df = scan_r_cuts(df, r_start=5.0, r_stop=14.0, r_step=2.0, n_min=3)
        best = find_best_r_cut(scan_df)
        out = tmp_path / "fig_twin.png"
        fig = generate_figure(df, out, best=best)
        assert len(fig.axes) >= 2


# ---------------------------------------------------------------------------
# 5. main() — CLI and regression guard
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_dict_keys(self, tmp_path):
        df = _flat_curve(30)
        csv = _write_csv(df, tmp_path)
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out])
        assert set(result.keys()) >= {"slope", "r_cut", "r_crit", "best", "scan_df", "figure_path", "pdf_path"}

    def test_slope_dict_keys(self, tmp_path):
        df = _flat_curve(30)
        csv = _write_csv(df, tmp_path)
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out])
        assert set(result["slope"].keys()) >= {"slope_tail", "intercept", "delta_f3", "p_slope", "n"}

    def test_r_cut_returned(self, tmp_path):
        df = _flat_curve(30)
        csv = _write_csv(df, tmp_path)
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out, "--r-cut", "10.0"])
        assert result["r_cut"] == 10.0

    def test_figure_path_exists(self, tmp_path):
        df = _flat_curve(30)
        csv = _write_csv(df, tmp_path)
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out])
        assert Path(result["figure_path"]).exists()

    def test_pdf_path_exists(self, tmp_path):
        df = _flat_curve(30)
        csv = _write_csv(df, tmp_path)
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out])
        assert Path(result["pdf_path"]).exists()

    def test_scan_df_is_dataframe(self, tmp_path):
        df = _flat_curve(30)
        csv = _write_csv(df, tmp_path)
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out])
        assert isinstance(result["scan_df"], pd.DataFrame)

    def test_missing_csv_raises(self, tmp_path):
        out = str(tmp_path / "fig.png")
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "nonexistent.csv"), "--out", out])

    def test_missing_column_raises(self, tmp_path):
        df = pd.DataFrame({"R_kpc": [5, 10, 15, 20]})  # missing Vc_kms
        csv = _write_csv(df, tmp_path)
        out = str(tmp_path / "fig.png")
        with pytest.raises(ValueError, match="missing required columns"):
            main(["--csv", str(csv), "--out", out])

    def test_insufficient_outer_data_raises(self, tmp_path):
        df = pd.DataFrame({
            "R_kpc": [5.0, 7.0],
            "Vc_kms": [230.0, 228.0],
        })
        csv = _write_csv(df, tmp_path)
        out = str(tmp_path / "fig.png")
        with pytest.raises(ValueError, match="at least 2"):
            main(["--csv", str(csv), "--r-cut", "20.0", "--out", out])

    # Regression guard on committed mw_cepheids.csv
    def test_regression_on_mw_cepheids_slope_negative(self, tmp_path):
        """slope_tail must be negative for the outer region of mw_cepheids.csv."""
        repo_root = Path(__file__).parent.parent
        csv = repo_root / "data" / "mw_cepheids.csv"
        if not csv.exists():
            pytest.skip("data/mw_cepheids.csv not found")
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out, "--r-cut", "13.0"])
        assert result["slope"]["slope_tail"] < 0

    def test_regression_on_mw_cepheids_delta_f3_negative(self, tmp_path):
        """delta_f3 must be negative: outer MW curve declines faster than MOND ref."""
        repo_root = Path(__file__).parent.parent
        csv = repo_root / "data" / "mw_cepheids.csv"
        if not csv.exists():
            pytest.skip("data/mw_cepheids.csv not found")
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out, "--r-cut", "13.0"])
        assert result["slope"]["delta_f3"] < 0

    def test_regression_on_mw_cepheids_n_outer(self, tmp_path):
        """At R_cut=13 kpc there must be >= 10 data points."""
        repo_root = Path(__file__).parent.parent
        csv = repo_root / "data" / "mw_cepheids.csv"
        if not csv.exists():
            pytest.skip("data/mw_cepheids.csv not found")
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out, "--r-cut", "13.0"])
        assert result["slope"]["n"] >= 10

    def test_regression_r_crit_is_set(self, tmp_path):
        """r_crit must be returned and be a float within the scan range."""
        repo_root = Path(__file__).parent.parent
        csv = repo_root / "data" / "mw_cepheids.csv"
        if not csv.exists():
            pytest.skip("data/mw_cepheids.csv not found")
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out])
        assert result["r_crit"] is not None
        assert isinstance(result["r_crit"], float)

    def test_regression_best_keys(self, tmp_path):
        repo_root = Path(__file__).parent.parent
        csv = repo_root / "data" / "mw_cepheids.csv"
        if not csv.exists():
            pytest.skip("data/mw_cepheids.csv not found")
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out])
        assert set(result["best"].keys()) == {"r_crit", "slope_tail", "delta_f3", "p_slope", "n", "score"}

    def test_scan_df_has_p_slope(self, tmp_path):
        repo_root = Path(__file__).parent.parent
        csv = repo_root / "data" / "mw_cepheids.csv"
        if not csv.exists():
            pytest.skip("data/mw_cepheids.csv not found")
        out = str(tmp_path / "fig.png")
        result = main(["--csv", str(csv), "--out", out])
        assert "p_slope" in result["scan_df"].columns


# ---------------------------------------------------------------------------
# 6. _parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.r_cut == R_CUT_DEFAULT
        assert args.out == "results/mw_delta_f3.png"

    def test_custom_r_cut(self):
        args = _parse_args(["--r-cut", "15.0"])
        assert args.r_cut == 15.0

    def test_custom_out(self):
        args = _parse_args(["--out", "/tmp/x.png"])
        assert args.out == "/tmp/x.png"

    def test_custom_csv(self):
        args = _parse_args(["--csv", "/tmp/rc.csv"])
        assert args.csv == "/tmp/rc.csv"


# ---------------------------------------------------------------------------
# 7. Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_beta_ref_is_half(self):
        assert BETA_REF == 0.5

    def test_r_cut_default_positive(self):
        assert R_CUT_DEFAULT > 0

    def test_figure_caption_is_string(self):
        assert isinstance(FIGURE_CAPTION, str)

    def test_figure_caption_non_empty(self):
        assert len(FIGURE_CAPTION) > 20
