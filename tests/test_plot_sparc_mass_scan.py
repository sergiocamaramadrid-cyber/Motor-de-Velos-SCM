"""
tests/test_plot_sparc_mass_scan.py -- Tests for scripts/plot_sparc_mass_scan.py.

Covers:
  1. scan_mass_thresholds() -- threshold loop and output schema.
  2. find_best_cut() -- best-score selection and edge cases.
  3. generate_figure() -- figure creation and file output.
  4. main() CLI -- end-to-end invocation.
  5. Integration: committed SPARC subset fixture (regression guard).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.plot_sparc_mass_scan import (
    BETA_REF,
    M_START_DEFAULT,
    M_STOP_DEFAULT,
    M_STEP_DEFAULT,
    N_MIN_DEFAULT,
    _SCORE_EPS,
    find_best_cut,
    generate_figure,
    main,
    scan_mass_thresholds,
    _parse_args,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(
    n: int = 40,
    seed: int = 0,
    logM_range: tuple = (10.0, 11.5),
) -> pd.DataFrame:
    """Minimal synthetic catalog with required columns + delta_f3."""
    rng = np.random.default_rng(seed)
    logM = rng.uniform(*logM_range, n)
    delta_mass_std = rng.normal(0.0, 1.0, n)
    slope_tail = 0.5 + rng.normal(0.0, 0.1, n)
    df = pd.DataFrame(
        {
            "galaxy": [f"NGC{i:04d}" for i in range(n)],
            "logM": logM,
            "delta_mass_std": delta_mass_std,
            "slope_tail": slope_tail,
        }
    )
    df["delta_f3"] = df["slope_tail"] - BETA_REF
    return df


def _write_catalog(df: pd.DataFrame, tmp_path: Path) -> Path:
    p = tmp_path / "catalog.csv"
    # Write without delta_f3 -- main() must compute it
    cols = [c for c in df.columns if c != "delta_f3"]
    df[cols].to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. scan_mass_thresholds()
# ---------------------------------------------------------------------------

class TestScanMassThresholds:
    def test_returns_dataframe(self):
        df = _make_catalog()
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.0, m_step=0.1)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        df = _make_catalog()
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.0, m_step=0.1)
        for col in ("m_cut", "rho", "p", "N", "score"):
            assert col in result.columns

    def test_m_cut_values_in_range(self):
        df = _make_catalog()
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.0, m_step=0.1)
        if not result.empty:
            assert result["m_cut"].min() >= 10.0
            assert result["m_cut"].max() < 11.0

    def test_n_column_is_integer_like(self):
        df = _make_catalog()
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.0, m_step=0.1)
        if not result.empty:
            assert (result["N"] == result["N"].astype(int)).all()

    def test_n_min_filter_respected(self):
        df = _make_catalog(n=60)
        result = scan_mass_thresholds(
            df, m_start=10.0, m_stop=11.5, m_step=0.1, n_min=20
        )
        if not result.empty:
            assert (result["N"] > 20).all()

    def test_score_is_non_negative(self):
        df = _make_catalog()
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.0, m_step=0.1)
        if not result.empty:
            assert (result["score"] >= 0).all()

    def test_p_values_in_valid_range(self):
        df = _make_catalog()
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.0, m_step=0.1)
        if not result.empty:
            assert (result["p"] >= 0).all()
            assert (result["p"] <= 1).all()

    def test_rho_in_valid_range(self):
        df = _make_catalog()
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.0, m_step=0.1)
        if not result.empty:
            assert (result["rho"].abs() <= 1.0).all()

    def test_empty_when_all_below_n_min(self):
        df = _make_catalog(n=5)
        result = scan_mass_thresholds(
            df, m_start=10.0, m_stop=11.0, m_step=0.1, n_min=100
        )
        assert result.empty

    def test_rows_sorted_by_m_cut(self):
        df = _make_catalog(n=80)
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.5, m_step=0.1)
        if len(result) > 1:
            assert (result["m_cut"].diff().dropna() > 0).all()

    def test_n_decreases_or_stays_as_cut_increases(self):
        df = _make_catalog(n=60)
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=11.5, m_step=0.1)
        if len(result) > 1:
            assert (result["N"].diff().dropna() <= 0).all()

    def test_score_formula_correct(self):
        df = _make_catalog(n=80, seed=3)
        result = scan_mass_thresholds(df, m_start=10.0, m_stop=10.2, m_step=0.1)
        if not result.empty:
            row = result.iloc[0]
            expected = abs(row["rho"]) * np.sqrt(row["N"]) * (
                -np.log10(row["p"] + _SCORE_EPS)
            )
            assert row["score"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. find_best_cut()
# ---------------------------------------------------------------------------

class TestFindBestCut:
    def _make_scan_df(self) -> pd.DataFrame:
        df = _make_catalog(n=60)
        return scan_mass_thresholds(df, m_start=10.0, m_stop=11.5, m_step=0.1)

    def test_returns_dict(self):
        scan_df = self._make_scan_df()
        result = find_best_cut(scan_df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        scan_df = self._make_scan_df()
        result = find_best_cut(scan_df)
        for key in ("m_cut", "rho", "p", "N", "score"):
            assert key in result

    def test_score_is_max(self):
        scan_df = self._make_scan_df()
        best = find_best_cut(scan_df)
        assert best["score"] == pytest.approx(scan_df["score"].max(), rel=1e-9)

    def test_raises_on_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["m_cut", "rho", "p", "N", "score"])
        with pytest.raises(ValueError, match="scan_df is empty"):
            find_best_cut(empty_df)

    def test_m_cut_is_in_scan_range(self):
        scan_df = self._make_scan_df()
        best = find_best_cut(scan_df)
        assert best["m_cut"] in scan_df["m_cut"].values

    def test_best_score_exceeds_all_others(self):
        scan_df = self._make_scan_df()
        best = find_best_cut(scan_df)
        assert (scan_df["score"] <= best["score"]).all()


# ---------------------------------------------------------------------------
# 3. generate_figure()
# ---------------------------------------------------------------------------

class TestGenerateFigure:
    def _scan(self) -> pd.DataFrame:
        df = _make_catalog(n=60)
        return scan_mass_thresholds(df, m_start=10.0, m_stop=11.5, m_step=0.1)

    def test_returns_figure(self, tmp_path):
        import matplotlib.pyplot as plt
        scan_df = self._scan()
        fig = generate_figure(scan_df, tmp_path / "fig.png")
        assert isinstance(fig, plt.Figure)

    def test_png_created(self, tmp_path):
        scan_df = self._scan()
        out = tmp_path / "fig.png"
        generate_figure(scan_df, out)
        assert out.exists()

    def test_pdf_sibling_created(self, tmp_path):
        scan_df = self._scan()
        out = tmp_path / "fig.png"
        generate_figure(scan_df, out)
        assert out.with_suffix(".pdf").exists()

    def test_png_nonzero_size(self, tmp_path):
        scan_df = self._scan()
        out = tmp_path / "fig.png"
        generate_figure(scan_df, out)
        assert out.stat().st_size > 0

    def test_accepts_string_path(self, tmp_path):
        scan_df = self._scan()
        out = str(tmp_path / "fig.png")
        generate_figure(scan_df, out)
        assert Path(out).exists()

    def test_creates_output_directory(self, tmp_path):
        scan_df = self._scan()
        out = tmp_path / "subdir" / "fig.png"
        generate_figure(scan_df, out)
        assert out.exists()

    def test_accepts_pre_computed_best(self, tmp_path):
        scan_df = self._scan()
        best = find_best_cut(scan_df)
        out = tmp_path / "fig.png"
        generate_figure(scan_df, out, best=best)
        assert out.exists()

    def test_figure_has_one_axis(self, tmp_path):
        scan_df = self._scan()
        out = tmp_path / "fig.png"
        fig = generate_figure(scan_df, out)
        assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# 4. main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_returns_dict(self, tmp_path):
        csv = _write_catalog(_make_catalog(), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        assert isinstance(result, dict)

    def test_required_keys(self, tmp_path):
        csv = _write_catalog(_make_catalog(), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        for key in ("scan_df", "best", "figure_path", "pdf_path"):
            assert key in result

    def test_figure_path_exists(self, tmp_path):
        csv = _write_catalog(_make_catalog(), tmp_path)
        out = tmp_path / "fig.png"
        main(["--csv", str(csv), "--out", str(out)])
        assert out.exists()

    def test_pdf_path_exists(self, tmp_path):
        csv = _write_catalog(_make_catalog(), tmp_path)
        out = tmp_path / "fig.png"
        result = main(["--csv", str(csv), "--out", str(out)])
        assert result["pdf_path"].exists()

    def test_scan_df_is_dataframe(self, tmp_path):
        csv = _write_catalog(_make_catalog(), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        assert isinstance(result["scan_df"], pd.DataFrame)

    def test_best_is_dict(self, tmp_path):
        csv = _write_catalog(_make_catalog(), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        assert isinstance(result["best"], dict)

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "missing.csv")])

    def test_missing_column_raises(self, tmp_path):
        df = _make_catalog().drop(columns=["slope_tail", "delta_f3"])
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            main(["--csv", str(p)])

    def test_custom_m_start_stop(self, tmp_path):
        csv = _write_catalog(_make_catalog(n=60), tmp_path)
        result = main([
            "--csv", str(csv),
            "--out", str(tmp_path / "fig.png"),
            "--m-start", "10.2",
            "--m-stop", "11.0",
            "--m-step", "0.2",
        ])
        if not result["scan_df"].empty:
            assert result["scan_df"]["m_cut"].min() >= 10.2
            assert result["scan_df"]["m_cut"].max() < 11.0

    def test_best_m_cut_in_scan_range(self, tmp_path):
        csv = _write_catalog(_make_catalog(n=60), tmp_path)
        result = main(["--csv", str(csv), "--out", str(tmp_path / "fig.png")])
        best_m = result["best"]["m_cut"]
        assert best_m in result["scan_df"]["m_cut"].values

    def test_default_csv_arg_is_absolute(self):
        args = _parse_args([])
        assert Path(args.csv).is_absolute()

    def test_default_csv_points_to_sparc_subset(self):
        args = _parse_args([])
        assert "sparc_subset.csv" in args.csv

    def test_default_out_is_png(self):
        args = _parse_args([])
        assert args.out.endswith(".png")

    def test_beta_ref_is_half(self):
        assert BETA_REF == pytest.approx(0.5)

    def test_n_min_default(self):
        assert N_MIN_DEFAULT == 15


# ---------------------------------------------------------------------------
# 5. Integration: committed SPARC subset fixture (regression guard)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_SPARC_CSV = _REPO_ROOT / "data" / "sparc_subset.csv"

# Regression values computed from the committed 79-galaxy CSV.
_EXPECTED_BEST_M_CUT = pytest.approx(10.05, abs=1e-9)
_EXPECTED_BEST_RHO = pytest.approx(-0.4803, abs=0.001)
_EXPECTED_BEST_P = pytest.approx(1.794e-4, abs=1e-5)
_EXPECTED_BEST_N = 56
_EXPECTED_N_ROWS = 21   # number of scan rows with N > 15


class TestSPARCSubsetIntegration:
    """Regression guard: run the full pipeline on the committed SPARC CSV and
    verify known numerical results."""

    def test_fixture_exists(self):
        assert _SPARC_CSV.exists(), (
            f"SPARC fixture not found: {_SPARC_CSV}"
        )

    def test_best_m_cut(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["best"]["m_cut"] == _EXPECTED_BEST_M_CUT

    def test_best_rho(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["best"]["rho"] == _EXPECTED_BEST_RHO

    def test_best_p(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["best"]["p"] == _EXPECTED_BEST_P

    def test_best_n(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert int(result["best"]["N"]) == _EXPECTED_BEST_N

    def test_scan_row_count(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert len(result["scan_df"]) == _EXPECTED_N_ROWS

    def test_all_scan_rho_negative(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert (result["scan_df"]["rho"] < 0).all()

    def test_all_scan_p_significant(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert (result["scan_df"]["p"] < 0.05).all()

    def test_figure_files_saved(self, tmp_path):
        out = tmp_path / "sparc_mass_scan.png"
        main(["--csv", str(_SPARC_CSV), "--out", str(out)])
        assert out.exists()
        assert out.with_suffix(".pdf").exists()

    def test_best_score_is_max_in_scan(self, tmp_path):
        result = main(["--csv", str(_SPARC_CSV), "--out", str(tmp_path / "fig.png")])
        assert result["best"]["score"] == pytest.approx(
            result["scan_df"]["score"].max(), rel=1e-9
        )
