"""
tests/test_plot_little_things_highmass.py — Tests for
scripts/plot_little_things_highmass.py.

Covers:
  - DELTA_MASS_STD_HM / DELTA_F3_HM data constants: shape, N=13
  - LOGM_CUT fixed threshold
  - compute_stats: rho sign/range, p-value range, n, OLS slope sign
  - generate_figure: returns Figure, saves PNG + PDF, axes labels/title
  - main: smoke test, returns stats dict, writes output files
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

plt_hm = importlib.import_module("plot_little_things_highmass")

DELTA_MASS_STD_HM = plt_hm.DELTA_MASS_STD_HM
DELTA_F3_HM = plt_hm.DELTA_F3_HM
LOGM_CUT = plt_hm.LOGM_CUT
compute_stats = plt_hm.compute_stats
generate_figure = plt_hm.generate_figure
main = plt_hm.main


# ---------------------------------------------------------------------------
# Data constant tests
# ---------------------------------------------------------------------------

def test_data_length():
    assert len(DELTA_MASS_STD_HM) == 13
    assert len(DELTA_F3_HM) == 13


def test_data_shapes_match():
    assert DELTA_MASS_STD_HM.shape == DELTA_F3_HM.shape


def test_delta_f3_all_negative():
    """All outer-slope residuals in this subsample are negative."""
    assert np.all(DELTA_F3_HM < 0)


def test_delta_mass_std_all_positive():
    """High-mass galaxies are all above-average in environmental proxy."""
    assert np.all(DELTA_MASS_STD_HM > 0)


def test_logm_cut_value():
    """Fixed threshold must be exactly 7.8 as documented."""
    assert LOGM_CUT == pytest.approx(7.8)


# ---------------------------------------------------------------------------
# compute_stats tests
# ---------------------------------------------------------------------------

def test_compute_stats_keys():
    stats = compute_stats(DELTA_MASS_STD_HM, DELTA_F3_HM)
    for key in ("rho", "p_val", "ols_slope", "ols_intercept", "n"):
        assert key in stats


def test_compute_stats_n():
    stats = compute_stats(DELTA_MASS_STD_HM, DELTA_F3_HM)
    assert stats["n"] == 13


def test_compute_stats_rho_negative():
    """Spearman rho should be negative for the high-mass subsample."""
    stats = compute_stats(DELTA_MASS_STD_HM, DELTA_F3_HM)
    assert stats["rho"] < 0


def test_compute_stats_rho_range():
    stats = compute_stats(DELTA_MASS_STD_HM, DELTA_F3_HM)
    assert -1.0 <= stats["rho"] <= 1.0


def test_compute_stats_p_val_range():
    stats = compute_stats(DELTA_MASS_STD_HM, DELTA_F3_HM)
    assert 0.0 <= stats["p_val"] <= 1.0


def test_compute_stats_ols_slope_negative():
    stats = compute_stats(DELTA_MASS_STD_HM, DELTA_F3_HM)
    assert stats["ols_slope"] < 0


def test_compute_stats_known_values():
    """Verify rho ≈ -0.454 and p ≈ 0.119 (Spearman on the 13 hardcoded points)."""
    stats = compute_stats(DELTA_MASS_STD_HM, DELTA_F3_HM)
    assert abs(stats["rho"] - (-0.454)) < 0.005
    assert abs(stats["p_val"] - 0.119) < 0.005


def test_stats_regression_guard():
    """Protect against silent changes to the hardcoded high-mass dataset."""
    stats = compute_stats(DELTA_MASS_STD_HM, DELTA_F3_HM)
    assert abs(stats["rho"] + 0.454) < 0.005


def test_compute_stats_custom_arrays():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    stats = compute_stats(x, y)
    assert stats["rho"] == pytest.approx(1.0)
    assert stats["n"] == 4


# ---------------------------------------------------------------------------
# generate_figure tests
# ---------------------------------------------------------------------------

def test_generate_figure_returns_figure(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "test_fig.png"
    fig = generate_figure(DELTA_MASS_STD_HM, DELTA_F3_HM, out_path=out)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_generate_figure_saves_png(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = generate_figure(DELTA_MASS_STD_HM, DELTA_F3_HM, out_path=out)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_figure_saves_pdf(tmp_path):
    """generate_figure must also save a sibling PDF."""
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = generate_figure(DELTA_MASS_STD_HM, DELTA_F3_HM, out_path=out)
    plt.close(fig)
    pdf = tmp_path / "fig.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 0


def test_generate_figure_axes_labels(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = generate_figure(DELTA_MASS_STD_HM, DELTA_F3_HM, out_path=out)
    ax = fig.axes[0]
    assert "delta" in ax.get_xlabel().lower() or "δ" in ax.get_xlabel()
    assert "f3" in ax.get_ylabel().lower() or "δ" in ax.get_ylabel()
    plt.close(fig)


def test_generate_figure_title_contains_n(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = generate_figure(DELTA_MASS_STD_HM, DELTA_F3_HM, out_path=out)
    ax = fig.axes[0]
    assert "13" in ax.get_title()
    plt.close(fig)


def test_generate_figure_title_mentions_highmass(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = generate_figure(DELTA_MASS_STD_HM, DELTA_F3_HM, out_path=out)
    ax = fig.axes[0]
    title_lower = ax.get_title().lower()
    assert "high" in title_lower or "mass" in title_lower
    plt.close(fig)


def test_generate_figure_creates_parent_dirs(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "subdir" / "deep" / "fig.png"
    fig = generate_figure(DELTA_MASS_STD_HM, DELTA_F3_HM, out_path=out)
    plt.close(fig)
    assert out.exists()


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------

def test_main_returns_dict(tmp_path):
    out = tmp_path / "out.png"
    result = main(["--out", str(out)])
    assert isinstance(result, dict)


def test_main_stats_keys(tmp_path):
    out = tmp_path / "out.png"
    result = main(["--out", str(out)])
    for key in ("rho", "p_val", "ols_slope", "ols_intercept", "n", "out_path", "pdf_path"):
        assert key in result


def test_main_writes_png(tmp_path):
    out = tmp_path / "figure02_env_little_things_highmass.png"
    main(["--out", str(out)])
    assert out.exists()


def test_main_writes_pdf(tmp_path):
    out = tmp_path / "figure02_env_little_things_highmass.png"
    main(["--out", str(out)])
    assert (tmp_path / "figure02_env_little_things_highmass.pdf").exists()


def test_main_out_path_in_result(tmp_path):
    out = tmp_path / "custom.png"
    result = main(["--out", str(out)])
    assert result["out_path"] == str(out)


def test_main_n_is_13(tmp_path):
    out = tmp_path / "out.png"
    result = main(["--out", str(out)])
    assert result["n"] == 13
