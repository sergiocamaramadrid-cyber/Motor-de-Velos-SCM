"""
tests/test_plot_little_things_final.py — Tests for scripts/plot_little_things_final.py.

Covers:
  - DELTA_MASS_STD / DELTA_F3 data constants: shape, N=26
  - compute_stats: rho sign/range, p-value range, n, OLS slope sign
  - generate_figure: returns Figure, saves PNG, axes labels/title
  - main: smoke test, returns stats dict, writes output file
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

plt_lt = importlib.import_module("plot_little_things_final")

DELTA_MASS_STD = plt_lt.DELTA_MASS_STD
DELTA_F3 = plt_lt.DELTA_F3
compute_stats = plt_lt.compute_stats
generate_figure = plt_lt.generate_figure
main = plt_lt.main


# ---------------------------------------------------------------------------
# Data constant tests
# ---------------------------------------------------------------------------

def test_data_length():
    assert len(DELTA_MASS_STD) == 26
    assert len(DELTA_F3) == 26


def test_data_shapes_match():
    assert DELTA_MASS_STD.shape == DELTA_F3.shape


def test_delta_f3_all_negative():
    """All outer-slope residuals in this sample are negative."""
    assert np.all(DELTA_F3 < 0)


def test_delta_mass_std_range():
    """Standardised environmental proxy should span roughly [-3, +2]."""
    assert DELTA_MASS_STD.min() < -2.0
    assert DELTA_MASS_STD.max() > 1.0


# ---------------------------------------------------------------------------
# compute_stats tests
# ---------------------------------------------------------------------------

def test_compute_stats_keys():
    stats = compute_stats(DELTA_MASS_STD, DELTA_F3)
    for key in ("rho", "p_val", "ols_slope", "ols_intercept", "n"):
        assert key in stats


def test_compute_stats_n():
    stats = compute_stats(DELTA_MASS_STD, DELTA_F3)
    assert stats["n"] == 26


def test_compute_stats_rho_negative():
    """Spearman rho should be negative (higher mass → more negative delta_f3)."""
    stats = compute_stats(DELTA_MASS_STD, DELTA_F3)
    assert stats["rho"] < 0


def test_compute_stats_rho_range():
    stats = compute_stats(DELTA_MASS_STD, DELTA_F3)
    assert -1.0 <= stats["rho"] <= 1.0


def test_compute_stats_p_val_range():
    stats = compute_stats(DELTA_MASS_STD, DELTA_F3)
    assert 0.0 <= stats["p_val"] <= 1.0


def test_compute_stats_ols_slope_negative():
    stats = compute_stats(DELTA_MASS_STD, DELTA_F3)
    assert stats["ols_slope"] < 0


def test_compute_stats_known_values():
    """Verify rho ≈ -0.37 and p ≈ 0.060 against pre-computed reference."""
    stats = compute_stats(DELTA_MASS_STD, DELTA_F3)
    assert abs(stats["rho"] - (-0.3734)) < 0.005
    assert abs(stats["p_val"] - 0.0603) < 0.005


def test_compute_stats_custom_arrays():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    stats = compute_stats(x, y)
    assert stats["rho"] == pytest.approx(1.0)
    assert stats["n"] == 4


def test_stats_regression_guard():
    """Protect against silent changes to the hardcoded dataset."""
    stats = compute_stats(DELTA_MASS_STD, DELTA_F3)
    assert abs(stats["rho"] + 0.37) < 0.02


# ---------------------------------------------------------------------------
# generate_figure tests
# ---------------------------------------------------------------------------

def test_generate_figure_returns_figure(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "test_fig.png"
    fig = generate_figure(DELTA_MASS_STD, DELTA_F3, out_path=out)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_generate_figure_saves_file(tmp_path):
    out = tmp_path / "LITTLE_THINGS_final.png"
    import matplotlib.pyplot as plt
    fig = generate_figure(DELTA_MASS_STD, DELTA_F3, out_path=out)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_figure_axes_labels(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = generate_figure(DELTA_MASS_STD, DELTA_F3, out_path=out)
    ax = fig.axes[0]
    assert "delta" in ax.get_xlabel().lower() or "δ" in ax.get_xlabel()
    assert "f3" in ax.get_ylabel().lower() or "δ" in ax.get_ylabel()
    plt.close(fig)


def test_generate_figure_title(tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = generate_figure(DELTA_MASS_STD, DELTA_F3, out_path=out)
    ax = fig.axes[0]
    assert "26" in ax.get_title() or "little things" in ax.get_title().lower()
    plt.close(fig)


def test_generate_figure_saves_pdf(tmp_path):
    """generate_figure must also save a sibling PDF."""
    import matplotlib.pyplot as plt
    out = tmp_path / "fig.png"
    fig = generate_figure(DELTA_MASS_STD, DELTA_F3, out_path=out)
    plt.close(fig)
    pdf = tmp_path / "fig.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 0



    import matplotlib.pyplot as plt
    out = tmp_path / "subdir" / "deep" / "fig.png"
    fig = generate_figure(DELTA_MASS_STD, DELTA_F3, out_path=out)
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


def test_main_writes_file(tmp_path):
    out = tmp_path / "LITTLE_THINGS_final.png"
    main(["--out", str(out)])
    assert out.exists()


def test_main_writes_pdf(tmp_path):
    out = tmp_path / "figure01_env_little_things.png"
    main(["--out", str(out)])
    assert (tmp_path / "figure01_env_little_things.pdf").exists()


def test_main_out_path_in_result(tmp_path):
    out = tmp_path / "custom.png"
    result = main(["--out", str(out)])
    assert result["out_path"] == str(out)
