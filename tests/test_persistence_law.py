import numpy as np
import pandas as pd
import pytest

from scripts.experimental.persistence_law import build_bins, compute_aicc, fit_parameters, recurrence, simulate_sequence


def test_recurrence_stability():
    r = 0.5
    lam = 0.1
    beta = 0.5

    for _ in range(10):
        r = recurrence(r, lam, beta)

    assert 0 < r < 1


def test_fit_parameters_recovers_signal():
    r_obs = simulate_sequence(r0=0.6, lam=0.08, beta=0.5, n=8)
    lam, beta, mse = fit_parameters(r_obs)

    assert mse < 1e-8
    assert lam >= 0
    assert beta > 0


def test_compute_aicc_handles_small_sample():
    assert np.isinf(compute_aicc(n=3, rss=1.0, k=2))


def test_build_bins_accepts_logmbar():
    df = pd.DataFrame(
        {
            "logMbar": np.linspace(8.0, 10.0, 10),
            "g_obs": np.linspace(1.0e-12, 2.0e-12, 10),
            "g_bar": np.linspace(0.5e-12, 1.5e-12, 10),
        }
    )
    out = build_bins(df, n_bins=4)
    assert out.size > 0
    assert np.all(np.isfinite(out))


def test_build_bins_requires_mass_column():
    df = pd.DataFrame({"g_obs": [1.0], "g_bar": [1.0]})
    with pytest.raises(ValueError, match="r_kpc"):
        build_bins(df, n_bins=2)


def test_build_bins_accepts_scale_column():
    df = pd.DataFrame(
        {
            "r_kpc": np.linspace(0.5, 5.0, 10),
            "g_obs": np.linspace(1.0e-12, 2.0e-12, 10),
            "g_bar": np.linspace(0.5e-12, 1.5e-12, 10),
        }
    )
    out = build_bins(df, n_bins=4)
    assert out.size > 0
    assert np.all(np.isfinite(out))
