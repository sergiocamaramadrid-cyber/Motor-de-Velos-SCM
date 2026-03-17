import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.test_rg_proxy import analyze_rg_proxy
from scripts.test_rg_proxy import compute_rg_proxy_for_galaxy
from scripts.test_rg_proxy import slope_from_df
from scripts.test_rg_proxy import split_outer_part


def test_split_outer_part_simple():
    r = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    v = np.array([50, 55, 60, 65, 70, 75])
    inner, outer = split_outer_part(r, v, r_threshold=0.5)
    assert inner is not None
    assert outer is not None
    assert len(inner) == 2
    assert len(outer) == 2
    assert inner["logr"].max() < outer["logr"].min()


def test_split_outer_part_insufficient():
    r = np.array([1.0, 2.0, 3.0])
    v = np.array([10.0, 20.0, 30.0])
    inner, outer = split_outer_part(r, v, r_threshold=0.5)
    assert inner is None
    assert outer is None


def test_slope_from_df_perfect_line():
    x = np.array([0.0, 0.3, 0.6, 0.9])
    y = 2.0 + 0.5 * x
    df = pd.DataFrame({"logr": x, "logv": y})
    slope, err, r = slope_from_df(df)
    assert slope == pytest.approx(0.5, rel=1e-6)
    assert err < 1e-6
    assert r == pytest.approx(1.0, rel=1e-6)


def test_compute_rg_proxy_sign():
    r = np.linspace(0.5, 10.0, 20)
    v = np.zeros_like(r)
    for i, ri in enumerate(r):
        if ri < 7.0:
            v[i] = 50.0 + 5.0 * ri
        else:
            v[i] = 85.0 - 0.2 * ri

    out = compute_rg_proxy_for_galaxy(r, v, r_threshold=0.7)
    assert np.isfinite(out["proxy_rg"])
    assert out["proxy_rg"] < 0
    assert out["proxy_rg_err"] > 0


def test_analyze_rg_proxy_integration_csv(tmp_path: Path):
    n_gal = 6
    pts_rows = []
    meta_rows = []
    rng = np.random.default_rng(123)

    for i in range(n_gal):
        gal = f"G{i:02d}"
        r = np.linspace(1.0, 10.0, 20)

        amp = 3.0 + 0.3 * i
        v = np.where(r < 7.0, 60.0 + amp * r, 82.0 - 0.15 * r + 0.1 * i)

        for rr, vv in zip(r, v):
            pts_rows.append({"galaxy": gal, "r_kpc": rr, "v_obs_kms": vv})

        meta_rows.append(
            {
                "galaxy": gal,
                "logMbar": rng.uniform(8.0, 10.0),
                "logRd": rng.uniform(0.0, 1.0),
                "logSigmaHI_out": rng.uniform(-0.5, 1.5),
            }
        )

    pts = pd.DataFrame(pts_rows)
    meta = pd.DataFrame(meta_rows)

    points_file = tmp_path / "points.csv"
    galaxies_file = tmp_path / "galaxies.csv"
    out_dir = tmp_path / "rg_out"

    pts.to_csv(points_file, index=False)
    meta.to_csv(galaxies_file, index=False)

    summary = analyze_rg_proxy(
        points_file=points_file,
        galaxies_file=galaxies_file,
        out_dir=out_dir,
        r_threshold=0.7,
        bootstrap_n=100,
        seed=123,
    )

    assert summary["status"] == "ok"
    assert summary["n_galaxies"] == n_gal
    assert "coef_logSigmaHI_out" in summary
    assert "bootstrap_ci_logSigmaHI_out" in summary

    per_gal = pd.read_csv(out_dir / "rg_per_galaxy.csv")
    assert len(per_gal) == n_gal
    assert (out_dir / "rg_summary.json").exists()

    with open(out_dir / "rg_summary.json", "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["n_galaxies"] == n_gal
