from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.prepare_big_sparc_catalog import (
    _GRAVITATIONAL_CONSTANT,
    _KG_M2_TO_MSUN_PC2,
    _KILOPARSEC_TO_METERS,
    _SOLAR_MASS_KG,
    prepare_catalog,
    prepare_catalog_from_sparc_dir,
)


def test_prepare_catalog_passthrough_when_g_columns_exist(tmp_path):
    inp = tmp_path / "input.csv"
    out = tmp_path / "big_sparc_catalog.csv"
    pd.DataFrame(
        [
            {"galaxy": "A", "g_obs": 2.0e-11, "g_bar": 1.1e-11, "logMbar": 10.2},
            {"galaxy": "A", "g_obs": 1.7e-11, "g_bar": 0.9e-11, "logMbar": 10.2},
            {"galaxy": "B", "g_obs": 1.4e-11, "g_bar": 0.8e-11, "logMbar": 9.9},
        ]
    ).to_csv(inp, index=False)

    df = prepare_catalog(inp, out)
    assert out.exists()
    assert list(df.columns) == ["galaxy", "g_obs", "g_bar", "logMbar"]
    assert len(df) == 3


def test_prepare_catalog_converts_contract_columns_to_acceleration(tmp_path):
    inp = tmp_path / "contract.csv"
    out = tmp_path / "big_sparc_catalog.csv"
    pd.DataFrame(
        [
            {"galaxy": "G1", "r_kpc": 1.0, "vobs_kms": 100.0, "vbar_kms": 80.0},
            {"galaxy": "G1", "r_kpc": 2.0, "vobs_kms": 120.0, "vbar_kms": 95.0},
            {"galaxy": "G2", "r_kpc": 1.5, "vobs_kms": 90.0, "vbar_kms": 70.0},
        ]
    ).to_csv(inp, index=False)

    df = prepare_catalog(inp, out)
    assert out.exists()
    assert list(df.columns) == ["galaxy", "g_obs", "g_bar"]
    assert len(df) == 3
    assert np.all(df["g_obs"] > 0.0)
    assert np.all(df["g_bar"] > 0.0)
    expected_g_obs_first = ((100.0 * 1_000.0) ** 2) / (1.0 * _KILOPARSEC_TO_METERS)
    expected_g_bar_first = ((80.0 * 1_000.0) ** 2) / (1.0 * _KILOPARSEC_TO_METERS)
    assert np.isclose(df.iloc[0]["g_obs"], expected_g_obs_first)
    assert np.isclose(df.iloc[0]["g_bar"], expected_g_bar_first)


def test_prepare_catalog_from_sparc_dir_reads_rotmod_and_derives_env_columns(tmp_path):
    sparc_dir = tmp_path / "SPARC"
    raw_dir = sparc_dir / "raw"
    raw_dir.mkdir(parents=True)
    out = tmp_path / "big_sparc_catalog.csv"

    # 6-column SPARC-like format: r, vobs, err, vgas, vdisk, vbul
    pd.DataFrame(
        [
            [1.0, 100.0, 3.0, 45.0, 70.0, 5.0],
            [2.0, 120.0, 4.0, 48.0, 76.0, 6.0],
            [3.0, 130.0, 5.0, 50.0, 80.0, 8.0],
        ]
    ).to_csv(raw_dir / "NGC2403_rotmod.dat", sep=" ", index=False, header=False)

    # 5-column simplified format: r, vobs, vgas, vdisk, vbul
    pd.DataFrame(
        [
            [1.2, 90.0, 35.0, 60.0, 0.0],
            [2.2, 95.0, 38.0, 62.0, 0.0],
            [3.2, 98.0, 39.0, 65.0, 0.0],
        ]
    ).to_csv(raw_dir / "UGC0128_rotmod.dat", sep=" ", index=False, header=False)

    df = prepare_catalog_from_sparc_dir(sparc_dir, out, galaxies=["NGC2403", "UGC0128"])

    assert out.exists()
    assert list(df.columns) == ["galaxy", "g_obs", "g_bar", "logMbar", "logSigmaHI_out"]
    assert set(df["galaxy"]) == {"NGC2403", "UGC0128"}
    assert np.all(df["g_obs"] > 0.0)
    assert np.all(df["g_bar"] > 0.0)
    # one environmental value per galaxy, repeated per row
    per_galaxy = df.groupby("galaxy")[["logMbar", "logSigmaHI_out"]].nunique()
    assert np.all(per_galaxy["logMbar"] == 1)
    assert np.all(per_galaxy["logSigmaHI_out"] == 1)

    ngc2403 = df[df["galaxy"] == "NGC2403"].iloc[0]
    r_out = 3.0 * _KILOPARSEC_TO_METERS
    vgas_out = 50.0 * 1_000.0
    vdisk_out = 80.0 * 1_000.0
    vbul_out = 8.0 * 1_000.0
    gbar_out = (vgas_out**2 + vdisk_out**2 + vbul_out**2) / r_out
    mbar_kg = gbar_out * (r_out**2) / _GRAVITATIONAL_CONSTANT
    expected_logmbar = np.log10(mbar_kg / _SOLAR_MASS_KG)
    ggas_out = (vgas_out**2) / r_out
    sigma_kg_m2 = ggas_out / (2.0 * np.pi * _GRAVITATIONAL_CONSTANT)
    expected_logsigma = np.log10(sigma_kg_m2 * _KG_M2_TO_MSUN_PC2)

    assert np.isclose(ngc2403["logMbar"], expected_logmbar)
    assert np.isclose(ngc2403["logSigmaHI_out"], expected_logsigma)
