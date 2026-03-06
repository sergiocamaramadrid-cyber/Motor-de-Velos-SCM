from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_sparc_full_catalog import (
    KPC_TO_M,
    UPSILON_BULGE,
    UPSILON_DISK,
    _find_existing_master_table,
    _find_existing_rotmod_files,
    add_master_derived_columns,
    norm_name,
    process_rotmod,
)


def test_norm_name_normalizes_spaces_and_case():
    assert norm_name(" ngc 2403 ") == "NGC2403"


def test_find_existing_rotmod_files_searches_data_paths(tmp_path):
    data_root = tmp_path / "data"
    repo_root = tmp_path / "repo"
    (data_root / "Rotmod_LTG" / "nested").mkdir(parents=True)
    (repo_root / "data" / "SPARC" / "raw").mkdir(parents=True)
    (data_root / "Rotmod_LTG" / "nested" / "A_rotmod.dat").write_text("1 2 3 4 5\n")
    (repo_root / "data" / "SPARC" / "raw" / "B_rotmod.dat").write_text("1 2 3 4 5\n")
    found = _find_existing_rotmod_files(data_root, repo_root, rot_dir=data_root / "Rotmod_LTG")
    names = [p.name for p in found]
    assert names == ["A_rotmod.dat", "B_rotmod.dat"]


def test_find_existing_master_table_prefers_existing_paths(tmp_path):
    data_root = tmp_path / "data"
    repo_root = tmp_path / "repo"
    data_root.mkdir(parents=True)
    (repo_root / "data").mkdir(parents=True)
    table = repo_root / "data" / "SPARC_Lelli2016c.mrt"
    table.write_text("header\n")
    found = _find_existing_master_table(data_root, repo_root)
    assert found == table.resolve()


def test_add_master_derived_columns_builds_log_columns():
    master = pd.DataFrame(
        [
            {"Galaxy": "NGC2403", "L_3.6": 2.0, "MHI": 1.0, "RHI": 10.0},
        ]
    )
    out = add_master_derived_columns(master)
    expected_mbar_1e9 = UPSILON_DISK * 2.0 + 1.33 * 1.0
    assert np.isclose(out.loc[0, "logMbar"], np.log10(expected_mbar_1e9) + 9.0)
    sigma = (1.0e9) / (np.pi * (10.0**2) * 1e6)
    assert np.isclose(out.loc[0, "logSigmaHI_out"], np.log10(sigma))


def test_process_rotmod_converts_to_si_and_joins_master_values(tmp_path):
    rotmod_path = tmp_path / "ngc2403_rotmod.dat"
    # r, vobs, err, vgas, vdisk, vbul
    np.savetxt(
        rotmod_path,
        np.array(
            [
                [1.0, 100.0, 2.0, 40.0, 60.0, 10.0],
                [2.0, 120.0, 3.0, 45.0, 70.0, 12.0],
            ]
        ),
    )
    params = {"NGC2403": {"logMbar": 10.2, "logSigmaHI_out": 0.1}}
    out = process_rotmod(rotmod_path, params)
    assert list(out.columns) == ["galaxy", "r_kpc", "g_obs", "g_bar", "logMbar", "logSigmaHI_out"]
    assert len(out) == 2
    expected_g_obs = ((100.0 * 1_000.0) ** 2) / (1.0 * KPC_TO_M)
    expected_g_bar = (
        ((40.0 * 1_000.0) ** 2)
        + (((60.0 * np.sqrt(UPSILON_DISK)) * 1_000.0) ** 2)
        + (((10.0 * np.sqrt(UPSILON_BULGE)) * 1_000.0) ** 2)
    ) / (1.0 * KPC_TO_M)
    assert np.isclose(out.loc[0, "g_obs"], expected_g_obs)
    assert np.isclose(out.loc[0, "g_bar"], expected_g_bar)
    assert np.all(out["logMbar"] == 10.2)
    assert np.all(out["logSigmaHI_out"] == 0.1)
