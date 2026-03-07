from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_population_master_table import build_population_master_table


def test_build_population_master_table_merges_required_and_optional_columns(tmp_path):
    f3 = tmp_path / "f3_catalog.csv"
    full = tmp_path / "sparc_full.csv"
    master = tmp_path / "sparc_master_catalog.csv"
    out = tmp_path / "population_master.csv"

    pd.DataFrame(
        [
            {
                "galaxy": "G1",
                "delta_f3": 0.01,
                "deep_slope": 0.51,
                "n_tail_points": 5,
                "tail_r_min": 2.0,
                "tail_r_max": 8.0,
            },
            {
                "galaxy": "G2",
                "delta_f3": -0.03,
                "deep_slope": 0.47,
                "n_tail_points": 5,
                "tail_r_min": 1.0,
                "tail_r_max": 6.0,
            },
        ]
    ).to_csv(f3, index=False)

    pd.DataFrame(
        [
            {"galaxy": "G1", "logSigmaHI_out": 0.12},
            {"galaxy": "G1", "logSigmaHI_out": 0.15},
            {"galaxy": "G2", "logSigmaHI_out": -0.05},
        ]
    ).to_csv(full, index=False)

    pd.DataFrame(
        [
            {"Galaxy": "G1", "L_3.6": 3.0, "Rdisk": 2.5, "Inc": 55.0},
            {"Galaxy": "G2", "L_3.6": 1.0, "Rdisk": 1.7, "Inc": 70.0},
        ]
    ).to_csv(master, index=False)

    table = build_population_master_table(
        f3_catalog_path=f3,
        full_catalog_path=full,
        master_catalog_path=master,
        out_path=out,
    )

    assert out.exists()
    assert list(table.columns) == [
        "galaxy",
        "delta_f3",
        "deep_slope",
        "n_tail_points",
        "tail_r_min",
        "tail_r_max",
        "logSigmaHI_out",
        "logMstar",
        "Rdisk",
        "inclination",
    ]
    assert np.isclose(table.loc[0, "logSigmaHI_out"], 0.12)
    assert np.isclose(table.loc[1, "logSigmaHI_out"], -0.05)
    assert np.isclose(table.loc[0, "logMstar"], np.log10(0.5 * 3.0) + 9.0)
    assert np.isclose(table.loc[1, "Rdisk"], 1.7)
    assert np.isclose(table.loc[1, "inclination"], 70.0)


def test_build_population_master_table_keeps_schema_when_optional_sources_missing(tmp_path):
    f3 = tmp_path / "f3_catalog.csv"
    out = tmp_path / "population_master.csv"

    pd.DataFrame(
        [
            {
                "galaxy": "G_ONLY",
                "delta_f3": 0.00,
                "deep_slope": 0.50,
                "n_tail_points": 5,
                "tail_r_min": 2.0,
                "tail_r_max": 9.0,
            }
        ]
    ).to_csv(f3, index=False)

    table = build_population_master_table(
        f3_catalog_path=f3,
        out_path=out,
    )

    assert out.exists()
    assert len(table) == 1
    assert np.isnan(table.loc[0, "logSigmaHI_out"])
    assert np.isnan(table.loc[0, "logMstar"])
    assert np.isnan(table.loc[0, "Rdisk"])
    assert np.isnan(table.loc[0, "inclination"])


def test_build_population_master_table_supports_legacy_f3_without_delta_or_tail_columns(tmp_path):
    f3 = tmp_path / "f3_legacy.csv"
    out = tmp_path / "population_master.csv"

    pd.DataFrame(
        [
            {
                "galaxy": "G_LEG",
                "deep_slope": 0.46,
            }
        ]
    ).to_csv(f3, index=False)

    table = build_population_master_table(
        f3_catalog_path=f3,
        out_path=out,
    )

    assert len(table) == 1
    assert np.isclose(table.loc[0, "delta_f3"], -0.04)
    assert np.isnan(table.loc[0, "n_tail_points"])
