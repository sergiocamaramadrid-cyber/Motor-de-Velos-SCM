from __future__ import annotations

import pandas as pd

from scripts.enrich_sparc_with_coordinates import enrich_with_coordinates


def test_enrich_with_coordinates_merges_by_normalized_name(tmp_path):
    input_csv = tmp_path / "rotation_curves-v1.0.csv"
    metadata_csv = tmp_path / "SPARC_Lelli2016c.csv"
    output_csv = tmp_path / "rotation_curves-v1.1-coords.csv"

    pd.DataFrame(
        [
            {"galaxy": " ngc2403 ", "r": 1.0, "g_obs": 1.0e-10, "g_bar": 8.0e-11},
            {"galaxy": "DDO 154", "r": 1.2, "g_obs": 9.0e-11, "g_bar": 7.0e-11},
        ]
    ).to_csv(input_csv, index=False)

    pd.DataFrame(
        [
            {"Name": "NGC2403", "RA": 114.2, "Dec": 65.6, "D": 3.2, "Type": "Scd"},
            {"Name": "DDO154", "RA": 193.5, "Dec": 27.1, "D": 4.3, "Type": "Irr"},
        ]
    ).to_csv(metadata_csv, index=False)

    out = enrich_with_coordinates(
        input_file=input_csv,
        metadata_file=metadata_csv,
        output_file=output_csv,
    )

    assert output_csv.exists()
    assert {"RA", "Dec", "D", "Type"}.issubset(set(out.columns))
    assert out["RA"].notna().sum() == 2
    assert out["D"].tolist() == [3.2, 4.3]


def test_enrich_with_coordinates_accepts_galaxy_and_t_columns(tmp_path):
    input_csv = tmp_path / "rotation_curves-v1.0.csv"
    metadata_csv = tmp_path / "SPARC_Lelli2016c.csv"
    output_csv = tmp_path / "rotation_curves-v1.1-coords.csv"

    pd.DataFrame([{"galaxy": "IC 2574", "r": 1.0, "g_obs": 1.0e-10, "g_bar": 8.0e-11}]).to_csv(
        input_csv, index=False
    )

    pd.DataFrame([{"Galaxy": "IC2574", "RA": 156.9, "DEC": 68.4, "D": 4.0, "T": 9.0}]).to_csv(
        metadata_csv, index=False
    )

    out = enrich_with_coordinates(
        input_file=input_csv,
        metadata_file=metadata_csv,
        output_file=output_csv,
    )

    assert output_csv.exists()
    assert out.loc[0, "RA"] == 156.9
    assert out.loc[0, "Dec"] == 68.4
    assert out.loc[0, "D"] == 4.0
    assert out.loc[0, "Type"] == 9.0


def test_enrich_with_coordinates_accepts_radeg_dedeg_dist_columns(tmp_path):
    input_csv = tmp_path / "rotation_curves-v1.0.csv"
    metadata_csv = tmp_path / "SPARC_Lelli2016c.csv"
    output_csv = tmp_path / "rotation_curves-v1.1-coords.csv"

    pd.DataFrame([{"galaxy": "NGC2403", "r": 1.0, "g_obs": 1.0e-10, "g_bar": 8.0e-11}]).to_csv(
        input_csv, index=False
    )

    pd.DataFrame([{"Name": "NGC2403", "RAdeg": 114.2, "DEdeg": 65.6, "Dist": 3.2}]).to_csv(
        metadata_csv, index=False
    )

    out = enrich_with_coordinates(
        input_file=input_csv,
        metadata_file=metadata_csv,
        output_file=output_csv,
    )

    assert output_csv.exists()
    assert out.loc[0, "RA"] == 114.2
    assert out.loc[0, "Dec"] == 65.6
    assert out.loc[0, "D"] == 3.2
