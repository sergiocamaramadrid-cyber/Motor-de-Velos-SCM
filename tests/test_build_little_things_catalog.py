from pathlib import Path

import pandas as pd

from scripts.build_little_things_catalog import build_catalog


def test_build_catalog_renames_and_filters_columns(tmp_path: Path) -> None:
    input_txt = tmp_path / "Hunter_2012.txt"
    output_csv = tmp_path / "results" / "little_things_catalog.csv"

    input_txt.write_text(
        "\n".join(
            [
                "# synthetic table",
                "Name Cl Dist VMag Rd Rad logSFR1 logSFR2 MHI [O/H] PA b/a i _RA _DE Unused",
                "DDO1 Im 3.1 -14.2 0.8 1.9 -2.1 -1.7 8.2 7.9 35 0.65 41 12.5 -8.3 999",
            ]
        ),
        encoding="utf-8",
    )

    written, merged_out = build_catalog(input_txt, output_csv)

    assert written == output_csv
    assert merged_out is None
    assert output_csv.exists()

    out = pd.read_csv(output_csv)
    required = {
        "galaxy",
        "distance_mpc",
        "MHI",
        "logSFR",
        "inclination",
        "Rd",
        "morphology",
        "log_mhi",
        "log_sfr_ha",
        "log_sfr_uv",
    }
    assert required.issubset(set(out.columns))
    assert out.loc[0, "galaxy"] == "DDO1"


def test_build_catalog_merges_with_pipeline(tmp_path: Path) -> None:
    input_txt = tmp_path / "Hunter_2012.txt"
    output_csv = tmp_path / "results" / "little_things_catalog.csv"
    pipeline_csv = tmp_path / "little_things_global.csv"
    pipeline_output = tmp_path / "results" / "little_things_global_enriched.csv"

    input_txt.write_text(
        "\n".join(
            [
                "Name Cl Dist VMag Rd Rad logSFR1 logSFR2 MHI [O/H] PA b/a i _RA _DE",
                "DDO43 Im 7.2 -14.2 0.8 1.9 -2.1 -1.7 8.2 7.9 35 0.65 41 12.5 -8.3",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"galaxy_id": "DDO43", "logM": 7.6, "logVobs": 1.47, "log_gbar": -12.1, "log_j": 1.62},
            {"galaxy_id": "OTHER", "logM": 6.0, "logVobs": 1.10, "log_gbar": -12.5, "log_j": 0.71},
        ]
    ).to_csv(pipeline_csv, index=False)

    written, merged_out = build_catalog(
        input_txt,
        output_csv,
        pipeline_csv=pipeline_csv,
        pipeline_output_csv=pipeline_output,
    )

    assert written == output_csv
    assert merged_out == pipeline_output
    assert pipeline_output.exists()

    merged = pd.read_csv(pipeline_output)
    assert {"galaxy", "distance_mpc", "MHI", "logSFR", "inclination", "Rd"}.issubset(merged.columns)
    ddo43 = merged.loc[merged["galaxy_id"] == "DDO43"].iloc[0]
    assert ddo43["galaxy"] == "DDO43"
    assert pd.notna(ddo43["distance_mpc"])
