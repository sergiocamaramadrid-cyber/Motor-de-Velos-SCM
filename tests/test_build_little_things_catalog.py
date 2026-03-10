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
                "Name Cl Dist VMag Rd Rad logSFR1 MHI [O/H] PA b/a i _RA _DE Unused",
                "DDO1 Im 3.1 -14.2 0.8 1.9 -2.1 8.2 7.9 35 0.65 41 12.5 -8.3 999",
            ]
        ),
        encoding="utf-8",
    )

    written = build_catalog(input_txt, output_csv)

    assert written == output_csv
    assert output_csv.exists()

    out = pd.read_csv(output_csv)
    assert list(out.columns) == [
        "galaxy",
        "morphology",
        "distance_mpc",
        "abs_mag_v",
        "disk_scale_kpc",
        "holmberg_radius_arcmin",
        "log_sfr_ha",
        "log_mhi",
        "metallicity_12logOH",
        "position_angle_deg",
        "axis_ratio",
        "inclination_deg",
        "ra_deg",
        "dec_deg",
    ]
    assert out.loc[0, "galaxy"] == "DDO1"
