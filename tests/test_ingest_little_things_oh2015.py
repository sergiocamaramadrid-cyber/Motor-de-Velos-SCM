from pathlib import Path

from scripts.ingest_little_things_oh2015 import read_table1, read_table2, read_rotdmbar


def test_parsers_return_expected_columns(tmp_path: Path):
    table1 = tmp_path / "table1.dat"
    table2 = tmp_path / "table2.dat"
    rotdmbar = tmp_path / "rotdmbar.dat"

    table1.write_text(
        "DDO 47  08 04 36.9 +16 46 11.5  5.2  273.0 1.2 315.0 2.0 37.0 1.0 -14.7 7.80 0.10 -2.50 0.03 -2.10 0.02\n",
        encoding="utf-8",
    )
    table2.write_text(
        "DDO 47   4.00 1.20  50.0  48.0 12.0 0.20 10.0  1.0 12.0  80.0   2.0  75.0  1.5 1.20 0.10   20.0  1.00 -0.30 0.05  -0.25 0.05   5.00  4.00  4.50 8.50 10.20\n",
        encoding="utf-8",
    )
    rotdmbar.write_text(
        "DDO 47  Data  1.200000 50.000000 0.500000 0.600000 0.050000\n"
        "DDO 47  Model 1.200000 50.000000 0.600000 0.700000 0.020000\n",
        encoding="utf-8",
    )

    df1 = read_table1(table1)
    df2 = read_table2(table2)
    dfr = read_rotdmbar(rotdmbar)

    assert "galaxy" in df1.columns
    assert "rmax_kpc" in df2.columns
    assert "r_scaled" in dfr.columns
    assert set(dfr["data_type"].unique()) == {"Data"}
