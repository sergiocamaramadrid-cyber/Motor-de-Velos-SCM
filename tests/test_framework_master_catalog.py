from pathlib import Path

import pandas as pd


def test_master_catalog_columns():
    path = Path("results/combined/framework_master_catalog.csv")

    if not path.exists():
        return

    df = pd.read_csv(path)

    expected = {
        "galaxy",
        "source_catalog",
        "framework_stage",
        "science_role",
        "dist_mpc",
        "incl_deg",
        "rmax_kpc",
        "r03_kpc",
        "v_rmax_kms",
        "mgas_1e7_msun",
        "mstar_proxy_1e7_msun",
        "logmdyn",
        "alphamin",
        "rotcurve_available",
    }

    assert expected.issubset(set(df.columns))
