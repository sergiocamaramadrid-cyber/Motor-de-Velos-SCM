from pathlib import Path

import pandas as pd


SAMPLE_PATH = Path("results/SPARC/sparc_175_master_sample.csv")


def test_sparc_master_sample_exists_with_f3_columns():
    assert SAMPLE_PATH.exists(), f"Missing sample CSV: {SAMPLE_PATH}"

    df = pd.read_csv(SAMPLE_PATH)
    required = {
        "galaxy",
        "logSigmaHI_out",
        "logMbar",
        "logRd",
        "F3",
        "f3_scm",
        "delta_f3",
        "fit_ok",
        "quality_flag",
        "beta",
        "beta_err",
        "reliable",
        "friction_slope",
        "velo_inerte_flag",
    }
    assert required.issubset(df.columns)
    assert len(df) > 0
