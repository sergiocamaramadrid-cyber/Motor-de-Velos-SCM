from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
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


def test_sparc_sample_respects_required_contract_columns() -> None:
    sample = Path("data/sparc_175_master_sample.csv")
    if not sample.exists():
        sample = Path("results/SPARC/sparc_175_master_sample.csv")
    assert sample.exists(), "No SPARC sample CSV found"

    df = pd.read_csv(sample)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing required contract columns: {sorted(missing)}"
