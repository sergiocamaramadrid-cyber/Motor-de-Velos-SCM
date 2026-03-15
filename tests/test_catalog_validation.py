from __future__ import annotations

import pandas as pd

from scripts.validate_sparc_catalog import validate_catalog


def test_validate_catalog_reports_basic_issues() -> None:
    df = pd.DataFrame(
        {
            "galaxy": ["G1", "G1"],
            "logSigmaHI_out": [1.0, None],
            "logMbar": [9.5, 9.7],
            "logRd": [0.5, 0.6],
            "F3": [0.5, 0.6],
            "f3_scm": [0.5, 0.6],
            "delta_f3": [0.0, 0.1],
            "fit_ok": [False, True],
            "quality_flag": ["ok", "ok"],
            "beta": [0.5, 0.6],
            "beta_err": [0.1, 0.2],
            "reliable": [True, True],
            "friction_slope": [0.5, 0.6],
            "velo_inerte_flag": [True, True],
        }
    )

    report = validate_catalog(df)
    issue_types = set(report["issue_type"].tolist())
    assert "nan_value" in issue_types
    assert "duplicate" in issue_types
    assert "inconsistent_flags" in issue_types
