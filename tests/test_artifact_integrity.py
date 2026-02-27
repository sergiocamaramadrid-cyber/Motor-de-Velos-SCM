from __future__ import annotations

import os

from src.audit_integrity import (
    discover_artifacts,
    find_duplicates,
    validate_audit_json,
    validate_reports_csv,
)

REPO_ROOT = "."


def test_audits_validated_json_schema_and_no_duplicates():
    artifacts = discover_artifacts(REPO_ROOT)
    audits = artifacts["audits"]

    # If someone clones without validated audits, we don't fail the whole repo.
    # But in your project you DO have them; so we enforce at least 1.
    assert len(audits) >= 1, "Expected at least 1 JSON in audits/validated/"

    all_errors = []
    for p in audits:
        ok, errors, _key, _h = validate_audit_json(p)
        if not ok:
            all_errors.append((p, errors))

    assert not all_errors, "Audit JSON schema errors:\n" + "\n".join(
        f"- {p}: {errs}" for p, errs in all_errors
    )

    dups = find_duplicates(audits)
    msg = []
    for k, v in dups.items():
        if v:
            msg.append(f"{k}: {v}")
    assert not msg, "Duplicate audit artifacts detected:\n" + "\n".join(msg)


def test_reports_csv_minimal_columns_and_no_duplicate_rows():
    artifacts = discover_artifacts(REPO_ROOT)
    csvs = artifacts["reports_csv"]

    # reports/ may be optional in some setups, so only validate if present.
    if not csvs:
        return

    required = ["galaxia", "xi", "vif_hinge"]
    all_errors = []
    for p in csvs:
        ok, errors = validate_reports_csv(p, required_cols=required)
        if not ok:
            all_errors.append((p, errors))

    assert not all_errors, "Report CSV integrity errors:\n" + "\n".join(
        f"- {os.path.basename(p)}: {errs}" for p, errs in all_errors
    )
