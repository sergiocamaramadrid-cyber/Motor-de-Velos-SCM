from scripts.stress_test_framework_scm import (
    STATUS_FRAMEWORK_READY,
    STATUS_FUTURE_EXTENSION,
    STATUS_OUT_OF_DOMAIN,
    run_stress_test,
)


def _status_for(df, name: str) -> str:
    matched = df.loc[df["object"] == name]
    if matched.empty:
        raise AssertionError(f"Object not found in stress-test output: {name}")
    row = matched.iloc[0]
    return str(row["framework_status"])


def test_cloud_9_is_out_of_domain():
    df = run_stress_test()
    assert _status_for(df, "Cloud-9") == STATUS_OUT_OF_DOMAIN


def test_aquarius_iii_is_out_of_domain():
    df = run_stress_test()
    assert _status_for(df, "Aquarius III") == STATUS_OUT_OF_DOMAIN


def test_platypus_is_future_extension_candidate():
    df = run_stress_test()
    assert _status_for(df, "Platypus") == STATUS_FUTURE_EXTENSION


def test_ngc_2403_is_framework_ready():
    df = run_stress_test()
    assert _status_for(df, "NGC 2403") == STATUS_FRAMEWORK_READY
