import json
from argparse import Namespace
from pathlib import Path

from scripts import ignition_sparc_real


def test_rotmod_dir_path_is_expected():
    assert str(ignition_sparc_real.ROTMOD_DIR) == "data/SPARC/rotmod"


def test_run_extra_executes_shell_split_commands(monkeypatch):
    calls = []

    def fake_ejecutar(cmd):
        calls.append(cmd)

    monkeypatch.setattr(ignition_sparc_real, "ejecutar", fake_ejecutar)

    ignition_sparc_real.run_extra(
        [
            "python scripts/intra_galaxy_gradient_test.py",
            "python scripts/another.py --flag value",
        ]
    )

    assert calls == [
        ["python", "scripts/intra_galaxy_gradient_test.py"],
        ["python", "scripts/another.py", "--flag", "value"],
    ]


def test_save_summary_persists_extra_run_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(ignition_sparc_real, "RESULTS_DIR", tmp_path / "results")

    args = Namespace(
        clean=False,
        clean_full=False,
        build_catalog=True,
        generate_f3=True,
        run=["python scripts/intra_galaxy_gradient_test.py"],
    )

    ignition_sparc_real.save_summary(175, Path("CURVAS_SPARC.zip"), args)

    out = (tmp_path / "results" / "ignition_summary.json")
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["extra_commands"] == ["python scripts/intra_galaxy_gradient_test.py"]
    assert payload["status"] == "ok"
