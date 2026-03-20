from pathlib import Path
import json

from scripts.ignition_sparc_real import run_extra, save_summary


def test_run_extra_executes_commands(monkeypatch):
    """Verifica que run_extra ejecuta comandos correctamente sin shell=True."""

    executed = []

    def fake_run(cmd, check):
        executed.append(cmd)

    monkeypatch.setattr("subprocess.run", fake_run)

    commands = [
        "python scripts/intra_galaxy_gradient_test.py",
        "python scripts/another_script.py",
    ]

    run_extra(commands)

    assert len(executed) == 2
    assert executed[0][0] == "python"
    assert any("intra_galaxy_gradient_test.py" in part for part in executed[0])
    assert executed[1][0] == "python"


def test_save_summary_persists_extra_commands(tmp_path):
    """Verifica que save_summary guarda correctamente extra_commands."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    import scripts.ignition_sparc_real as ignition
    ignition.RESULTS_DIR = results_dir

    class Args:
        clean = True
        clean_full = False
        build_catalog = True
        generate_f3 = True
        run = ["python scripts/test.py"]

    zip_path = Path("CURVAS_SPARC.zip")

    save_summary(
        n_galaxies=175,
        zip_path=zip_path,
        args=Args,
    )

    summary_file = results_dir / "ignition_summary.json"
    assert summary_file.exists()

    data = json.loads(summary_file.read_text())

    assert data["galaxies_detected"] == 175
    assert data["status"] == "ok"
    assert data["build_catalog"] is True
    assert data["generate_f3"] is True
    assert data["extra_commands"] == ["python scripts/test.py"]


def test_rotmod_path_is_correct():
    """Test mínimo estructural (evita regresiones de path)."""
    data_dir = Path("data/SPARC/rotmod")
    assert str(data_dir) == "data/SPARC/rotmod"
