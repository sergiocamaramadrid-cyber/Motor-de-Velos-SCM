#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROTECTED = [
    Path("data/SPARC"),
    Path("data/big_sparc"),
    Path("src"),
    Path("scripts"),
    Path("tests"),
    Path(".git"),
]


CLEAN_TARGETS = [
    Path("results"),
    Path("outputs"),
    Path("logs"),
    Path("__pycache__"),
    Path(".pytest_cache"),
    Path(".mypy_cache"),
]


def is_protected(path: Path) -> bool:
    path_resolved = path.resolve()
    for protected_path in PROTECTED:
        try:
            path_resolved.relative_to(protected_path.resolve())
            return True
        except ValueError:
            continue
    return False


def safe_remove(path: Path) -> None:
    if not path.exists():
        return

    if is_protected(path):
        print(f"⛔ PROTEGIDO: {path}")
        return

    if path.is_file():
        path.unlink()
        print(f"🗑️ Archivo eliminado: {path}")
    else:
        shutil.rmtree(path)
        print(f"🗑️ Carpeta eliminada: {path}")


def clean_standard() -> None:
    print("\n🧹 Limpieza estándar...\n")

    for target in CLEAN_TARGETS:
        safe_remove(target)

    for path in Path(".").rglob("__pycache__"):
        safe_remove(path)

    for path in Path(".").rglob("*.pyc"):
        safe_remove(path)

    print("\n✅ Limpieza estándar completada\n")


def clean_full() -> None:
    print("\n🔥 Limpieza profunda (FULL)...\n")

    extra_targets = [
        Path("results/SPARC"),
        Path("results/oos_validation"),
        Path("results/scm_results_final"),
    ]

    for target in extra_targets:
        safe_remove(target)

    clean_standard()

    print("\n✅ Limpieza FULL completada\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Limpieza profunda")
    args = parser.parse_args()

    if args.full:
        clean_full()
    else:
        clean_standard()


if __name__ == "__main__":
    main()
