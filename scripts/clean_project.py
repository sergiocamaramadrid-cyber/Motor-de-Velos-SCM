#!/usr/bin/env python3
"""
clean_project.py

Production-safe cleaner for SCM project.

- Removes generated artifacts and caches
- Protects critical paths (data, code, git)
- Provides standard and full cleanup modes

Usage:
    python scripts/clean_project.py
    python scripts/clean_project.py --full
"""

import argparse
import shutil
from pathlib import Path


# 🔒 Paths that must NEVER be deleted
PROTECTED_PATHS = [
    Path("data/SPARC"),
    Path("data/big_sparc"),
    Path("src"),
    Path("scripts"),
    Path("tests"),
    Path(".git"),
]


# 🧹 Standard cleanup targets
STANDARD_TARGETS = [
    Path("results"),
    Path("outputs"),
    Path("logs"),
    Path(".pytest_cache"),
    Path(".mypy_cache"),
]


# 🔥 Extra targets for --full mode
FULL_EXTRA_TARGETS = [
    Path("results/SPARC"),
    Path("results/oos_validation"),
    Path("results/scm_results_final"),
]


def is_protected(path: Path) -> bool:
    """Check if path is inside a protected directory."""
    try:
        resolved = path.resolve()
    except Exception:
        return False

    for protected in PROTECTED_PATHS:
        try:
            if resolved.is_relative_to(protected.resolve()):
                return True
        except Exception:
            continue
    return False


def safe_remove(path: Path) -> None:
    """Safely remove file or directory unless protected."""
    if not path.exists():
        return

    if is_protected(path):
        print(f"⛔ PROTECTED: {path}")
        return

    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
            print(f"🗑️ Removed file: {path}")
        else:
            shutil.rmtree(path)
            print(f"🗑️ Removed directory: {path}")
    except Exception as e:
        print(f"⚠️ Failed to remove {path}: {e}")


def remove_python_artifacts() -> None:
    """Remove __pycache__ and *.pyc recursively."""
    for p in Path(".").rglob("__pycache__"):
        safe_remove(p)

    for p in Path(".").rglob("*.pyc"):
        safe_remove(p)


def clean_standard() -> None:
    print("\n🧹 Running standard cleanup...\n")

    for target in STANDARD_TARGETS:
        safe_remove(target)

    remove_python_artifacts()

    print("\n✅ Standard cleanup completed\n")


def clean_full() -> None:
    print("\n🔥 Running FULL cleanup...\n")

    for target in FULL_EXTRA_TARGETS:
        safe_remove(target)

    clean_standard()

    print("\n✅ FULL cleanup completed\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCM project cleaner")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full cleanup (includes extended SCM results)",
    )
    args = parser.parse_args()

    if args.full:
        clean_full()
    else:
        clean_standard()


if __name__ == "__main__":
    main()
