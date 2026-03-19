#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path


def clean_project() -> None:
    results_dir = Path("results")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, _ in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(Path(root) / d)

    print("🧹 Limpieza completada.")


if __name__ == "__main__":
    clean_project()
