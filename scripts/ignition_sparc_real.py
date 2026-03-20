#!/usr/bin/env python3
"""
ignition_sparc_real.py

Pipeline de arranque completo para SPARC (modo botón rojo).

Hace:
1. Limpieza del entorno (opcional)
2. Validación del ZIP SPARC
3. Extracción limpia a data/SPARC/rotmod
4. Precheck de número de galaxias (esperado: 175)
5. Ejecución opcional de comandos adicionales
6. Generación de resumen JSON reproducible

Uso:
    python scripts/ignition_sparc_real.py CURVAS_SPARC.zip --clean --overwrite
    python scripts/ignition_sparc_real.py CURVAS_SPARC.zip --clean-full --overwrite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROTMOD_DIR = Path("data/SPARC/rotmod")
RESULTS_DIR = Path("results")


def run_clean(mode: str = "standard") -> None:
    print("\n🧹 Limpieza del entorno...\n")
    if mode == "full":
        subprocess.run([sys.executable, "scripts/clean_project.py", "--full"], check=True)
    else:
        subprocess.run([sys.executable, "scripts/clean_project.py"], check=True)


def run_ingestion(zip_path: Path, overwrite: bool = False) -> None:
    print("\n📦 Validación + extracción SPARC...\n")

    cmd = [
        sys.executable,
        "scripts/verify_and_extract_sparc_rotmod.py",
        str(zip_path),
        "--extract",
    ]

    if overwrite:
        cmd.append("--overwrite")

    subprocess.run(cmd, check=True)


def precheck() -> int:
    print("\n🔍 Precheck de galaxias...\n")

    files = list(ROTMOD_DIR.glob("*_rotmod.dat"))
    n = len(files)

    print(f"Galaxias detectadas: {n}")

    if n != 175:
        print("⚠️ ATENCIÓN: número distinto de 175")
    else:
        print("✅ Dataset completo (175 galaxias)")

    return n


def run_extra(commands: list[str]) -> None:
    for cmd in commands:
        print(f"\n🚀 Ejecutando: {cmd}\n")
        subprocess.run(cmd, shell=True, check=True)


def save_summary(n_galaxies: int, zip_path: Path) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = {
        "zip_source": str(zip_path),
        "galaxies_detected": n_galaxies,
        "status": "ok" if n_galaxies == 175 else "warning",
    }

    out = RESULTS_DIR / "ignition_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📄 Resumen guardado en: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ignición completa del dataset SPARC")

    parser.add_argument("zipfile", help="ZIP del catálogo SPARC")
    parser.add_argument("--clean", action="store_true", help="Limpieza estándar previa")
    parser.add_argument("--clean-full", action="store_true", help="Limpieza completa previa")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir datos existentes")
    parser.add_argument("--run", action="append", help="Comandos adicionales a ejecutar")

    args = parser.parse_args()

    zip_path = Path(args.zipfile)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    if args.clean_full:
        run_clean("full")
    elif args.clean:
        run_clean("standard")

    run_ingestion(zip_path, overwrite=args.overwrite)

    n = precheck()

    if args.run:
        run_extra(args.run)

    save_summary(n, zip_path)

    print("\n🎯 IGNICIÓN COMPLETADA\n")


if __name__ == "__main__":
    main()
