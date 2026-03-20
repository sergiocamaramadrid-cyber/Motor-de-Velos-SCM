#!/usr/bin/env python3
"""
ignition_sparc_real.py

Pipeline de arranque completo para SPARC (modo botón rojo).

Hace:
1. Limpieza del entorno (opcional)
2. Validación del ZIP SPARC
3. Extracción limpia a data/SPARC/rotmod
4. Precheck de número de galaxias (esperado: 175)
5. Construcción opcional del catálogo SPARC completo
6. Generación opcional del catálogo F3 desde contrato
7. Ejecución opcional de comandos adicionales
8. Generación de resumen JSON reproducible

Uso:
    python scripts/ignition_sparc_real.py CURVAS_SPARC.zip --clean --overwrite
    python scripts/ignition_sparc_real.py CURVAS_SPARC.zip --clean-full --overwrite
    python scripts/ignition_sparc_real.py CURVAS_SPARC.zip --clean --overwrite --build-catalog --generate-f3
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROTMOD_DIR = Path("data/SPARC/rotmod")
RESULTS_DIR = Path("results")


def ejecutar(cmd: list[str]) -> None:
    print(f"\n🚀 Ejecutando: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def run_clean(mode: str = "standard") -> None:
    print("\n🧹 Limpieza del entorno...\n")
    if mode == "full":
        ejecutar([sys.executable, "scripts/clean_project.py", "--full"])
    else:
        ejecutar([sys.executable, "scripts/clean_project.py"])


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

    ejecutar(cmd)


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
        ejecutar(shlex.split(cmd))


def build_catalog() -> None:
    print("\n🧱 Construcción del catálogo SPARC completo...\n")
    ejecutar([sys.executable, "scripts/build_sparc_full_catalog.py"])


def generate_f3() -> None:
    print("\n📈 Generación del catálogo F3 desde contrato...\n")
    ejecutar([sys.executable, "scripts/generate_f3_catalog_from_contract.py"])


def save_summary(
    n_galaxies: int,
    zip_path: Path,
    build_catalog_done: bool = False,
    generate_f3_done: bool = False,
    extra_commands: list[str] | None = None,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = {
        "zip_source": str(zip_path),
        "galaxies_detected": n_galaxies,
        "build_catalog_executed": build_catalog_done,
        "generate_f3_executed": generate_f3_done,
        "extra_commands": extra_commands or [],
        "status": "ok" if n_galaxies == 175 else "warning",
    }

    out = RESULTS_DIR / "ignition_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Resumen guardado en: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ignición completa del dataset SPARC")

    parser.add_argument("zipfile", help="ZIP del catálogo SPARC")
    parser.add_argument("--clean", action="store_true", help="Limpieza estándar previa")
    parser.add_argument("--clean-full", action="store_true", help="Limpieza completa previa")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir datos existentes")
    parser.add_argument(
        "--build-catalog",
        action="store_true",
        help="Ejecuta scripts/build_sparc_full_catalog.py tras la ingesta",
    )
    parser.add_argument(
        "--generate-f3",
        action="store_true",
        help="Ejecuta scripts/generate_f3_catalog_from_contract.py tras construir el catálogo",
    )
    parser.add_argument(
        "--run",
        action="append",
        help="Comandos adicionales a ejecutar (se puede repetir)",
    )

    args = parser.parse_args()

    zip_path = Path(args.zipfile)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP file not found: {zip_path}. "
            "Please ensure the CURVAS_SPARC.zip file exists in the specified location."
        )

    if args.clean_full:
        run_clean("full")
    elif args.clean:
        run_clean("standard")

    run_ingestion(zip_path, overwrite=args.overwrite)

    n = precheck()

    if args.build_catalog:
        build_catalog()

    if args.generate_f3:
        generate_f3()

    if args.run:
        run_extra(args.run)

    save_summary(
        n_galaxies=n,
        zip_path=zip_path,
        build_catalog_done=args.build_catalog,
        generate_f3_done=args.generate_f3,
        extra_commands=args.run,
    )

    print("\n🎯 IGNICIÓN COMPLETADA\n")


if __name__ == "__main__":
    main()
