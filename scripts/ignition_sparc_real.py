#!/usr/bin/env python3
"""
ignition_sparc_real.py

Pipeline de arranque completo para SPARC (modo botón rojo).

Hace:
1. Limpieza del entorno (opcional)
2. Validación del ZIP SPARC
3. Extracción limpia a data/SPARC/rotmod
4. Precheck de número de galaxias (esperado: 175)
5. Construcción opcional del catálogo SPARC
6. Generación opcional del catálogo F3
7. Ejecución opcional de comandos adicionales
8. Generación de resumen JSON reproducible

Uso:
    python scripts/ignition_sparc_real.py CURVAS_SPARC.zip --clean --overwrite
    python scripts/ignition_sparc_real.py CURVAS_SPARC.zip --clean --overwrite --build-catalog --generate-f3
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROTmod_DIR = Path("data/SPARC/rotmod")
RESULTS_DIR = Path("results")


# =========================
# Utilidades
# =========================

def ejecutar(cmd):
    print(f"\n🚀 Ejecutando: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


# =========================
# Bloques pipeline
# =========================

def run_clean(mode="standard"):
    print("\n🧹 Limpieza del entorno...\n")

    if mode == "full":
        ejecutar([sys.executable, "scripts/clean_project.py", "--full"])
    else:
        ejecutar([sys.executable, "scripts/clean_project.py"])


def run_ingestion(zip_path, overwrite=False):
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


def precheck():
    print("\n🔍 Precheck de galaxias...\n")

    files = list(ROTmod_DIR.glob("*_rotmod.dat"))
    n = len(files)

    print(f"Galaxias detectadas: {n}")

    if n != 175:
        print("⚠️ ATENCIÓN: número distinto de 175")
    else:
        print("✅ Dataset completo (175 galaxias)")

    return n


def build_catalog():
    print("\n🧱 Construcción del catálogo SPARC...\n")
    ejecutar([sys.executable, "scripts/build_sparc_full_catalog.py"])


def generate_f3():
    print("\n📈 Generación del catálogo F3...\n")
    ejecutar([sys.executable, "scripts/generate_f3_catalog_from_contract.py"])


def run_extra(commands):
    for cmd in commands:
        ejecutar(shlex.split(cmd))


# =========================
# Resumen reproducible
# =========================

def save_summary(n_galaxies, zip_path, args):
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = {
        "zip_source": str(zip_path),
        "galaxies_detected": n_galaxies,
        "clean_mode": "full" if args.clean_full else ("standard" if args.clean else "none"),
        "build_catalog": args.build_catalog,
        "generate_f3": args.generate_f3,
        "extra_commands": args.run or [],
        "status": "ok" if n_galaxies == 175 else "warning",
    }

    out = RESULTS_DIR / "ignition_summary.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Resumen guardado en: {out}")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Ignición completa del dataset SPARC")

    parser.add_argument("zipfile", help="ZIP del catálogo SPARC")
    parser.add_argument("--clean", action="store_true", help="Limpieza estándar previa")
    parser.add_argument("--clean-full", action="store_true", help="Limpieza completa previa")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir datos existentes")

    parser.add_argument("--build-catalog", action="store_true", help="Construye catálogo SPARC")
    parser.add_argument("--generate-f3", action="store_true", help="Genera catálogo F3")

    parser.add_argument("--run", action="append", help="Comandos extra (repetible)")

    args = parser.parse_args()

    zip_path = Path(args.zipfile)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    # Limpieza
    if args.clean_full:
        run_clean("full")
    elif args.clean:
        run_clean("standard")

    # Ingesta
    run_ingestion(zip_path, overwrite=args.overwrite)

    # Precheck
    n = precheck()

    # Pipeline científico
    if args.build_catalog:
        build_catalog()

    if args.generate_f3:
        generate_f3()

    # Extra
    if args.run:
        run_extra(args.run)

    # Resumen
    save_summary(n, zip_path, args)

    print("\n🎯 IGNICIÓN COMPLETADA\n")


if __name__ == "__main__":
    main()
