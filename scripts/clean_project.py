#!/usr/bin/env python3
"""
clean_project.py

Limpieza segura del proyecto SCM (modo producción).

Funciones:
- Elimina artefactos generados y caches
- Protege rutas críticas (datos, código, git)
- Ofrece modo estándar y modo completo (--full)

Uso:
    python scripts/clean_project.py
    python scripts/clean_project.py --full
"""

import argparse
import shutil
from pathlib import Path


# 🔒 Rutas que NUNCA deben eliminarse
RUTAS_PROTEGIDAS = [
    Path("data/SPARC"),
    Path("data/big_sparc"),
    Path("src"),
    Path("scripts"),
    Path("tests"),
    Path(".git"),
]


# 🧹 Objetivos de limpieza estándar
OBJETIVOS_ESTANDAR = [
    Path("results"),
    Path("outputs"),
    Path("logs"),
    Path(".pytest_cache"),
    Path(".mypy_cache"),
]


# 🔥 Objetivos extra para modo completo
OBJETIVOS_FULL = [
    Path("results/SPARC"),
    Path("results/oos_validation"),
    Path("results/scm_results_final"),
]


def es_ruta_protegida(ruta: Path) -> bool:
    """Comprueba si la ruta está dentro de una zona protegida."""
    try:
        ruta_resuelta = ruta.resolve()
    except Exception:
        return False

    for protegida in RUTAS_PROTEGIDAS:
        try:
            if ruta_resuelta.is_relative_to(protegida.resolve()):
                return True
        except Exception:
            continue
    return False


def eliminar_seguro(ruta: Path) -> None:
    """Elimina archivo o carpeta solo si no está protegida."""
    if not ruta.exists():
        return

    if es_ruta_protegida(ruta):
        print(f"⛔ PROTEGIDO: {ruta}")
        return

    try:
        if ruta.is_file() or ruta.is_symlink():
            ruta.unlink()
            print(f"🗑️ Archivo eliminado: {ruta}")
        else:
            shutil.rmtree(ruta)
            print(f"🗑️ Carpeta eliminada: {ruta}")
    except Exception as e:
        print(f"⚠️ Error eliminando {ruta}: {e}")


def limpiar_artefactos_python() -> None:
    """Elimina __pycache__ y archivos .pyc en todo el proyecto."""
    for p in Path(".").rglob("__pycache__"):
        eliminar_seguro(p)

    for p in Path(".").rglob("*.pyc"):
        eliminar_seguro(p)


def limpieza_estandar() -> None:
    print("\n🧹 Ejecutando limpieza estándar...\n")

    for objetivo in OBJETIVOS_ESTANDAR:
        eliminar_seguro(objetivo)

    limpiar_artefactos_python()

    print("\n✅ Limpieza estándar completada\n")


def limpieza_full() -> None:
    print("\n🔥 Ejecutando limpieza COMPLETA...\n")

    for objetivo in OBJETIVOS_FULL:
        eliminar_seguro(objetivo)

    limpieza_estandar()

    print("\n✅ Limpieza COMPLETA finalizada\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpieza segura del proyecto SCM")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ejecuta limpieza completa (incluye resultados SCM)",
    )
    args = parser.parse_args()

    if args.full:
        limpieza_full()
    else:
        limpieza_estandar()


if __name__ == "__main__":
    main()
