#!/usr/bin/env python3
"""
verify_and_extract_sparc_rotmod.py

Validación y extracción segura del dataset SPARC rotmod.

Funciones:
- Verifica estructura de archivos *_rotmod.dat
- Detecta duplicados (idénticos vs conflictivos)
- Comprueba formato numérico (8 columnas)
- Extrae dataset limpio a data/SPARC/rotmod
- Genera informe JSON reproducible

Uso:
    python scripts/verify_and_extract_sparc_rotmod.py CURVAS_SPARC.zip --extract --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


EXPECTED_COLUMNS = 8
OUTPUT_DIR = Path("data/SPARC/rotmod")


def file_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def validate_rotmod(content_bytes: bytes) -> bool:
    try:
        text = content_bytes.decode("utf-8", errors="ignore")
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]

        for line in lines:
            parts = line.split()
            if len(parts) != EXPECTED_COLUMNS:
                return False
            for part in parts:
                float(part)

        return True
    except Exception:
        return False


def process_zip(zip_path: Path) -> tuple[dict, dict[str, str]]:
    report = {
        "zip_file": str(zip_path),
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "unique_galaxies": set(),
        "duplicates": {},
        "conflicts": [],
    }

    seen: dict[str, str] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith("_rotmod.dat"):
                continue

            report["total_files"] += 1
            content = zf.read(name)

            if not validate_rotmod(content):
                report["invalid_files"] += 1
                continue

            report["valid_files"] += 1
            galaxy = Path(name).name.replace("_rotmod.dat", "")
            content_hash = file_hash(content)

            if galaxy not in seen:
                seen[galaxy] = content_hash
                report["unique_galaxies"].add(galaxy)
            elif seen[galaxy] == content_hash:
                report["duplicates"].setdefault(galaxy, 0)
                report["duplicates"][galaxy] += 1
            else:
                report["conflicts"].append(galaxy)

    report["unique_galaxies"] = len(report["unique_galaxies"])
    return report, seen


def extract_clean(zip_path: Path, seen_hashes: dict[str, str], overwrite: bool = False) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith("_rotmod.dat"):
                continue

            galaxy_file = Path(name).name
            galaxy_name = galaxy_file.replace("_rotmod.dat", "")
            content = zf.read(name)
            content_hash = file_hash(content)

            if seen_hashes.get(galaxy_name) != content_hash:
                continue

            out_path = OUTPUT_DIR / galaxy_file
            if out_path.exists() and not overwrite:
                continue

            with out_path.open("wb") as f:
                f.write(content)

            extracted += 1

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zipfile", help="Ruta al ZIP SPARC")
    parser.add_argument("--extract", action="store_true", help="Extraer archivos válidos")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir existentes")
    args = parser.parse_args()

    zip_path = Path(args.zipfile)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    report, seen_hashes = process_zip(zip_path)

    print("\n=== VALIDACIÓN SPARC ===")
    for key, value in report.items():
        if key not in ["duplicates", "conflicts"]:
            print(f"{key}: {value}")

    print(f"duplicates: {len(report['duplicates'])}")
    print(f"conflicts: {len(report['conflicts'])}")

    if report["conflicts"]:
        print("⚠️ Conflictos detectados:", report["conflicts"])

    if args.extract:
        extracted = extract_clean(zip_path, seen_hashes, overwrite=args.overwrite)
        print(f"\n✅ Archivos extraídos: {extracted}")

    report_path = Path("results/sparc_validation_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
