#!/usr/bin/env bash
# run_f3_pipeline.sh
# Genera el catálogo F3, valida su integridad y, opcionalmente,
# ejecuta el análisis ambiental β.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_CSV="${SCRIPT_DIR}/data/big_sparc/contract/big_sparc_contract.parquet"
OUTPUT_DIR="${SCRIPT_DIR}/results/SPARC"
OUTPUT_CSV="${OUTPUT_DIR}/f3_catalog.csv"
VALIDATION_COL="F3_SCM"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/scripts/run_big_sparc_veil_test.py"
ANALYSIS_CATALOG=""
FULL_ANALYSIS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full-analysis)
            FULL_ANALYSIS=1
            shift
            ;;
        --input)
            INPUT_CSV="$2"
            shift 2
            ;;
        --out)
            OUTPUT_DIR="$2"
            OUTPUT_CSV="${OUTPUT_DIR}/f3_catalog.csv"
            shift 2
            ;;
        --analysis-catalog)
            ANALYSIS_CATALOG="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}❌ Argumento desconocido: $1${NC}"
            exit 1
            ;;
    esac
done

if [[ -z "${ANALYSIS_CATALOG}" ]]; then
    ANALYSIS_CATALOG="${INPUT_CSV}"
fi

echo -e "${YELLOW}🔧 Iniciando pipeline F3...${NC}"

if [[ ! -f "${INPUT_CSV}" ]]; then
    echo -e "${RED}❌ No se encuentra el archivo de entrada: ${INPUT_CSV}${NC}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo -e "${GREEN}▶️ Generando catálogo F3...${NC}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/generate_f3_catalog_from_contract.py" \
    --input "${INPUT_CSV}" \
    --out "${OUTPUT_DIR}"

if [[ ! -f "${OUTPUT_CSV}" ]]; then
    echo -e "${RED}❌ No se generó el archivo esperado: ${OUTPUT_CSV}${NC}"
    exit 1
fi

echo -e "${GREEN}🔍 Validando catálogo generado...${NC}"

VALIDATION_OUTPUT="$(
"${PYTHON_BIN}" - "${OUTPUT_CSV}" "${VALIDATION_COL}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
preferred_col = sys.argv[2]

df = pd.read_csv(path)
if df.empty:
    print("❌ Error: el catálogo está vacío.")
    raise SystemExit(1)

if preferred_col in df.columns:
    col = preferred_col
elif "deep_slope" in df.columns:
    col = "deep_slope"
else:
    print("❌ Error: no se encontró columna de pendiente (F3_SCM o deep_slope).")
    print("Columnas disponibles:", list(df.columns))
    raise SystemExit(1)

s = df[col].dropna()
if s.empty:
    print("❌ Error: la columna de pendiente no tiene valores válidos.")
    raise SystemExit(1)

std = float(s.std())
print(
    f"✅ [VALIDACIÓN SCM] columna={col} registros={len(df)} válidos={len(s)} "
    f"min={s.min():.6g} max={s.max():.6g} mean={s.mean():.6g} std={std:.6g}"
)

if len(s) < 2 or pd.isna(std):
    print("❌ Error: no hay suficientes valores para estimar la dispersión.")
    raise SystemExit(1)

if std <= 1e-12:
    print("❌ Error: la desviación estándar es esencialmente cero (catálogo degenerado).")
    raise SystemExit(1)

if len(s) < 10:
    print("⚠️ Advertencia: muy pocas galaxias válidas (<10).")

if std < 0.01:
    print("⚠️ Advertencia: dispersión muy baja; el observable puede estar colapsado.")

raise SystemExit(0)
PY
)" || {
    echo "${VALIDATION_OUTPUT}"
    echo -e "${RED}❌ Fallo en la validación del catálogo.${NC}"
    exit 1
}

echo "${VALIDATION_OUTPUT}"
echo -e "${GREEN}✅ Catálogo validado correctamente.${NC}"

if [[ "${FULL_ANALYSIS}" -eq 1 ]]; then
    if [[ ! -f "${ANALYSIS_SCRIPT}" ]]; then
        echo -e "${RED}❌ No se encuentra el script de análisis: ${ANALYSIS_SCRIPT}${NC}"
        exit 1
    fi

    if [[ ! -f "${ANALYSIS_CATALOG}" ]]; then
        echo -e "${RED}❌ No se encuentra el catálogo para análisis: ${ANALYSIS_CATALOG}${NC}"
        exit 1
    fi

    echo -e "${YELLOW}🔬 Ejecutando análisis ambiental...${NC}"
    "${PYTHON_BIN}" "${ANALYSIS_SCRIPT}" --catalog "${ANALYSIS_CATALOG}" --out "${OUTPUT_DIR}"
    echo -e "${GREEN}✅ Análisis ambiental completado.${NC}"
fi

echo -e "${GREEN}✅ Pipeline completado exitosamente.${NC}"
