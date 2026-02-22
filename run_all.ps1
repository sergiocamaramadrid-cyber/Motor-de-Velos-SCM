$ErrorActionPreference = "Stop"

# (0) Opcional: evita problemas de activación de venv por policy
# Solo si te falla Activate.ps1, descomenta la siguiente línea:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# (1) Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# (2) Actualizar pip e instalar dependencias del proyecto
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (2b) Herramientas de desarrollo (por si no están en requirements.txt)
python -m pip install pytest flake8 mypy jupyter nbconvert

# (3) Comprobar que el directorio de datos SPARC existe
if (-not (Test-Path "data\SPARC")) {
    throw "Falta data\SPARC\. Crea data\SPARC\raw\ y data\SPARC\processed\ o apunta a la ruta correcta."
}
$env:SPARC_DATA_DIR = (Resolve-Path .\data\SPARC).Path

# (4) Crear directorios de salida (evita fallos si no existen)
New-Item -ItemType Directory -Force -Path "results", "results\sensitivity" | Out-Null

# (5) Ejecutar análisis principal
python -m src.scm_analysis --data-dir data\SPARC --out results\

# (6) Ejecutar análisis de sensibilidad
python -m src.sensitivity --data-dir data\SPARC --out results\sensitivity\

# (7) Tests y calidad de código
pytest -q
flake8 src\ tests\
mypy src\

# (8) Ejecutar notebook en modo batch
jupyter nbconvert --to notebook --execute notebooks\SPARC_validation.ipynb `
    --ExecutePreprocessor.timeout=600 `
    --output results\SPARC_validation_executed.ipynb

Write-Host "✔ Todo ejecutado correctamente."
