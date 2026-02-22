# Setup + Validación completa (Windows / PowerShell)
# Ejecutar desde la raíz del repositorio Motor-de-Velos-SCM

$ErrorActionPreference = "Stop"

# 1️⃣ Clonar (solo la primera vez)
# git clone <url-del-repo>
# cd Motor-de-Velos-SCM

# 2️⃣ Entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3️⃣ Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ Variable de entorno SPARC
$env:SPARC_DATA_DIR = (Resolve-Path .\data\SPARC).Path

# 5️⃣ Análisis principal
python -m src.scm_analysis --data-dir data\SPARC --out results\
python -m src.sensitivity --data-dir data\SPARC --out results\sensitivity\

# 6️⃣ Tests y calidad
pytest -q
flake8 src\ tests\
mypy src\

# 7️⃣ Ejecutar notebook en modo batch (validación notarial)
jupyter nbconvert --to notebook --execute notebooks\SPARC_validation.ipynb `
  --ExecutePreprocessor.timeout=600 `
  --output results\SPARC_validation_executed.ipynb

Write-Host "✔ Todo ejecutado correctamente."
