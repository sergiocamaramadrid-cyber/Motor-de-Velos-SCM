# scripts/setup_validate_windows.ps1
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_validate_windows.ps1
# Or (recommended from repo root):
#   .\scripts\setup_validate_windows.ps1

$ErrorActionPreference = "Stop"

function Info($msg) { Write-Host "[INFO] $msg" }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Die($msg)  { throw $msg }

# ---------
# Config (edit if needed)
# ---------
$VenvDir   = ".venv"
$DataDir   = "data\SPARC"
$Results   = "results"
$SensOut   = "results\sensitivity"
$Notebook  = "notebooks\SPARC_validation.ipynb"
$NbOut     = "results\SPARC_validation_executed.ipynb"

# Dev tools: installed if missing from requirements.txt
$DevTools = @("pytest", "flake8", "mypy", "jupyter", "nbconvert")

# ---------
# 0) Repo root guard
# ---------
if (-not (Test-Path "src")) { Die "No veo carpeta 'src' aquí. Ejecuta este script desde la raíz del repo." }

# ---------
# 1) venv
# ---------
Info "Creating venv: $VenvDir"
python -m venv $VenvDir

$activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) { Die "No encuentro el activador del venv: $activate" }

# Execution policy workaround (only for this process)
try {
  Info "Activating venv"
  . $activate
} catch {
  Warn "Falló la activación del venv (posible ExecutionPolicy). Intento Bypass en el proceso..."
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
  . $activate
}

# ---------
# 2) deps
# ---------
Info "Upgrading pip"
python -m pip install --upgrade pip

if (-not (Test-Path "requirements.txt")) {
  Warn "No existe requirements.txt. Instalo mínimos."
  python -m pip install numpy pandas scipy scikit-learn matplotlib
} else {
  Info "Installing requirements.txt"
  python -m pip install -r requirements.txt
}

# Ensure dev tools exist (safe even if already installed)
Info "Installing dev tools (safe if already present): $($DevTools -join ', ')"
python -m pip install $DevTools

# ---------
# 3) SPARC dir guard + env var
# ---------
if (-not (Test-Path $DataDir)) {
  Die "Falta '$DataDir'. Crea '$DataDir\raw\' y '$DataDir\processed\' o ajusta `$DataDir en este script."
}

$env:SPARC_DATA_DIR = (Resolve-Path $DataDir).Path
Info "SPARC_DATA_DIR=$env:SPARC_DATA_DIR"

# ---------
# 4) outputs
# ---------
Info "Ensuring output dirs"
New-Item -ItemType Directory -Force -Path $Results, $SensOut | Out-Null

# ---------
# 5) Run analysis
# ---------
Info "Running scm_analysis"
python -m src.scm_analysis --data-dir $DataDir --out "$Results\"
if ($LASTEXITCODE -ne 0) { Die "scm_analysis failed (exit=$LASTEXITCODE)" }

Info "Running sensitivity"
python -m src.sensitivity --data-dir $DataDir --out "$SensOut\"
if ($LASTEXITCODE -ne 0) { Die "sensitivity failed (exit=$LASTEXITCODE)" }

# ---------
# 6) Quality checks
# ---------
Info "pytest"
pytest -q
if ($LASTEXITCODE -ne 0) { Die "pytest failed (exit=$LASTEXITCODE)" }

Info "flake8"
flake8 src\ tests\
if ($LASTEXITCODE -ne 0) { Die "flake8 failed (exit=$LASTEXITCODE)" }

Info "mypy"
mypy src\
if ($LASTEXITCODE -ne 0) { Die "mypy failed (exit=$LASTEXITCODE)" }

# ---------
# 7) Notebook batch execute
# ---------
if (Test-Path $Notebook) {
  Info "nbconvert execute: $Notebook"
  jupyter nbconvert --to notebook --execute $Notebook `
    --ExecutePreprocessor.timeout=600 `
    --output $NbOut
  if ($LASTEXITCODE -ne 0) { Die "nbconvert failed (exit=$LASTEXITCODE)" }
} else {
  Warn "No encuentro notebook '$Notebook'. Skip nbconvert."
}

Write-Host "✔ Todo ejecutado correctamente." -ForegroundColor Green
