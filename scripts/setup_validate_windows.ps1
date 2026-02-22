$ErrorActionPreference = "Stop"

# --- Paths ---
$venvActivate = ".\.venv\Scripts\Activate.ps1"
$dataDir = "data\SPARC"
$outDir = "results\"
$outSens = "results\sensitivity\"
$nbIn = "notebooks\SPARC_validation.ipynb"
$nbOut = "results\SPARC_validation_executed.ipynb"

# --- 0) Prechecks ---
if (-not (Test-Path $dataDir)) {
  throw "Missing dataset directory: $dataDir (expected SPARC under data\SPARC)"
}

# --- 1) Venv ---
if (-not (Test-Path ".\.venv")) {
  python -m venv .venv
}
if (-not (Test-Path $venvActivate)) {
  throw "Missing venv activate script: $venvActivate"
}
& $venvActivate

# --- 2) Deps ---
python -m pip install --upgrade pip
pip install -r requirements.txt

# Ensure notebook tooling exists (if not pinned in requirements.txt)
pip install -q jupyter nbconvert

# --- 3) Env var ---
$env:SPARC_DATA_DIR = (Resolve-Path .\data\SPARC).Path

# --- 4) Run analysis ---
python -m src.scm_analysis --data-dir $dataDir --out $outDir
python -m src.sensitivity --data-dir $dataDir --out $outSens

# --- 5) QA ---
pytest -q
flake8 src\ tests\
mypy src\

# --- 6) Notebook (headless) ---
jupyter nbconvert --to notebook --execute $nbIn `
  --ExecutePreprocessor.timeout=600 `
  --output $nbOut

Write-Host "✔ Todo ejecutado correctamente."
