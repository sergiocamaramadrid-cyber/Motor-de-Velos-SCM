# SPARC Dataset

Place the SPARC dataset files here before running the analysis pipeline.

## Expected layout

```
data/SPARC/
├── raw/          ← original SPARC downloads
└── processed/    ← pre-processed files consumed by src/
```

## Download instructions

1. Visit the SPARC portal: <https://sparc.science>
2. Download the dataset for your study.
3. Extract files into `data/SPARC/raw/`.

## Environment variable

Define `SPARC_DATA_DIR` to point to this directory (optional but recommended):

```powershell
# Windows PowerShell
$env:SPARC_DATA_DIR = (Resolve-Path .\data\SPARC).Path
```

```bash
# Unix/macOS
export SPARC_DATA_DIR=$(pwd)/data/SPARC
```
