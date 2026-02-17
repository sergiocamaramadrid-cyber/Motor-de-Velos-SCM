# Data Directory

This directory should contain the SPARC dataset.

## Structure

Recommended structure:
```
data/
├── SPARC/
│   ├── raw/           # Raw downloaded data
│   └── processed/     # Processed data files
└── README.md          # This file
```

## Download Instructions

[Instructions for downloading the SPARC dataset would go here]

For reproducibility, you can set an environment variable:

```bash
# Unix/macOS
export SPARC_DATA_DIR=$(pwd)/data/SPARC

# Windows PowerShell
$env:SPARC_DATA_DIR = (Resolve-Path .\data\SPARC).Path
```
