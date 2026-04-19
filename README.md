# SCM – Spectral Environmental Modulation

This repository contains a spectral-level validation of environmental effects in galaxy dynamics using SPARC rotation curves.

## Result

Spearman correlation between environment and mass-controlled residual spectral power:

- ρ = −0.135
- p = 5×10⁻⁴
- N = 656 (logM ≥ 10)

## Method

- FFT decomposition of rotation curves
- Spectral power extraction
- Linear mass control (OLS)
- Residual correlation with environment

## Reproducibility

Run:

```bash
python scripts/spectral_analysis.py --csv spectral_dataset_final.csv
```

## Contents

- `spectral_dataset_final.csv`
- `spectral_summary.json`
- `figure_env_residual_spectral.png`
- `figure_env_residual_spectral.pdf`
- `scripts/spectral_analysis.py`

## Statement

Environmental modulation is not captured by global scaling relations but appears in the internal spectral structure of galaxy rotation curves.
