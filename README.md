# SCM-BH — Regime transition in AGN jet collimation

This repository contains a reproducible empirical analysis of AGN jet opening angles using the MOJAVE catalogue.

## Main result

Using the MOJAVE catalogue (VizieR J/MNRAS/468/4992), we identify a statistically robust regime separation around:

```text
r15 ≈ 10
```

Results from the full canonical sample:

```text
N total = 360
LOW  (r15 < 10)  = 274
HIGH (r15 ≥ 10) = 86

KS p-value ≈ 5.6 × 10^-16
Spearman rho in HIGH ≈ -0.331
Spearman p-value ≈ 0.0019
```

## Interpretation

The analysis shows:

- a statistically significant LOW/HIGH regime separation;
- progressive collimation within the HIGH regime;
- no assumed physical mechanism.

The threshold r15 = 10 was identified empirically and tested for stability under nearby cuts.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/run_analysis.py
```

Expected output:

```text
TOTAL: 360
LOW: 274
HIGH: 86
KS p: ~5.6e-16
rho: ~-0.331
p: ~0.0019
```

## Data

The source data are public and available from:

- MOJAVE catalogue, VizieR J/MNRAS/468/4992.

The analysis uses the table containing:

- `r15`
- `alphaApp15`

## Citation

If you use this repository, please cite the associated release DOI and the MOJAVE catalogue.
