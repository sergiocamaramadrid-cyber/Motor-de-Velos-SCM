# SCM-BH — Regime transition in AGN jet collimation

This repository contains a reproducible empirical analysis of AGN jet opening angles using the MOJAVE catalogue.

## Main result

Using the MOJAVE catalogue (VizieR J/MNRAS/468/4992), we identify a statistically robust regime separation around:

```text
r15 ≈ 10
```

Results from the full canonical sample (real VizieR data):

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

## Data

> **The paper results are obtained from the public MOJAVE VizieR table J/MNRAS/468/4992/table3.**
> **Synthetic example data are provided only for testing script execution and do not reproduce the published statistics.**

The real data are public and freely available at:

- <https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/MNRAS/468/4992/table3>

The analysis uses the columns:

- `r15` — brightness-temperature ratio
- `alphaApp15` — apparent jet opening angle (deg)

A synthetic example file (`data/mojave_vizier_table3_synthetic_example.csv`) is included **solely** to allow testing that the script runs without errors. It must not be used to reproduce or validate any scientific result.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

### With real VizieR data (auto-download)

The script attempts to download the real table automatically:

```bash
python scripts/run_analysis.py
```

### With a locally saved real CSV

If you have downloaded `table3` from VizieR yourself:

```bash
python scripts/run_analysis.py --data path/to/real_table3.csv
```

Expected output (real data only):

```text
TOTAL: 360
LOW: 274
HIGH: 86
KS p: ~5.6e-16
rho: ~-0.331
p: ~0.0019
```

### Synthetic example (testing only)

If no network and no real CSV are available, the script falls back to the synthetic example and prints a prominent warning. **These results do not reproduce the paper.**

## Citation

If you use this repository, please cite the associated release DOI and the MOJAVE catalogue:

- Lister et al. (2017), MNRAS 468, 4992 — <https://doi.org/10.1093/mnras/stx677>
