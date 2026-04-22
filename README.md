# SCM Framework — Environmental Regime Transition

## Overview

This repository contains the final environmental analysis of the **SCM (Slope Coupling Model)** framework applied to the SPARC galaxy sample.

The analysis demonstrates a **regime-dependent environmental modulation** of outer rotation curve slopes.

---

## Key Result

A statistically significant environmental effect is detected **only above a critical baryonic mass scale**:

- High-mass regime (logM ≥ 10):
  - Spearman ρ ≈ −0.463
  - p ≈ 2.8 × 10⁻⁴
  - OLS (HC3): β ≈ −0.143, p ≈ 0.004
  - R² ≈ 0.25

- Low-mass regime (logM < 10):
  - No correlation (p ≈ 0.98)

This establishes a **mass-dependent transition (SCM-TR)** in the coupling between galaxies and their environment.

---

## Robustness

The result is validated using:

- Bootstrap resampling:
  - β_mean ≈ −0.168
  - 95% CI ≈ [−0.266, −0.094]

- Permutation test:
  - p_perm ≈ 0.0002

- Multivariate regression:
  - Environmental effect remains significant after controlling for baryonic mass

---

## Interpretation

- Low-mass galaxies:
  - Dynamics dominated by internal processes

- High-mass galaxies:
  - External environment modulates outer rotation curves

This supports a **regime-dependent interaction between baryonic structure and environment**.

---

## Repository Contents

```
results/paper1_environment/
    SCM_figure_final.png
    SCM_figure_final.pdf
    SCM_results_table.csv

docs/
    SCM_FINAL_REPORT.docx
```

---

## Data Sources

- SPARC database (Lelli et al. 2016)
- Yang et al. (2007) environment catalog

---

## Citation

If you use this work, please cite the Zenodo DOI associated with this repository.

---

## Status

✔ Fully reproducible  
✔ Statistically robust  
✔ Publication-ready  

---

## Author

Sergio Cámara Madrid  
Independent Researcher — SCM Framework
