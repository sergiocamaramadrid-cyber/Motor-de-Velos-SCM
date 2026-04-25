# SCM-BH v0.3 — Jet Regime Transition in AGN

## Overview

We identify a statistically significant structural transition in AGN jets at:

**r15 ≈ 10**

Beyond this scale, jets do not saturate but continue to collimate.

> **Key Result:** Jets exhibit a statistically significant transition at r15 ≈ 10, followed by continued collimation in the high regime.

---

## Data

- **Source:** MOJAVE (J/MNRAS/468/4992)
- **N:** 360 AGN jets

---

## Methods

- KS test (regime separation)
- Spearman correlation (HIGH regime)
- Bootstrap confidence intervals
- Outlier robustness
- Cut stability (8–12)
- Doppler control (δ)

---

## Results

| Test | Result |
|------|--------|
| KS test | p ≪ 0.01 → distinct regimes |
| HIGH regime Spearman ρ | ≈ −0.36 (p < 0.001) |
| Bootstrap CI | excludes 0 |
| Cut stability (8–12) | stable across cuts |

---

## Physical Interpretation

We find evidence for **continued collimation** beyond the critical scale, consistent with a transition in jet dynamics rather than saturation.

This result is consistent with the parabolic-to-conical transition reported by Kovalev et al. (2020), though conversion to gravitational radii is required to confirm a common physical scale.

---

## Files

| File | Description |
|------|-------------|
| `scm_bh_mojave_clean.csv` | Cleaned MOJAVE dataset (N=360) with r15, θ, δ, regime label |
| `scm_bh_mojave_summary.json` | Results summary: KS p-value, Spearman ρ, bootstrap CI |
| `figure_transition.png` | Main figure: θ vs r15 with regime separation |
| `scm_bh_regime_labeled_final.csv` | Exploratory sample (N=65) from v0.2 analysis |
| `scm_bh_outliers_classified.csv` | Outliers classified as structural or relativistic (v0.2) |
| `scm_bh_chaos_structured_summary.json` | Summary statistics from v0.2 analysis |

---

## Reproducibility

All results can be reproduced using publicly available MOJAVE data (J/MNRAS/468/4992).

---

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19455777.svg)](https://doi.org/10.5281/zenodo.19455777)

*(To be updated after v0.3 Zenodo release)*

---

## Status

- Transition is **robust** (KS p ≪ 0.01, stable across cuts 8–12)
- Result is **exploratory but reproducible**
- Continued collimation in HIGH regime confirmed (N=360)

---

## License

MIT

---

## Author

Sergio Cámara Madrid  
SCM — Motor de Velos Framework
