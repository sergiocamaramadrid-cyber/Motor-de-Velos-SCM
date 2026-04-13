# SCM — Motor de Velos: Paper Summary

**Authors:** Sergio Cámara Madrid  
**Date:** 2026-04-13  
**DOI:** 10.5281/zenodo.19455777

---

## Abstract

We present a comprehensive empirical evaluation of the SCM (Motor de Velos / Fluid Condensation Model) applied to galaxy rotation curves from the SPARC dataset. Using 175 galaxies, we fit the deep-regime slope β in the F3 relation:

    log g_obs = β · log g_bar + C

and find a mean β = 0.503 ± 0.004, statistically consistent with the MOND/SCM prediction of β = 0.5 (p = 0.47, t-test). We further demonstrate that β correlates with an environmental overdensity proxy (Spearman ρ = 0.48, p = 4.2×10⁻⁶), and that inclusion of an environmental term improves model fit by ΔAIC = 6.2 relative to a baryonic-only model.

---

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| N galaxies (reliable β) | 162 | After quality cuts |
| Mean β | 0.503 ± 0.004 | Bootstrap CI |
| Median β | 0.502 | |
| σ(β) | 0.029 | Galaxy-to-galaxy scatter |
| Δ from β = 0.5 | +0.003 | SCM-consistent |
| p-value (H₀: β=0.5) | 0.47 | Two-tailed t-test |
| Spearman ρ (env vs β) | 0.48 | p = 4.2×10⁻⁶ |
| ΔAIC vs baryonic-only | 6.2 | Environment term significant |
| χ²_red (median) | 0.996 | Good fit quality |

---

## Methods

### Data
- **SPARC dataset**: 175 disk galaxies with measured rotation curves.
- **Gas fractions**: computed from SPARC baryonic mass components.
- **Environment proxy**: stellar-mass-derived isolation index (0–1 scale).

### SCM pipeline
1. Per-galaxy deep-regime fit: radial points with g_bar < 0.3 × a₀ are selected; OLS is applied to log g_obs vs log g_bar.
2. β distribution: one-sample t-test vs H₀: β = 0.5.
3. Environment correlation: Spearman ρ between env_proxy and β.
4. Model comparison: ΔAIC between SCM + environment and baryonic-only.

### Gas analysis
- Gas-dominated galaxies (f_gas > 0.5): 28 galaxies.
- Gas-velocity relation: log V_gas ∝ 0.24 · log M_gas + C (R² = 0.94).

---

## Conclusions

1. The SCM deep-regime slope is β ≈ 0.5, fully consistent with the Motor de Velos / MOND prediction.
2. A statistically significant positive correlation exists between β and environmental overdensity, indicating environmental modulation of the rotation-curve kinematics.
3. The environmental term provides a measurable improvement in model fit (ΔAIC = 6.2).
4. Gas-fraction analysis confirms gas-dominated dwarfs follow a tight V_gas–M_gas relation consistent with SCM predictions.

---

## Reproducibility

All results are fully reproducible:

```bash
cd SCM_Framework_Final_20260413_165000/
python scripts/build_e_env.py    # build environment catalog
python scripts/análisis_gas.py   # gas analysis
python scripts/run_scm.py        # full pipeline
```

Outputs are written to `results/`.
