# SCM-BH Progress Report — Jet Structure and Doppler-Regime Modulation

**Author:** Sergio Cámara Madrid  
**Date:** 2026-04-24  
**Tag:** scm-bh-v0.1  
**Status:** Exploratory — reproducible but not yet a final publication claim

---

## 1. Overview

This document summarizes the current state of the SCM-BH exploratory analysis, aimed at understanding the physical drivers of AGN jet opening angles using MOJAVE and Wu & Shen (DR16Q) datasets.

The central question is:

> **What controls the observed jet opening angle θ_jet?**

---

## 2. Data Sources

### 2.1 MOJAVE (VizieR: J/MNRAS/468/4992)

Used variables:

- `theta_jet` → apparent jet opening angle
- `r15` → characteristic jet scale
- `delta` → Doppler factor
- `Gamma`, `beta15` → relativistic parameters

Final usable sample:

- N ≈ 360 (structure)
- N ≈ 65 (full relativistic clean sample)

---

### 2.2 Wu & Shen DR16Q

Used variables:

- `LOGMBH`
- `LOGLBOL`
- `LOGEDD_RATIO`

Final crossmatch:

- N = 77

---

## 3. Methodology

### 3.1 Structural Model

```
θ_jet ~ log(r15)
```

Fitted using OLS regression.

---

### 3.2 Residual Analysis

```
residual_theta = θ_jet − θ_pred(r15)
```

Tested against:

- Eddington ratio
- Black hole mass
- Bolometric luminosity

---

### 3.3 Doppler Analysis

Tested:

- θ_jet vs δ
- residual_theta vs δ

Also performed:

- bootstrap checks
- Cook's distance filtering
- regime split (low δ vs high δ)

---

### 3.4 Combined Model

```
θ_jet ~ log(r15) + δ + δ·log(r15)
```

---

## 4. Results

### 4.1 Structural Dominance

- R² ≈ 0.25–0.32

Jet geometry is strongly correlated with structural scale.

---

### 4.2 No Engine Dependence

No statistically robust correlation found for:

- θ_jet vs LOGEDD_RATIO
- θ_jet vs LOGMBH
- θ_jet vs LOGLBOL

Also no signal in residuals.

---

### 4.3 Doppler Effect (Key Result)

Global:

- ρ ≈ 0.31
- p ≈ 0.011

---

### 4.4 Regime Transition

Split at median δ:

| Regime | Result |
|---|---|
| LOW δ | ρ ≈ 0.52, p ≈ 0.002 |
| HIGH δ | ρ ≈ 0.02, p ≈ 0.91 |

**Interpretation:** Strong dependence at low δ, saturation at high δ.

---

### 4.5 Combined Model

- R² ≈ 0.405
- p_global ≈ 2×10⁻⁶

Significant improvement over structure-only model.

---

## 5. Physical Interpretation

### 5.1 Structural Layer

```
θ_intrinsic = f(r15)
```

Jet opening angle is primarily determined by intrinsic jet structure.

---

### 5.2 Relativistic Layer

```
θ_observed = θ_intrinsic modulated by δ
```

Doppler boosting affects observed geometry, not intrinsic geometry.

---

### 5.3 Regime Behaviour

- **Low δ** → observable geometry varies with δ
- **High δ** → saturation (alignment / beaming limit)

---

## 6. Final Conclusion

Jet opening angle is structurally determined and relativistically modulated.

No evidence is found for direct control by black hole mass or accretion state.

---

## 7. Status

- ✔ Reproducible
- ✔ Statistically validated
- ✔ Physically interpretable
- ⚠ Exploratory (not yet formal publication)

---

## 8. Next Steps

- [ ] Improve sample size for relativistic parameters
- [ ] Produce publication-grade figures
- [ ] Prepare MNRAS manuscript

---

## 9. Citation

See `CITATION.cff` for repository citation.

---

## 10. Author

Sergio Cámara Madrid  
Independent Researcher  
Framework SCM — Motor de Velos

---

## Annex: Full Analysis Narrative

### Objective

The objective of the analysis has been to determine what controls the observable geometry of jets in AGN, specifically the opening angle θ_jet.

Three possible physical drivers were evaluated:

1. Engine (mass, luminosity, accretion rate)
2. Intrinsic jet structure
3. Relativistic effects (Doppler, Lorentz)

---

### Starting Hypothesis

The initial hypothesis was:

> Jet geometry depends on the central engine (M_BH, Eddington ratio)

This was later extended to:

> Jet geometry may also be modulated by relativistic effects

---

### Analysis Steps

**Step 1 — Engine test**

Correlations evaluated:

- θ_jet vs LOGEDD_RATIO
- θ_jet vs LOGMBH
- θ_jet vs LOGLBOL

Result: No significant signal.

---

**Step 2 — Structural model**

Fitted: `θ_jet ~ log(r15)`

Result: R² ≈ 0.25–0.32

Interpretation: Jet structure explains a significant fraction of the geometry.

---

**Step 3 — Residual analysis**

Computed: `residual_theta = θ_jet − model(r15)`

Tested against engine variables:

- residual vs LOGEDD_RATIO
- residual vs LOGMBH
- residual vs LOGLBOL

Result: No signal.

Conclusion: Engine does not control jet geometry either directly or through residuals.

---

**Step 4 — Relativistic analysis**

Evaluated: θ_jet vs δ

Result: ρ ≈ 0.31, p ≈ 0.011

Interpretation: Significant relation between δ and θ_jet.

---

**Step 5 — Regime test**

Split at median δ:

- LOW δ → ρ ≈ 0.52, p ≈ 0.002
- HIGH δ → ρ ≈ 0.02, p ≈ 0.91

Conclusion: The relation depends on the regime.

---

**Step 6 — Combined model**

Fitted: `θ_jet ~ log(r15) + δ + δ·log(r15)`

Result: R² ≈ 0.405, p_global ≈ 2×10⁻⁶

Interpretation: Combined model significantly improves explanatory power.

---

**Step 7 — Visualisation**

Generated: θ_jet vs log(r15) coloured by δ

Visual result:

- Clear slope → structural effect
- Colour gradient → relativistic effect
- Saturation at high δ → regime transition

---

### Key Results Summary

| Layer | Result |
|---|---|
| ✔ Structure | θ_jet depends on r15 |
| ✘ Engine | No dependence on mass, luminosity, or Eddington ratio |
| ✔ Relativistic (conditional) | δ affects θ_jet only in low-δ regime |

---

### Physical Interpretation

The final model is:

```
θ_intrinsic = f(r15)
θ_observed  = θ_intrinsic modulated by δ
```

**Translation:** The jet has its own geometry (structure), and relativity deforms what we observe.

**Regime behaviour:**

- LOW δ → relativistic effect is visible
- HIGH δ → saturation (alignment / beaming)

---

### Final Conclusion

Jet geometry is dominated by its intrinsic structure.

The engine does not directly control the jet.

The Doppler factor introduces an observational modulation, with a regime transition between low-δ and high-δ jets.

---

### What Has Been Achieved

- ✔ Falsified an initial hypothesis (engine)
- ✔ Identified the dominant variable (structure)
- ✔ Detected a real relativistic effect
- ✔ Discovered a regime transition
- ✔ Built a coherent physical model

---

### Work Status

| Dimension | Status |
|---|---|
| Exploratory phase | Complete |
| Physical model | Defined |
| Result | Reproducible |
| Level | Publishable (pre-paper) |

---

### Natural Next Steps

1. Formalise paper (MNRAS)
2. Extend relativistic sample
3. Refine figures and visualisation
