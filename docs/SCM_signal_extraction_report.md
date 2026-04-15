# SCM Framework – Multi-Scale Signal Extraction Report

**Author:** Sergio Cámara Madrid  
**Project:** SCM – Motor de Velos  
**Date:** April 2026

---

## 1. Conceptual Motivation

The SCM framework initially focused on detecting environmental modulation in the outer regions of galaxy rotation curves through a single observable: the outer slope ("slope_tail" / ΔF3).

Once a statistically significant signal was identified, a natural question emerged:

> «If a signal is detectable at one scale, could additional hidden signals exist at other scales?»

This motivated extending the framework beyond a single observable into a multi-scale signal detection approach.

---

## 2. Core Validated Result (SCM v1.0)

Using SPARC galaxies:

- Global correlation: ρ ≈ −0.365 (p < 10⁻³)
- Low mass regime: no signal
- High mass regime:
  - ρ ≈ −0.42 (p ≈ 0.003)
  - β_env ≈ −0.14 (p ≈ 0.006)

**Interpretation**

Environmental modulation:

- ❌ Not universal
- ✅ Activated above a critical mass scale

This defines a regime transition, not a continuous effect.

---

## 3. From Single Observable to Multi-Scale Analysis

The outer slope captures macroscopic dynamics, but rotation curves may contain additional structure.

We therefore test:

> «Do residuals of rotation curves contain characteristic frequencies?»

---

## 4. Methodology

### 4.1 Preprocessing

For each galaxy:

1. Load rotation curve (R, Vobs)
2. Fit quadratic trend
3. Compute residual:

```
V_resid = Vobs − Vtrend
```

### 4.2 Interpolation

- Uniform grid (512 points)
- Required for FFT stability

### 4.3 Spectral Analysis

- FFT applied to residuals
- Power spectrum computed
- Peak detection via percentile threshold

### 4.4 Statistical Validation

- Permutation test (100–200 realizations)
- Null hypothesis: no preferred frequency

---

## 5. Synthetic Validation

A synthetic signal test confirmed:

- Correct recovery of injected frequency
- Robust peak detection
- Non-random spectral signature

---

## 6. SPARC Application

- ~525 galaxies processed
- Output catalog: `sparc_frequency_analysis.csv`

Detected frequency range:

- ~0.02 to ~1.7 (1/kpc)

---

## 7. Interpretation of Frequency Detection

**Important constraint:**

> «A detected frequency does NOT automatically imply a physical oscillation.»

It indicates:

- Presence of preferred spatial scales in residuals

Possible origins:

- Disk structure
- Resonances
- Environmental coupling
- Sampling effects

---

## 8. Revalidation of Core SCM Result

Using a clean dataset (N=79):

**Global:**

ρ ≈ −0.365 (p ≈ 9×10⁻⁴)

**Low mass:**

No correlation

**High mass:**

ρ ≈ −0.42 (p ≈ 0.003)

**OLS (HC3):**

β_env ≈ −0.139 (p ≈ 0.006)

---

## 9. Failure of Continuous Interaction Model

A model including:

```
env + mass + (env × mass)
```

was tested and failed to produce significance.

**Interpretation:**

> «Environmental coupling is NOT continuous»

Instead:

> «It activates above a threshold»

---

## 10. Physical Interpretation

The system behaves as:

- Low mass → internally dominated
- High mass → environment-coupled

This defines a **two-regime dynamical system**.

---

## 11. Framework Evolution

**SCM v1.0 (validated)**

- ΔF3 / slope_tail
- Environmental modulation
- Mass threshold

**SCM v1.1 (exploratory)**

- Frequency-domain analysis
- Residual structure detection
- Multi-scale probing

---

## 12. Methodological Insight

The key advancement is procedural:

1. Detect a signal
2. Validate it rigorously
3. Extend the search space
4. Keep observables separated until validated

---

## 13. Conclusions

- Environmental modulation is real and statistically robust
- It is not universal, but regime-dependent
- The framework can be extended to multi-scale signal detection
- Rotation curves may encode additional hidden structure

---

## 14. Outlook

Next steps:

- Correlate dominant frequency with mass and environment
- Evaluate independence from ΔF3
- Define second observable if validated

---

## Final Statement

> «The SCM framework is not only a detector of signals, but a structured methodology to reveal hidden dynamical regimes in galaxy rotation curves.»
