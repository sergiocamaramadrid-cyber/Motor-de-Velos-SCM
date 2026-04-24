# SCM-BH — Regime Transition and Structured Chaos in AGN Jets

## Overview

This module extends the SCM framework to black hole jets (SCM-BH), focusing on the structural behavior of AGN jets.

Using a sample of 65 sources, we identify a critical transition scale:

**r15 ≈ 10 (log r15 ≈ 1.0)**

At this scale, the system transitions between two distinct dynamical regimes.

---

## Main Results

### 1) Regime Transition

- KS test: p ≈ 3.95e-06
- Logistic regression: complete separation

→ Two populations are clearly separated.

---

### 2) Two Dynamical Regimes

**LOW regime (r15 < 10)**

- High dispersion
- No internal scaling law
- Dynamically unstable
- Allows extreme configurations

**HIGH regime (r15 > 10)**

- Lower dispersion
- More collimated jets
- More stable behavior

---

### 3) No Universal Law

A global relation θ vs r15 exists (R² ≈ 0.32), but:

→ It disappears when splitting the sample

> «The apparent correlation is due to population mixing, not a universal physical law.»

---

### 4) Nature of the Transition

The transition is:

- Structural (driven by r15)
- Not purely relativistic
- Not explained by a single parameter

Instead:

> «The system exhibits an emergent transition between dynamical states.»

---

### 5) Structured Chaos

The system behaves as:

- A low-scale regime with high variability and freedom
- A high-scale regime with constrained configurations

→ Order emerges from an underlying chaotic phase.

---

### 6) Outliers

Outliers are not random and split into two mechanisms:

**Structural** (intrinsic instability)

- Occur mainly in LOW regime
- Do not require extreme Doppler factors

**Relativistic** (amplification)

- High δ sources
- Opening angles boosted observationally

---

## Post-Transition Behavior

Within the HIGH regime:

- Weak negative trend between excess radius and opening angle
- Not statistically significant (N = 22, p ≈ 0.23)

> «Post-transition evolution remains unresolved due to limited sample size.»

---

## Key Insight

> «Extreme jet configurations are confined to the low-scale regime, while large-scale jets suppress dynamical freedom and converge toward stable configurations.»

---

## Files

| File | Description |
|------|-------------|
| `scm_bh_regime_labeled_final.csv` | Full source table with r15, θ, and LOW/HIGH regime labels |
| `scm_bh_outliers_classified.csv` | Outliers classified as structural or relativistic |
| `scm_bh_chaos_structured_summary.json` | Summary statistics: regime stats, Fisher-z test, Spearman ρ |

---

## Status

- Transition is **robust** (KS p ≈ 3.95e-06)
- Result is **exploratory but reproducible**
- Post-transition evolution requires larger samples

---

## Author

Sergio Cámara Madrid  
SCM — Motor de Velos Framework
