# SCM-TR: Environmental Regime Transition in Galaxy Outer Dynamics

## 1. Hypothesis

We test whether the outer dynamical state of galaxies, quantified by

F3_SCM ≡ d log V / d log r

is governed solely by baryonic mass, or whether an additional environmental component emerges once mass-driven variance is controlled.

Working hypothesis:

> Mass alone does not explain the signal and appears to partially mask an underlying environmental dependence.

## 2. Methodology

### 2.1 Datasets

- SPARC cleaned sample with environmental proxy
- Yang-like validation dataset
- Gaia Milky Way Cepheid consistency check

### 2.2 Strategy

1. Compute the outer slope observable
2. Split by stellar mass threshold
3. Measure Spearman correlation with environment
4. Regress outer slope against mass
5. Correlate residuals with environment
6. Compare with Yang and Gaia secondary checks

## 3. Results

### 3.1 Global SPARC result

For the cleaned sample, we detect a statistically significant negative correlation between environmental proxy and outer slope:

- N = 79
- Spearman ρ = -0.365
- p = 9.3 × 10^-4

### 3.2 Mass split

Low-mass regime:

- no significant correlation

High-mass regime:

- strong negative correlation
- ρ ≈ -0.44 to -0.49
- p ≈ 10^-3 to 10^-4

Interpretation:

> A statistically significant environmental modulation is detected only in the high-mass regime, indicating a regime-dependent dynamical effect.

### 3.3 Residual test

We fit slope_tail ~ logM and remove the mass-dependent component.

Base fit:

- slope = +0.0982
- intercept = -0.4311
- R² = 0.0981
- p = 0.0111

Residual vs environment:

- Global: ρ = -0.3917, p = 0.00125
- Low mass: ρ = -0.3450, p = 0.116
- High mass: ρ = -0.4100, p = 0.00632

Interpretation:

> The environmental signal remains significant in the high-mass regime after controlling for mass, showing that the trend is not reducible to baryonic mass alone.

### 3.4 Yang validation

The same qualitative structure is recovered with the Yang-like estimator:

- no signal at low mass
- significant signal at high mass

### 3.5 Gaia consistency check

The Milky Way analysis is used as a consistency check rather than as a primary test.

- continuous outer trend present
- no statistically significant hemispheric anisotropy detected

## 4. Discussion

These results indicate that baryonic mass sets part of the baseline organisation of outer dynamics, but does not absorb the full signal. Once the mass-driven variance is removed, a statistically significant environmental modulation remains in the high-mass regime.

This supports a regime-dependent picture in which environment acts as a secondary dynamical contribution that becomes observable only above a characteristic mass scale.

## 5. Conclusion

The outer dynamics of galaxies cannot be fully described by baryonic mass alone.

A statistically significant environmental modulation is detected only in the high-mass regime, while no significant signal is found at low mass. After controlling for mass, the signal remains significant, indicating that outer-disk dynamics retain an environmental dependence beyond baryonic mass alone.
