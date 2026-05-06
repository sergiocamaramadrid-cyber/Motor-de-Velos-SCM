
# SCM-BH Magnetic Candidate Block v1

## Objective

Evaluate whether polarization variability contributes independently
to jet geometric residual structure after controlling for global jet scale.

---

# Model

res_abs ~ logr15 + Pmed_rms

Where:

- res_abs = absolute geometric residual
- logr15 = global jet scale proxy
- Pmed_rms = polarization variability proxy

---

# Dataset

MOJAVE VLBI jet sample

N = 65

---

# Results

## HC3 Robust Regression

beta(logr15) = -4.04
p(logr15) = 0.041

beta(Pmed_rms) = +5.42
p(Pmed_rms) = 0.0027

R² = 0.088

---

# Bootstrap

Median beta(Pmed_rms) = +5.55

IC95(beta):
[-0.72, 18.35]

Fraction beta > 0:
0.9685

---

# Permutation

p_perm_t = 0.0235

p_perm_beta = 0.1245

---

# Collinearity

VIF(logr15) ≈ 1.00
VIF(Pmed_rms) ≈ 1.00

No relevant multicollinearity detected.

---

# Interpretation

A weak but stable association is detected between polarization
variability and jet residual structure after controlling
for global jet geometry.

The result survives:

- HC3 robust regression
- permutation testing on the test statistic
- bootstrap directional stability

However:

- the bootstrap confidence interval still crosses zero
- the explained variance remains modest

This result is therefore treated as:

SCM-L3 candidate evidence

and not as confirmed physical mechanism.

---

# Status

BLOCK STATUS:
FROZEN / REPRODUCIBLE / CANDIDATE

No further exploratory correlations should be added
to this block without independent validation.
