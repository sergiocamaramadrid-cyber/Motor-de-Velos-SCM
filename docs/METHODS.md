# METHODS — Statistical Validation Framework of the SCM Model

## 1. Overview

The SCM global-feature model is evaluated using a structurally valid,
galaxy-level out-of-sample (OOS) validation framework designed to eliminate
hierarchical data leakage and ensure physical interpretability.

This document defines the statistical guarantees underpinning the Framework.

---

## 2. Out-of-Sample Evaluation (Galaxy-Level)

Validation is performed using **GroupKFold cross-validation**, where the
grouping unit is the entire galaxy.

- Model parameters are fitted exclusively on **TRAIN** galaxies.
- Performance is evaluated strictly on unseen **TEST** galaxies.
- Radial points from the same galaxy never appear in both TRAIN and TEST.

This eliminates **hierarchical leakage** that would otherwise arise from radial
correlations within a galaxy.

---

## 3. Model Hierarchy

Three nested models are compared:

### 3.1 BTFR Baseline

$$v_{\text{obs}} = \beta \log M + C$$

### 3.2 SCM (Linear)

$$v_{\text{obs}} = \beta \log M + C + a \log g_{\text{bar}} + b \log j$$

### 3.3 SCM (Full with Hinge)

$$
v_{\text{obs}} =
\beta \log M + C +
a \log g_{\text{bar}} +
b \log j +
d \cdot \max\!\left(0,\; \log g_0 - \log g_{\text{bar}}\right)
$$

### Physical Constraint

The hinge strength parameter is constrained: **d ≥ 0**.

If the unconstrained optimiser returns d < 0, the parameter is clipped to
zero.  Effective parameter count is reduced accordingly.

This enforces physical consistency (no negative pressure regime).

---

## 4. Structural Permutation Test (Strong Null Model)

### 4.1 Motivation

A simple target permutation (shuffling $v_\text{obs}$) only tests for the
presence of any statistical signal.  It does not test whether the model is
rediscovering a known baryonic scaling relation (e.g., the McGaugh RAR).

To address this, we construct a **structural null model**.

### 4.2 Procedure

1. $\log g_\text{bar}$ is shuffled **globally across galaxies**.
2. Marginal distributions of mass and velocity are preserved.
3. The spatial/acceleration coupling is destroyed.
4. The model is refit on TRAIN folds for each permutation.
5. OOS RMSE is recomputed on TEST folds.

**Empirical p-value:**

$$
p = \frac{1 + \#\{\,\text{RMSE}_{\text{perm}} \le \text{RMSE}_{\text{real}}\,\}}{1 + N_{\text{perm}}}
$$

### 4.3 Why This is a Strong Null

Unlike a row-wise shuffle, this procedure:

- Breaks the dynamical coupling between baryonic mass distribution and the
  local velocity field.
- Preserves the global distribution of acceleration values.
- Removes any point-to-point structural coherence.

Therefore, any surviving improvement under this null cannot be attributed to
trivial statistical correlation.

> If the SCM hinge term remains predictive under OOS validation **and** fails
> under structural permutation, it demonstrates that the model responds to a
> genuine acceleration-scale structure rather than rediscovering a baryonic
> scaling tautology.

---

## 5. Galaxy-Level Non-Parametric Test

For each galaxy, OOS RMSE deltas are computed:

- Δ(full − BTFR)
- Δ(full − no-hinge)

A **one-sided Wilcoxon signed-rank test** is applied:

```
alternative = "less"
```

Testing whether SCM(full) significantly reduces prediction error across
galaxies.  This avoids Gaussian residual assumptions and ensures robustness to
outliers.

---

## 6. Reproducibility and Artefacts

Each audit run (via `scripts/audit_scm.py`) produces the following files in
the output directory:

| File | Contents |
|---|---|
| `groupkfold_metrics.csv` | Fold-level RMSE for all three models |
| `gal_results.csv` | Per-galaxy OOS RMSE + signed BTFR residuals |
| `coeffs_by_fold.csv` | Fitted coefficients (β, C, a, b, d, log g₀) per fold |
| `permutation_runs.csv` | OOS RMSE for each permutation run |
| `permutation_summary.json` | p-value, permutation statistics, Wilcoxon result |
| `audit_summary.txt` | Human-readable audit report |

These files ensure full traceability of:

- OOS performance by model and fold
- Coefficient stability across GroupKFold splits
- Permutation null distribution
- Statistical significance at both fold and galaxy level

The per-galaxy audit input (`results/audit/sparc_global.csv`) is generated
deterministically by `src/scm_analysis.run_pipeline` and contains the columns
`galaxy_id`, `logM`, `log_gbar`, `log_j`, `v_obs`.

---

## 7. Philosophical Position

This framework **does not assume new physics**.

It tests whether the SCM correction term:

1. Survives galaxy-level OOS validation (GroupKFold, unit = galaxy).
2. Fails under structural permutation (breaks g_bar coupling).
3. Maintains physical sign consistency (d ≥ 0).
