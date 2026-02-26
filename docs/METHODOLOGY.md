# SCM — Motor de Velos: Methodological Scope

**Author:** Sergio Cámara Madrid  
**Version:** v0.6.0  
**Document type:** Formal scope declaration

---

## 1. What this framework is

Motor de Velos SCM is a **phenomenological rotation-curve model** evaluated
against the SPARC catalogue of 175 disc galaxies (Lelli et al. 2016).

The model parameterises a supplementary centripetal term V_velos that, added in
quadrature to the baryonic contributions (gas, disc, bulge), reproduces the
observed flat or rising rotation curves without invoking an explicit dark-matter
halo.

The repository provides an **auditable statistical evaluation pipeline** —
not a cosmological theory.

---

## 2. What has been validated (v0.6.0)

The following tests have been implemented, are reproducible, and are covered
by automated unit and integration tests:

| Validation layer | Method | Status |
|---|---|---|
| In-sample goodness-of-fit | Reduced χ² per galaxy | ✅ v0.5 |
| Model comparison | Corrected AIC (AICc) vs. four ν-models | ✅ v0.5 |
| Out-of-sample generalisation | Radial-split OOS (no post-hoc tuning) | ✅ PR #56 |
| Permutation test (structural) | Label-shuffled null; p-value reported | ✅ PR #56 |
| Sensitivity analysis | χ² vs. a₀ grid; frozen at 1.2×10⁻¹⁰ m/s² | ✅ PR #56 |
| Deep-regime slope diagnostic | β = 0.5 test in g_bar ≪ a₀ regime | ✅ v0.5 |
| Multicollinearity (VIF) | VIF per predictor; threshold 5 / 10 | ✅ PR #57 |
| Design-matrix stability | Condition number κ of standardised X | ✅ PR #57 |
| Partial correlation | ρ(log g_bar, log v_obs) controlling for M, j | ✅ PR #57 |
| Audit table | 175-galaxy sparc_global.csv, gapless index | ✅ PR #57 |

All outputs are machine-readable, versioned, and reproducible from the public
codebase.

---

## 3. What has NOT been established

The following claims are **outside the current evidential scope** of this
repository and must not be inferred from the validation results above:

| Claim | Why it is out of scope |
|---|---|
| "The model replaces dark matter" | SPARC rotation curves alone cannot discriminate between a dark-matter halo and a functional equivalent; independent cosmological probes are required. |
| "This is a new cosmological model" | The model operates at the galactic (rotation-curve) scale only. CMB, BAO, large-scale structure, and cluster dynamics are untested. |
| "ΛCDM is falsified" | No formal comparison against NFW or ΛCDM galaxy simulations has been conducted in this repository. |
| "The result is definitive" | This is a v0.6.0 evaluation. Peer review and independent replication have not occurred. |
| "The hinge parameter g₀ is universal" | g₀ = 3.5×10⁻¹¹ m/s² is frozen for internal diagnostics; it has not been derived from first principles or fitted to an independent dataset. |

---

## 4. What is required before stronger claims can be made

Progression to cosmological-level claims requires at minimum:

1. **External dataset validation** — LITTLE THINGS (dwarf irregulars), galaxy
   clusters (e.g., Coma, Bullet), or CMB fits.
2. **Formal MOND/NFW comparison** — head-to-head AICc or Bayesian model
   comparison against a properly implemented NFW halo and MOND (simple / standard ν).
3. **Independent replication** — reproduction of key results by a group not
   involved in the original analysis.
4. **Peer-reviewed preprint or publication** — submission to a refereed venue
   (e.g., MNRAS, A&A, JCAP) and response to referee criticism.
5. **GroupKFold OOS validation** — full leave-one-galaxy-out cross-validation
   beyond the current radial-split protocol.

---

## 5. Falsifiable predictions

The following statements are falsifiable with available data or near-term
observations; they represent the scientific frontier of the v0.6.0 model:

1. **Deep-regime slope** — in the regime g_bar < a₀/3, the log-log slope
   β(log g_obs / log g_bar) = 0.50 ± 0.02.  A measured β < 0.44 would
   constitute structural failure of the model.

2. **VIF stability** — all three predictors (log M_bar, log g_bar, log j_*)
   should have VIF < 10 when evaluated on a second independent rotation-curve
   catalogue.  VIF ≥ 10 for any predictor would indicate feature
   reparameterisation is required.

3. **OOS residual distribution** — the distribution of OOS residuals should be
   consistent with N(0, 1) under the fitted uncertainty model.  Systematic
   skew or excess kurtosis would indicate model mis-specification.

4. **Permutation null** — permuting galaxy labels should degrade χ²_reduced by
   at least a factor of 2 relative to the structured model.

---

## 6. Epistemic status summary

> The Motor de Velos SCM v0.6.0 is a phenomenological model that describes
> galaxy rotation curves with statistical rigour comparable to established
> interpolation functions (simple ν, standard ν, AQUAL).  It has passed
> internal structural diagnostics (VIF, condition number, partial correlation)
> and a fixed out-of-sample protocol on the SPARC sample.
>
> It has **not** been validated at cosmological scales, has not been formally
> compared to dark-matter halo models, and has not undergone peer review.
> No cosmological claims are made or supported by this repository.

---

## 7. References

- Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). *SPARC: Mass Models for
  175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves.*
  AJ, 152, 157. doi:10.3847/0004-6256/152/6/157

- Famaey, B. & McGaugh, S. S. (2012). *Modified Newtonian Dynamics (MOND):
  Observational Phenomenology and Relativistic Extensions.*
  Living Rev. Relativity, 15, 10.

- McGaugh, S. S. (2004). *The Mass Discrepancy–Acceleration Relation:
  Disk Mass and the Dark Matter Distribution.*
  ApJ, 609, 652.

- Burnham, K. P. & Anderson, D. R. (2002). *Model Selection and Multimodel
  Inference: A Practical Information-Theoretic Approach* (2nd ed.).
  Springer.
