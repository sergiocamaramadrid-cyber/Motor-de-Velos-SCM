# Discussion — SCM Regime-Dependent Structural Modulation

**Author:** Sergio Cámara Madrid  
**Version:** v2.3.0  
**Target journal:** Monthly Notices of the Royal Astronomical Society (MNRAS)

---

## 5. Discussion

### 5.1 Summary of results

This work characterises the relationship between the SCM environmental proxy (E_SCM) and the outer velocity-curve slope across a sample of 79 late-type galaxies drawn from the SPARC catalogue. The central finding is that E_SCM does **not** introduce a single, universal modulation of rotation-curve shape. Instead, the signal is strongly mass-dependent:

- In the **low-mass subsample** (log M_bar < 10; N = 40), no statistically significant correlation is detected between E_SCM and the outer slope (p > 0.05 after correction).
- In the **high-mass subsample** (log M_bar ≥ 10; N = 39), a significant negative regression coefficient is recovered (β ≈ −0.14, p ≈ 0.0036), indicating that higher E_SCM values are associated with a flatter outer slope.
- The global interaction term (E_SCM × log M_bar) is not significant (p ≈ 0.42), confirming that the modulation is non-linear and regime-based rather than a smooth continuous interaction.
- Stratification into four quantile bins of E_SCM reveals a progressive flattening of the mass–slope relation with increasing E_SCM — visible in Figure 1 — consistent with a threshold or regime-change interpretation.

### 5.2 Physical interpretation

The regime-dependent nature of the signal is consistent with environmental effects playing a secondary, mass-mediated role in shaping outer kinematics. In low-mass systems, baryonic feedback and internal turbulence may dominate outer-disk kinematics, diluting any environmental imprint. In high-mass systems, where the outer disk is more dynamically coherent and the baryonic potential well is deeper, the influence of large-scale structure on the velocity profile becomes detectable.

This interpretation is deliberately agnostic about the specific physical mechanism. E_SCM is treated here as a **secondary structural descriptor** — a proxy for large-scale environmental conditions — rather than as a causal driver of rotation-curve shape. Causal inference would require additional controls (merger history, HI morphology, filament orientation) beyond the scope of the present dataset.

### 5.3 Relation to existing work

The mass threshold near log M_bar ≈ 10 echoes the transition reported in baryonic Tully–Fisher residual analyses and in studies of the radial acceleration relation scatter. The absence of a global interaction term is consistent with results from environment–kinematics studies using group catalogues, where significant signals typically emerge only in the most massive or isolated subsamples.

The SCM framework adds value by providing a self-consistent, rotation-curve-derived environmental proxy that does not rely on external group membership assignments, making the analysis fully reproducible from publicly available SPARC data.

### 5.4 Limitations

1. **Sample size.** With N = 39 high-mass galaxies, the statistical power is moderate. Replication with a larger, deeper sample (e.g., MaNGA or TNG50 mock curves) is warranted.
2. **Proxy calibration.** E_SCM is computed from rotational kinematics and depends on the outer-slope fitting procedure. Systematic uncertainties in inclination and distance propagate into E_SCM and are not fully accounted for in the present analysis.
3. **Non-linear effects.** The quantile-bin analysis (Figure 1) suggests a non-linear response. A more flexible regression (e.g., GAM or piecewise OLS with a mass knot) may better capture the underlying functional form.
4. **Causality.** The observational design is purely correlational. No causal claim is made.

### 5.5 Conclusions

E_SCM acts as a regime-dependent structural descriptor of outer rotation-curve morphology. The signal is:
- **absent** at low mass,
- **significant** at high mass (β ≈ −0.14, p ≈ 0.0036),
- **non-linear** in its mass dependence (no significant global interaction term).

This characterisation is **defendable**, **reproducible**, and **publishable** as a phenomenological result. It motivates targeted follow-up in larger samples and provides a quantitative baseline for comparing SCM predictions against alternative kinematic frameworks.

---

*Figure reference:* `results/figures/fig_mass_slope_Ebins.pdf`  
*Dataset:* `data/scm_canonical_dataset.csv` (N = 79)  
*Analysis script:* `scripts/plot_mass_slope_by_Ebins.py`
