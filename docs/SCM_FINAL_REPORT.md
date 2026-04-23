# SCM Framework — Final Empirical Report (v2.4.0)

Author: Sergio Cámara Madrid  
Date: April 2026  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM  
DOI: https://doi.org/10.5281/zenodo.19455777  

---

## 1. Objective

This work investigates whether galaxy rotation dynamics depend solely on internal properties (baryonic mass) or also on environmental factors.

The approach is strictly empirical and fully reproducible.

---

## 2. Data and Method

- Dataset: SPARC (Lelli et al. 2016)  
- Observable:
  - F3 (slope_tail) — outer rotation curve slope  

Sample division:
- High-mass: logM ≥ 10  
- Low-mass: logM < 10  

Methods applied:
- Spearman correlation  
- OLS regression (HC3 robust errors)  
- Matched-pairs analysis  
- Bootstrap and permutation testing  

---

## 3. Results

### High-mass galaxies (logM ≥ 10)

- Spearman correlation:  
  ρ = −0.463  
  p = 2.85 × 10⁻⁴  

- OLS (HC3):  
  β = −0.143  
  p = 0.0036  
  R² ≈ 0.21  

- Matched pairs:  
  p = 0.00067  

→ A statistically significant correlation is observed between environment proxy and outer slope.

---

### Low-mass galaxies (logM < 10)

- Spearman correlation:  
  ρ ≈ 0.006  
  p ≈ 0.98  

→ No statistically significant correlation is detected.

---

## 4. Interpretation

The results indicate a mass-dependent regime transition:

- Low-mass galaxies behave independently of environment  
- High-mass galaxies show measurable environmental modulation  

This is an empirical statement derived from data.  
No causal mechanism is proposed.

---

## 5. Robustness

The main result is stable under:

- Bootstrap resampling  
- Permutation testing  
- Independent matched-pair analysis  

No single statistical method drives the signal.

---

## 6. Limitations

- Environmental proxy may not capture all external influences  
- Sample size limits fine-grained subdivision  
- The analysis is observational and does not establish causation  

---

## 7. Reproducibility

All data and analysis pipelines are publicly available:

Repository:  
https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM  

DOI:  
https://doi.org/10.5281/zenodo.19455777  

---

## 8. Conclusion

A clear empirical result is established:

- Environmental dependence emerges only in high-mass galaxies  
- No dependence is observed in low-mass systems  

This supports a regime-based description of galaxy dynamics, where environmental sensitivity is activated above a critical mass scale.
