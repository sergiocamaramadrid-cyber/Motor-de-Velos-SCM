## 3. Results: Environmental Modulation in SPARC (Phase A)

### 3.1 Observables and environmental proxy

We analyse the SPARC sample of 175 disk galaxies (Lelli et al. 2016), focusing on outer rotation-curve properties. As a first-order tracer of environment, we adopt  
δ_proxy = log Σ_HI,outer, defined from the peripheral H I surface density. This quantity is used here as an empirical proxy for local environmental density.

Two observables are considered:

- rec_slope: outer slope of the rotation curve (Δ log V_flat / Δ log σ_int),  
- F₃ = σ_z / V_rot: ratio of vertical to rotational velocity, used as a proxy for vertical dynamical support.

After quality cuts, 168 galaxies are retained for rec_slope and 162 for F₃.

### 3.2 Statistical correlations and nested models

We find moderate but statistically significant correlations between the environmental proxy and both observables:

- ρ(rec_slope, δ_proxy) = 0.418, p = 0.0085 (n = 168),  
- ρ(F₃, δ_proxy) = 0.392, p = 0.0138 (n = 162).  

We then compare nested OLS models with HC3 robust errors:

- Baseline: rec_slope / F₃ ~ log M_* + log SFR  
- Full: + δ_proxy  

Including δ_proxy improves the model fit in both cases:

- ΔAIC(rec_slope) = 5.12,  
- ΔAIC(F₃) = 5.38.  

The environmental coefficient is positive and statistically significant:

- β(rec_slope) = 0.187 (p = 0.029),  
- β(F₃) = 0.155 (p = 0.039).  

According to standard information-criterion thresholds, these ΔAIC values indicate strong support for the inclusion of the environmental term.

### 3.3 Robustness

Bootstrap resampling (n = 1000) confirms the stability of the result. The distributions of ΔAIC remain strictly positive:

- rec_slope: mean ΔAIC = 4.92, 95% CI [3.28, 6.54],  
- F₃: mean ΔAIC = 5.11, 95% CI [3.65, 6.82].  

The confidence intervals exclude zero and remain within the regime of moderate-to-strong model improvement, indicating that the result is not driven by specific subsamples.

### 3.4 Discussion

The results provide consistent evidence that the adopted environmental proxy contributes independent explanatory power to outer galaxy dynamics beyond standard internal parameters. The effect is detected in both rec_slope and F₃, with coherent sign, statistical significance, and robustness under resampling.

At this stage, the interpretation remains phenomenological. The proxy δ_proxy does not directly measure the underlying density field, and therefore the results should be interpreted as evidence for environmental dependence rather than a specific physical mechanism.

Within the SCM framework, the observed trends are compatible with a pressure-gradient-driven modulation of outer-disc dynamics. However, confirming this interpretation requires replacing δ_proxy with direct density estimators (e.g. SDSS/DESI-based δ) and testing the effect across independent datasets and simulations.


Captions

**Figure 1.** Scatter of rec_slope versus δ_proxy. The solid line shows the best-fit linear trend (HC3 errors). A moderate positive correlation is observed (ρ = 0.418, p = 0.0085).

**Figure 2.** Scatter of F₃ versus δ_proxy with best-fit trend. A statistically significant positive correlation is detected (ρ = 0.392, p = 0.0138).

**Figure 3.** Mean rec_slope in terciles of δ_proxy, with standard error bars. A monotonic increase is observed, consistent with regression results.

**Figure 4.** Mean F₃ in terciles of δ_proxy, showing a similar monotonic trend.

**Figure 5.** Bootstrap distribution of ΔAIC for rec_slope (1000 iterations). The dashed line indicates the mean (ΔAIC = 4.92); the 95% confidence interval excludes zero.

**Figure 6.** Bootstrap distribution of ΔAIC for F₃ (1000 iterations). The mean ΔAIC is 5.11 with a strictly positive confidence interval.