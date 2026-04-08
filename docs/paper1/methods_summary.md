# Methods summary — SCM environmental analysis (SPARC)

## Pipeline overview

```
Data: SPARC subset (N = 79)
Selection: cleaned, deduplicated
Observable: delta_f3 = slope_tail − 0.5
Environment proxy: delta_mass_std
Mass threshold: derived via scan (logM_crit ≈ 10.05)
Regression: OLS with HC3 errors
```

---

## 1. Data

| Item | Value |
|---|---|
| Source | SPARC catalogue (Lelli et al. 2016) |
| Fixture | `data/sparc_subset.csv` (versioned, N = 79) |
| Deduplication | One row per galaxy; full-sample duplicates removed |
| Columns used | `galaxy`, `logM`, `delta_mass_std`, `slope_tail` |

---

## 2. Observable — ΔF₃

ΔF₃ measures the deviation of the outer-disk logarithmic rotation-curve slope from the SCM reference value (β_ref = 0.5):

```
delta_f3 = slope_tail − 0.5
```

where `slope_tail` is the best-fit dlog V_obs / dlog r over the outer radial bins (deep-regime).  A value of ΔF₃ = 0 indicates perfect agreement with the SCM flat-rotation prediction; negative (positive) values indicate a steeper (shallower) outer slope.

Script: `scripts/generate_f3_catalog.py` / `scripts/generate_f3_catalog_from_contract.py`

---

## 3. Environment proxy — δ_mass,std

`delta_mass_std` is the standardised (z-scored) log specific angular momentum `log j`, used as a proxy for environmental pressure on outer-disk kinematics.  Standardisation is performed over the full 79-galaxy sample so that the mean is zero and the standard deviation is one.

For the physical definition of the underlying δ_mass overdensity quantity, see `docs/paper1/methods_delta_mass.md`.

---

## 4. Mass threshold — logM_crit

The optimal stellar mass cut is determined by a data-driven scan over logM values from 10.0 to 11.3 (step 0.05).  For each candidate cut the script computes:

- Spearman ρ between `delta_f3` and `delta_mass_std` for galaxies with `logM ≥ logM_cut`
- A composite signal score:  `|ρ| × √N × (−log₁₀ p)`

The cut that maximises the score is selected as `logM_crit`.

| logM_crit | N (high-mass) | ρ (Spearman) | p-value |
|---|---|---|---|
| **10.05** | 56 | −0.480 | 1.79 × 10⁻⁴ |

Script: `scripts/plot_sparc_mass_scan.py`  
All 21 evaluated cuts yield ρ < 0, p < 0.05.

---

## 5. Regression — OLS with HC3 robust errors

Two nested OLS models are fitted on the high-mass subsample (N = 56, logM ≥ 10.05):

| Model | Formula |
|---|---|
| Model 1 (simple) | `delta_f3 ~ delta_mass_std` |
| Model 2 (mass-controlled) | `delta_f3 ~ delta_mass_std + logM` |

Standard errors use the HC3 heteroscedasticity-consistent estimator (`statsmodels` `HC3`), which is robust to non-constant variance without assuming a specific functional form for the variance.

Key result: the `delta_mass_std` coefficient is **negative and significant (p < 0.05, HC3)** in both models, confirming the environmental signal survives baryonic mass control.

Script: `scripts/sparc_ols_regression.py`

---

## 6. Split-by-mass figure

For visual inspection the full 79-galaxy sample is split at the median logM (≈ 10.64) and ΔF₃ vs δ_mass,std is plotted for each half independently.

| Subsample | N | ρ (Spearman) | p-value |
|---|---|---|---|
| Low mass (logM < 10.64)  | 39 | −0.15 | 0.36 (n.s.) |
| High mass (logM ≥ 10.64) | 40 | **−0.49** | **0.001** |

Script: `scripts/plot_sparc_split_mass.py`

---

## 7. Reproducibility

All scripts are deterministic given the committed fixture `data/sparc_subset.csv`.  The full pipeline can be re-run as:

```bash
pip install -r requirements.txt

# Mass threshold scan
python scripts/plot_sparc_mass_scan.py --csv data/sparc_subset.csv

# Split-by-mass figure
python scripts/plot_sparc_split_mass.py --csv data/sparc_subset.csv

# OLS regression summary
python scripts/sparc_ols_regression.py --csv data/sparc_subset.csv --m-crit 10.05

# Run all tests
python -m pytest -q   # expected: 443 passed
```

---

## 8. Key references

- Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass Models for 175 Disk Galaxies. *AJ*, 152, 157.
- White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator. *Econometrica*, 48, 817–838.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators. *Journal of Econometrics*, 29, 305–325.
