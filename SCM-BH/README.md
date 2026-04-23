# SCM-BH — Statistical Correlation Metrics for Black Hole Systems

## Overview

SCM-BH is an independent extension of the SCM methodology applied to black hole systems with resolved jets.

The goal is empirical:

- identify whether jet observables show statistical organization
- test whether any signal depends on black hole mass
- avoid assuming a physical mechanism in advance

---

## Core Question

Do black hole jet properties depend only on mass, or do they exhibit regime-dependent behavior when combined with a local energetic proxy?

---

## Working Hypothesis

Jet opening angle may show regime-dependent statistical behavior when combining black hole mass and radiative energy scale.

This is a testable empirical hypothesis, not a causal claim.

---

## Minimum Variables

- `source_id`
- `theta_jet`
- `logM_BH`
- `logL_bol`

---

## Analysis Plan

### Level A
- Spearman: `theta_jet` vs `logM_BH`
- Spearman: `theta_jet` vs `logL_bol`
- Split by mass

### Level B
- Construct: `E_BH = logL_bol - 2*logM_BH`
- Compare against individual variables

---

## Robustness

- permutation test
- bootstrap
- robust regression

---

## Interpretation Policy

- no causal claims
- no physical mechanism assumed
- negative results are valid

---

## Status

v0.1 — initialized, pending real dataset ingestion

---

## Expected input file

`data/processed/bh_catalog_clean.csv`

Required columns:
- `source_id`
- `theta_jet`
- `logM_BH`
- `logL_bol`
