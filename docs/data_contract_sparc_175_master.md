# SPARC 175 master catalog data contract

This document defines the expected schema for `data/sparc_175_master.csv`.

## Required columns

- `galaxy`: SPARC galaxy identifier (string)
- `delta_f3`: residual or differential F3 diagnostic used for environment analyses (dimensionless)
- `F3`: base F3 observable for each galaxy (dimensionless)
- `logSigmaHI_out`: outer HI surface-density proxy in logarithmic units
- `logMbar`: logarithm of baryonic mass
- `logRd`: logarithm of disk scale length
- `fit_ok`: boolean quality gate for valid fits
- `quality_flag`: categorical quality annotation (`good`, `ok`, `usable`, `clean`, etc.)
- `n_tail_points`: number of deep-regime tail points used in fitting
- `inclination`: disk inclination angle in degrees

## Quality criteria

For environment-model scripts, rows are typically filtered by:

1. `fit_ok == True`
2. finite values in model-required numeric columns
3. optional quality thresholds (e.g., `n_tail_points >= 3`, `inclination >= 30`)
4. optional `quality_flag` whitelist

## Meaning of `delta_f3`

`delta_f3` is the environmental-response target used by paired, controlled, and
out-of-sample analyses. It is interpreted as the galaxy-level deviation signal to
test whether environment (`logSigmaHI_out`) contributes beyond structural controls.
