# fit_f3_linear_regression.py

Purpose
-------
Fits a linear model:

F3 ~ logSigmaHI_out + logMbar + logRd

Usage
-----

python scripts/fit_f3_linear_regression.py --input data/sparc_175_master.csv

Outputs
-------

results/f3_regression/

f3_regression_coefficients.csv
f3_regression_summary.json

The summary reports:

• intercept
• regression coefficients
• R²
• number of galaxies used
• number of rows removed due to NaN


# test_paired_environment.py

Purpose
-------
Runs a matched-pairs causal-style check of the environmental signal under mass/size controls:

ΔF3 ~ ΔlogSigmaHI_out

Minimum input columns
---------------------
- galaxy
- delta_f3
- F3
- logSigmaHI_out
- logMbar
- logRd
- fit_ok

Optional quality columns used when available:
- n_tail_points
- inclination
- quality_flag

Quality filters
---------------
- fit_ok must be true
- finite values for delta_f3, F3, logSigmaHI_out, logMbar, logRd
- if present: n_tail_points >= --min-tail-points
- if present: inclination >= --min-inclination
- if present: quality_flag in {good, ok, usable, clean}

Outputs
-------
results/paired_environment/

- paired_sample.csv
- paired_stats_summary.csv
- paired_bootstrap.csv
- placebo_tests.csv
- delta_f3_vs_delta_logSigmaHI.png
- run_metadata.json

Physical interpretation
-----------------------
If matched pairs with similar logMbar and logRd show a systematic positive association
between ΔlogSigmaHI_out and Δdelta_f3, this supports a robust environmental contribution
to the F3 residual structure beyond mass-size matching alone.


# stress_test_framework_scm.py

Purpose
-------
Stress-test the observational applicability domain of the SCM framework.

Usage
-----

python scripts/stress_test_framework_scm.py

Output
------

scripts/stress_test_results.csv

Framework states
----------------
- OUT_OF_DOMAIN
- FUTURE_EXTENSION_CANDIDATE
- FRAMEWORK_READY

Paper-ready phrasing
--------------------
> This stress test does not validate the framework universally; it defines its observational domain of applicability and identifies physically motivated out-of-domain systems.
