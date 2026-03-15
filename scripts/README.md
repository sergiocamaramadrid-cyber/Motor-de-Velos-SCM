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
