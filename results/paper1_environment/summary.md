# Environmental Modulation — Summary (SCM Framework)

## Global sample (N = 79)
- Spearman rho = -0.365
- p-value = 9.3e-04

## Mass split

### Low mass (logM < 10.6)
- rho ~ -0.15
- p ~ 0.36
- No significant correlation

### High mass (logM >= 10.6)
- rho ~ -0.49
- p ~ 0.001
- Strong negative correlation

## Key result

The environmental signal is not universal but emerges in the high-mass regime.

## Interpretation

This supports a scenario where outer disk dynamics are modulated by environment,
but only above a characteristic mass threshold (log M ~ 10.6 solar masses).

## Reproducibility

All results are reproducible via:

```bash
python scripts/run_environment_analysis.py \
    --input data/galaxy_catalog_with_env.csv \
    --output results/paper1_environment/tables/summary_results.csv
```

Full statistical tables: `results/paper1_environment/tables/summary_results.csv`

Related analysis (three-shield test):
- `docs/paper1/SCM_TR_results.md`
- `results/main/scm_tr_summary.csv`
- `results/yang/scm_tr_yang_dataset.csv`
- `results/gaia/mw_radial_scan.csv`
