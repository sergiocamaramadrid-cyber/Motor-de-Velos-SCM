# LITTLE THINGS Blind Test (SCM Framework)

This directory contains the results of a blind validation test of the SCM framework using a LITTLE THINGS galaxy sample.

## Overview

The blind test evaluates the predictive performance of two deterministic models:

- **BTFR-based prediction**
- **Interpolation-based SCM prediction**

No model parameters are fitted using this dataset. All predictions are generated using fixed analytical relations.

## Dataset

The input dataset is:

```
data/little_things_global.csv
```

This dataset is treated as **blind**, meaning it is not used for calibration or parameter tuning within this pipeline.

### Important note on repeated values

Some galaxies in the dataset share identical values of `logVobs`.  
As a result:

- Residuals in `predictions.csv` may appear repeated across multiple galaxies.
- This is expected behavior and **does not indicate a bug** in the prediction formulas.
- Additional repetition may arise from rounding outputs to 4 decimal places.

## Outputs

### `predictions.csv`

Per-galaxy predictions:

- `galaxy_id`
- `logVobs` (observed)
- `logV_btfr`, `logV_interp` (predicted)
- `residual_btfr`, `residual_interp`

### `summary.csv`

Aggregate metrics for each model:

- `RMSE_dex`
- `MAE`
- `bias`
- `improvement_frac`
- `wilcoxon_p`

## Reproducibility

The pipeline is fully deterministic:

- No randomness is used
- Running the script multiple times produces identical outputs

To reproduce:

```bash
python scripts/blind_test_little_things.py \
  --csv data/little_things_global.csv \
  --out results/blind_test_lt
```

## Data integrity

SHA256 checksum of the dataset:

```
SHA256: 46b3afa9f770929cf19421816d8a650bfd2bbcf6e3b93d3d3b93d402a2976960
```

This checksum ensures that the blind dataset has not been modified.

To verify locally:

```bash
sha256sum data/little_things_global.csv
```

## Scientific interpretation

This blind test evaluates whether SCM-based relations can predict galaxy kinematics without re-fitting parameters.

Consistent improvement over baseline models supports predictive validity.

Residual structure should be interpreted in light of dataset discretization and observational uncertainties.

---

**Note:** This test does not, by itself, guarantee absence of external calibration overlap.
The provenance and independence of the dataset must be documented at the manuscript level.