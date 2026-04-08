# Results catalog

## LITTLE THINGS sample catalog

**File:** `results/lt_sample_catalog.csv`

The LITTLE THINGS sample catalog contains 26 galaxies with available
environmental proxy measurements. We restrict the analysis to galaxies with
available environmental proxy measurements, resulting in N = 26.

### Column definitions

| Column | Description |
|--------|-------------|
| `galaxy` | Canonical galaxy name |
| `logM` | Baryonic mass proxy (log scale) |
| `delta_mass_std` | Standardized proxy derived from log_j (specific angular momentum), expressed as a z-score within the sample |
| `slope_tail` | Outer-disk logarithmic slope dlogV/dlogr, measured over the outermost reliable points available for each galaxy |
| `Rmax_kpc` | Outer radius in kpc when available (see note below) |
| `delta_f3` | `slope_tail − 0.5` — deviation from the reference flat-rotation slope, consistent with the SCM formalism |

### Notes

- **`delta_mass_std`** is a standardized proxy derived from `log_j` (specific
  angular momentum), expressed as a z-score within the sample.

- **`slope_tail`** is the outer-disk logarithmic slope dlogV/dlogr, measured
  over the outermost reliable points available for each galaxy. In LITTLE
  THINGS, rotation curves do not always reach a robust outer radius, so
  `slope_tail` is derived from the available `residual_btfr` BTFR measurement.

- **`delta_f3`** is defined as `slope_tail − 0.5`. The reference value 0.5
  corresponds to a flat rotation curve (dlogV/dlogr = 0.5 in the BTFR
  parameterization). `delta_f3 > 0` indicates a rising outer curve;
  `delta_f3 < 0` indicates a falling curve. In the LITTLE THINGS sample,
  `delta_f3` is constructed from the available outer-disk slope measurement and
  interpreted consistently with the SCM formalism.

- **`Rmax_kpc`** is available only for a subset of galaxies (4/26) and is not
  used in the primary statistical analysis.

### Generation

```bash
python scripts/build_sample_csv.py
```

Optional explicit paths:

```bash
python scripts/build_sample_csv.py \
  --lt-global data/little_things_global.csv \
  --predictions results/blind_test_lt/predictions.csv \
  --rot-dir data/raw/lt_oh2015 \
  --out results/lt_sample_catalog.csv
```
