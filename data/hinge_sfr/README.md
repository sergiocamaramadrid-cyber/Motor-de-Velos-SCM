# data/hinge_sfr/ — Sample input data for hinge_sfr_test.py

This directory contains 30 representative synthetic galaxies that demonstrate
the input format expected by `scripts/hinge_sfr_test.py`.  The galaxies span
baryonic masses from 10^8.5 to 10^11.1 M_sun in six mass groups, with varying
disk concentrations to generate realistic F1/F2/F3 proxy values.

**These are synthetic data for demonstration purposes only.**  Replace with
your real rotation-curve measurements before drawing scientific conclusions.

---

## profiles.csv — per radial point

One row per (galaxy, radial-point) pair.

| Column | Type | Units | Required | Description |
|--------|------|-------|----------|-------------|
| `galaxy` | str | — | ✅ | Galaxy identifier; must match `galaxy_table.csv` |
| `r_kpc` | float | **kpc** | ✅ | Galactocentric radius.  Typical range 0.1–100 kpc. |
| `vbar_kms` | float | **km/s** | ✅¹ | Baryonic rotation velocity.  Typical range 10–500 km/s. |
| `gbar_m_s2` | float | **m/s²** | ✅¹ | Baryonic centripetal acceleration.  Alternative to `vbar_kms`. |
| `rmax_kpc` | float | kpc | optional | Outermost measured radius.  Defaults to `max(r_kpc)` per galaxy. |

¹ Provide **either** `vbar_kms` **or** `gbar_m_s2`.  If both are present,
`gbar_m_s2` takes priority.

> **Unit check** — `compute_gbar_from_vbar()` emits a `UserWarning` when
> `r_kpc > 5000` (possible pc or m input) or `vbar_kms > 2000` (possible m/s
> input).  Fix units before proceeding.

---

## galaxy_table.csv — per galaxy

One row per galaxy.

| Column | Type | Units | Required | Description |
|--------|------|-------|----------|-------------|
| `galaxy` | str | — | ✅ | Galaxy identifier; must match `profiles.csv` |
| `log_mbar` | float | log10(M_sun) | ✅ | log baryonic mass |
| `log_sfr` | float | log10(M_sun/yr) | ✅¹ | log star-formation rate |
| `sfr` | float | M_sun/yr | ✅¹ | Linear SFR; converted to `log_sfr` automatically |
| `morph_bin` | str | — | optional | Morphology label (e.g. `"dwarf"`, `"late"`, `"early"`).  Used for matched-pair grouping and OLS dummies. |
| `quality_flag` | int/bool | — | optional | Filter rows before passing to the script (not used internally). |

¹ Provide **either** `log_sfr` **or** `sfr`.

---

## Running the test

```bash
python scripts/hinge_sfr_test.py \
  --profiles  data/hinge_sfr/profiles.csv \
  --galaxy-table data/hinge_sfr/galaxy_table.csv \
  --out results/hinge_sfr \
  --log-g0 -9.921   \
  --d 1.0
```

Replace `--log-g0` with `log10(g0)` from your SCM best-fit (default: `log10(1.2e-10) ≈ -9.921`).

Outputs are written to `results/hinge_sfr/`:

| File | Contents |
|------|---------|
| `hinge_features.csv` | Per-galaxy F1/F2/F3 values |
| `regression_F*.txt` | OLS summary (HC3 robust errors) |
| `permutation_F*.txt` | One-sided permutation p-value |
| `matched_pairs_F*.txt` | Wilcoxon matched-pairs result |
