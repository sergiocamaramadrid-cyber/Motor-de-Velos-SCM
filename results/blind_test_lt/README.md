## LITTLE THINGS blind dataset note

The file `data/little_things_global.csv` is used as the blind validation dataset for the LITTLE THINGS experiment. The prediction pipeline is deterministic and does not perform any training or parameter fitting on this dataset.

Some galaxies share identical `logVobs` values in the packaged CSV. As a result, and because `predictions.csv` is exported with rounded values (4 decimal places), repeated residuals may appear in `results/blind_test_lt/predictions.csv`. This behaviour is expected and does not indicate a bug in the interpolation or BTFR prediction formulas.

For reproducibility, record the SHA256 checksum of the packaged dataset below. To compute the checksum locally run:

```bash
sha256sum data/little_things_global.csv
```

Then replace the placeholder below with the resulting hash.

`SHA256: <INSERT_HASH_HERE>`

---

Notes and recommended follow-ups

- This file documents why repeated residual values can occur: duplicates in the observed velocities combined with rounding of predictions. No changes to the prediction formulas are required.
- If you want to increase numeric differentiation in the outputs, consider exporting predictions with higher precision (e.g. 6 decimal places) or keeping internal full precision in the saved CSV; this is only a presentation change and not necessary for correctness.
- Consider adding the computed SHA256 to the project README or a central DATA.md to make the blind dataset provenance explicit for reviewers.

---

Interpretation (short):

> Repeated residual values in `predictions.csv` are expected for some galaxies because the input blind dataset contains repeated `logVobs` values, and the exported predictions are rounded to 4 decimal places. This does not indicate a bug in the prediction formulas.