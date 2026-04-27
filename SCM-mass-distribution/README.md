# SCM Framework — Mass & Distribution Result

This repository presents a reproducible result from the SCM (Motor de Velos) framework.

## Key Result

The outer structure of galaxies is not governed by a single global law.

Instead:

**Mass sets the scale, and mass distribution modulates the structure.**

slope_tail ≈ f(logMbar, Σ)

## Dataset

- Source: SPARC rotation curves (reconstructed from rotmod files)
- Valid galaxies: 146
- Final clean sample: 86

## Pipeline

1. Extract rotmod curves
2. Compute outer slope (r ≥ 0.7 Rmax)
3. Merge with mass catalog
4. Test global variables
5. Fit structural model:
   slope_tail ~ logMbar + Sigma_resid

## Main Result

- logMbar → highly significant
- Sigma_resid → significant
- R² ≈ 0.19

## Interpretation

The diversity in galaxy outer slopes is explained by:

- total baryonic mass
- how that mass is distributed

## Extension

This framework may extend to black hole jets:

jet_structure ≈ f(logMBH, energy distribution)

## Author

Sergio Cámara Madrid

## License

MIT
