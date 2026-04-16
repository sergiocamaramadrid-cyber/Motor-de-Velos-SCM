# SCM Spectral Multi-Scale Analysis (SPARC)

## Overview

This repository presents a data-driven spectral analysis of galaxy rotation curves using SPARC data.

The goal is to identify intrinsic spatial structures without assuming a predefined physical model.

---

## Method

For each galaxy:

1. Polynomial detrending (degree 2)
2. Residual extraction
3. Interpolation to uniform grid
4. Fast Fourier Transform (FFT)
5. Peak detection (85th percentile threshold)
6. Extraction of:
   - Number of peaks (n_peaks)
   - Dominant wavelength (λ)

---

## Dataset

- Initial sample: 19 galaxies
- Final clean sample: 13 galaxies

File:

data/scm_sparc_multiscale_13.csv

---

## Key Results

### Multi-scale structure

- λ range: ~1–20 kpc
- n_peaks range: 3–15

### Correlations

| Relation | Spearman ρ | p-value |
|--------|-----------|--------|
| λ vs Mass | 0.79 | 0.0013 |
| λ vs Mass (robust) | 0.68 | 0.021 |
| n_peaks vs Mass | ~0.00 | 0.99 |

---

## Interpretation

- The dominant spatial scale increases with baryonic mass
- Spectral complexity is independent of mass
- Rotation curves exhibit multi-scale structure

---

## Conclusion

Rotation curves cannot be described by a single characteristic scale.  
They contain structured multi-scale components linked to galaxy mass.

---

## Disclaimer

This is a data-driven exploratory analysis.  
No claim of new physics is made at this stage.

---

## Author

Sergio Cámara Madrid  
Independent Researcher

---

## License

MIT License
