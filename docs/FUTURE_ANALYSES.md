### External environmental proxy: Yang group catalog

A cross-match with the Yang et al. SDSS group catalog can be used to obtain an alternative environmental proxy, `delta_mass_yang`, for robustness tests. This quantity is **not identical** to the canonical \(\delta_{\rm mass}\) adopted in the main paper.

- If a density-like column is available in the external catalog (e.g. `DELTA_3MPC`), it is used directly.
- Otherwise, an exploratory pseudo-overdensity can be derived from `NGROUP` or `MGROUP` by normalizing to the matched-sample median.

This proxy is intended for robustness and exploratory analyses only. It should not replace the canonical \(\delta_{\rm mass}\) definition used in the manuscript without explicit redefinition.
