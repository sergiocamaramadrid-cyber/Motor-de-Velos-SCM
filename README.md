# SCM Framework – Motor de Velos

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19455777.svg)](https://doi.org/10.5281/zenodo.19455777)

## Overview

This repository provides a reproducible framework to test regime-dependent behavior in astrophysical systems.

The SCM framework does not aim to find a universal law, but to detect **regime transitions** in dynamical systems.

---

## Key Results

### Galaxies (SPARC)

- No strong global law (ΔAIC < 2)
- Presence of **mass-dependent regime transition**
- Strongest signal at:
  - logMbar < 9.7
  - ΔAIC ≈ 8.8
  - p_perm ≈ 0.043

### LITTLE THINGS

- Weak replication (ΔAIC ≈ 2)
- Transition not robust due to low N / dwarf regime

### Jets (MOJAVE)

- Clear transition:
  - r15 ≈ 13.8 pc
  - bootstrap support ≈ 92%
  - p_perm ≈ 0.09

---

## Main Conclusion

Astrophysical systems are not universally governed by a single dynamical law.

Instead:

- Systems with **low effective degrees of freedom** (jets) → clear transitions
- Systems with **multiple coupled components** (galaxies) → regime-dependent behavior

---

## Repository Structure

```
Motor-de-Velos-SCM/
│
├── data/
│   ├── raw/                  # SPARC, LITTLE THINGS originals
│   └── processed/            # processed CSV catalogs
│
├── results/
│   ├── sparc/
│   ├── little_things/
│   ├── mojave/
│   └── figures/
│
├── scripts/
│   ├── build_sparc_catalog.py
│   ├── run_crtt.py
│   ├── run_force_models.py
│   └── run_mojave_test.py
│
├── docs/
│   └── paper/
│
├── README.md
├── requirements.txt
└── CITATION.cff
```

---

## Installation

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducibility

Run the full pipeline:

```bash
python scripts/build_sparc_catalog.py
python scripts/run_crtt.py
python scripts/run_force_models.py
python scripts/run_mojave_test.py
```

Each script accepts `--help` for full options.

---

## Data Policy

Raw datasets (SPARC, LITTLE THINGS, MOJAVE) are **not versioned** in this repository.  
Download and preprocessing scripts are provided for reproducibility.  
See `docs/SPARC_EXPECTED_BEHAVIOUR.md` for the formal data contract.

---

## Citation

See `CITATION.cff` or the [Zenodo archive](https://doi.org/10.5281/zenodo.19455777).

---

## License

MIT. See LICENSE file.

---

## Author

Sergio Cámara Madrid  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM