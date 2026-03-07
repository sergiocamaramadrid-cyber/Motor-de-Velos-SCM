SPARC Dataset (Lelli, McGaugh & Schombert 2016)

This directory contains the data products used by the SCM–Motor de Velos framework.

Source:
http://astroweb.case.edu/SPARC/

Required metadata tables (place under `data/SPARC/metadata/`)

SPARC_Lelli2016c.mrt
    Global galaxy properties.

CDR_Lelli2016b.mrt
    Circular velocity parameters (Vmax, Vflat, Rmax).

BTFR_Lelli2019.mrt
    Baryonic Tully-Fisher relation dataset.

MassModels_Lelli2016c.mrt
    Baryonic mass models and rotation curve decomposition.
    Used for radial analyses; validated as part of SPARC metadata completeness.

Expected local structure:

data/
└── SPARC
    ├── metadata
    │   ├── SPARC_Lelli2016c.mrt
    │   ├── CDR_Lelli2016b.mrt
    │   ├── BTFR_Lelli2019.mrt
    │   └── MassModels_Lelli2016c.mrt
    └── README_DATA.md

These data are used within the framework to compute:

- F3_SCM (outer rotation curve slope)
- β (deep regime slope)
- Out-of-sample validation tests of the SCM framework.

Original references

Lelli et al. 2016, AJ, 152, 157  
Lelli et al. 2019, MNRAS, 484, 3267
