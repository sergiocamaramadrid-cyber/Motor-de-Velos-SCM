# results/paper1_environment — Environmental modulation results (SPARC)

Results for Paper 1: *Environmental modulation of outer rotation curve slopes
in SPARC galaxies using the SCM Framework.*

## Key result

Physical subset `n_tail_points >= 10` (N = 54):

| Parameter | Value |
|-----------|-------|
| β_env | −0.099 |
| p (HC3 robust) | 0.012 |
| Bootstrap CI₉₅ (BCa) | [−0.183, −0.023] |

## Directory structure

```
results/paper1_environment/
├── data/
│   ├── README.md
│   ├── scm_final_dataset_79.csv     ← upload from Drive: SCM_RESULTS_FINAL/
│   └── scm_final_subset_n10.csv     ← upload from Drive: SCM_RESULTS_FINAL/
├── figures/
│   ├── README.md
│   ├── figure_env_final.pdf         ← upload from Drive (rename scm_figure_env_n10.pdf)
│   ├── figure_env_final.png         ← upload from Drive (rename scm_figure_env_n10.png)
│   ├── figure_robustness_final.pdf  ← upload from Drive (rename scm_robustness_threshold.pdf)
│   └── figure_robustness_final.png  ← upload from Drive (rename scm_robustness_threshold.png)
├── tables/
│   └── scm_table_results.tex        ← LaTeX table (ready)
└── summary/
    └── scm_resumen_final.txt        ← narrative summary (ready)
```

## Reproducibility

All analysis scripts live in `scripts/`:

| Script | Role |
|--------|------|
| `scripts/build_env_real_input.py` | Merge F3 + SPARC + Chae catalogs |
| `scripts/analyze_env_real_merged.py` | OLS HC3 + bootstrap + permutation |
| `scripts/generate_env_figure.py` | Publication figure |

Run the full suite: `python3 -m pytest -q` → 284 passed.

## Release & DOI

Target release tag: `v1.0-environmental-results`

After creating the GitHub release, activate [Zenodo](https://zenodo.org/account/settings/github/)
to obtain a citable DOI. Add to paper:

> *Data and code available at Zenodo: DOI: XXXXX*
