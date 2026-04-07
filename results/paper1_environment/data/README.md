# results/paper1_environment/data

This directory contains the datasets for the environmental modulation analysis.

## Files

| File | N | Description |
|------|---|-------------|
| `scm_final_dataset_79.csv` | 79 | Full SPARC sample with F3, logM, e_env columns |
| `scm_final_subset_n10.csv` | 54 | Physical subset: galaxies with `n_tail_points >= 10` |

## Column reference

| Column | Description |
|--------|-------------|
| `galaxy_name` | SPARC galaxy identifier |
| `delta_f3` | F3 − 0.5 (deviation from flat rotation) |
| `logM` | log10(stellar + gas mass / M☉) |
| `e_env` | Environmental density proxy (Chae+2021) |
| `n_tail_points` | Number of data points in outer tail used for F3 fit |

## Provenance

- F3 catalog: derived from SPARC rotation curve fits (Lelli+2016)
- Environmental proxy: Chae et al. (2021), Table 1
- Merged via `scripts/build_env_real_input.py`
- Analysis via `scripts/analyze_env_real_merged.py`

> **Note:** Upload `scm_final_dataset_79.csv` and `scm_final_subset_n10.csv`
> from `SCM_RESULTS_FINAL/` (Drive) to this directory before creating the GitHub release.
