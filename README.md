<img src="https://img.shields.io/badge/status-active-brightgreen">

# SCM-Galaxy-Dynamics

Empirical analysis of outer galaxy rotation-curve slopes within the SCM framework.

## Current status

This repository studies correlations between outer rotation-curve structure and internal baryonic properties of galaxies.

The current clean result is:

- `slope_tail` is measured from the outer rotation curve.
- A physically defined internal proxy is used:

```text
env_std = z-score[ log10(MHI / Rdisk^2) ]
```

This quantity should be interpreted as an **internal HI surface-density proxy**, not as an external environment measurement.

## Scope

This repository does not claim a confirmed external environmental mechanism.

Current interpretation:

> Internal HI surface density correlates with outer rotation-curve slope.

## Data

The analysis uses curated SPARC-derived catalogues.

Raw SPARC data are not redistributed here. Users should obtain them from the [official SPARC source](http://astroweb.cwru.edu/SPARC/).

## Quick test (example data)

```bash
python scripts/run_galaxy_dynamics.py --data data/example/example_dataset.csv
```

## Reproducibility

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/run_galaxy_dynamics.py --data data/your_dataset.csv
```

Expected input columns:

```text
galaxy
logM
MHI
Rdisk
slope_tail
```

## Output

The script reports:

- sample size
- high-mass sample size
- OLS-HC3 regression: `slope_tail ~ env_std + logM`

## Generate figure and table

```bash
python scripts/make_galaxy_figure_table.py --data data/your_dataset.csv --outdir results
```

Outputs:

- `results/fig_envstd_slope_tail.png`
- `results/table_ols_hc3.csv`

### Caption paper

```latex
\begin{figure}
\centering
\includegraphics[width=\columnwidth]{fig_envstd_slope_tail.png}
\caption{Outer rotation-curve slope as a function of the standardized internal HI surface-density proxy, defined as $z[\log_{10}(M_{\rm HI}/R_{\rm disk}^2)]$, for the high-mass subsample.}
\label{fig:envstd_slope_tail}
\end{figure}
```

## Scientific caution

Previous exploratory variables such as `delta_mass_std` are not treated as final physical definitions unless reconstructed from documented base quantities.

This repository prioritizes reproducibility and conservative interpretation.
