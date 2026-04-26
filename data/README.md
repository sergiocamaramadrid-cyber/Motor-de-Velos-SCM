# Data

Raw SPARC data are not redistributed in this repository.

Expected processed input columns:

```text
galaxy
logM
MHI
Rdisk
slope_tail
```

The proxy used by the current clean analysis is:

```text
env_new = log10(MHI / Rdisk^2)
env_std = z-score(env_new)
```

This is an **internal HI surface-density proxy**, not an external environment measurement.
