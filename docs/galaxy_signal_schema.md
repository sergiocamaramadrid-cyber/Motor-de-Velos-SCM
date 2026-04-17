# Galaxy Signal Table — Schema Reference

*File:* `data/galaxy_signal_table.csv`  
*Built by:* `scripts/build_galaxy_signal_table.py`  
*Status:* **FROZEN** — do not modify without a justified physical change.

---

## Build command

```bash
python scripts/build_galaxy_signal_table.py \
    --sparc-dir data/SPARC \
    --out data/galaxy_signal_table.csv \
    --env-csv data/env_proxy.csv
```

The `--env-csv` flag is optional. When omitted, `env_proxy` is `NaN` for all rows.

---

## Output columns

| Column | Type | Units | Description |
|---|---|---|---|
| `galaxy` | string | — | Galaxy name (from SPARC catalog) |
| `logMbar` | float | log₁₀(M☉) | log10(0.5·L36 + 1.33·MHI); NaN if both absent |
| `Mgas` | float | 1e9 M☉ | Gas mass = 1.33 × MHI; NaN if MHI absent |
| `Rmax` | float | kpc | Maximum observed galactocentric radius |
| `Vmax` | float | km/s | Maximum observed circular velocity |
| `slope_tail` | float | — | Log-log slope of g_obs vs g_bar in outer regime; NaN if < 4 points |
| `delta_f3` | float | — | `slope_tail − 0.5`; NaN when slope_tail is NaN |
| `env_proxy` | float | — | Environmental density proxy; NaN if not provided |
| `width_kpc` | float | kpc | Approximate disk diameter = 2.5 × Rdisk; NaN if Rdisk absent |
| `thickness_kpc` | float | kpc | Approximate disk thickness = 0.1 × Rdisk; NaN if Rdisk absent |
| `outer_fit_ok` | bool | — | True when slope_tail is finite (≥ 4 outer points) |
| `n_tail_points` | int | — | Number of radial points used for the slope_tail fit |

---

## Frozen formulas

### Baryonic mass

```
logMbar = log10(0.5 · L36 + 1.33 · MHI)
```

- `L36` in 1e9 L☉ (SPARC column); stellar mass-to-light ratio fixed at **Υ = 0.5**
- `MHI` in 1e9 M☉ (SPARC column); He correction factor fixed at **1.33**
- If only one component is available the missing one is treated as zero
- If both are absent, `logMbar = NaN`

### Gas mass

```
Mgas = 1.33 · MHI   [1e9 M☉]
```

NaN when `MHI` is absent from the SPARC catalog.

### Outer-regime slope (`slope_tail`)

The outer regime is defined **purely geometrically**:

```
outer region: r >= 0.7 · Rmax
```

Within this region:

```
g_bar = V_bar² / r
V_bar² = upsilon_disk · V_disk² + upsilon_bulge · V_bul² + V_gas²
         with upsilon_disk = 1.0, upsilon_bulge = 1.0

g_obs = V_obs² / r
```

`slope_tail` is the OLS slope of `log10(g_obs)` vs `log10(g_bar)` over the outer points.

**Minimum points required:** 4. If fewer than 4 outer points satisfy the mask,
`slope_tail = NaN`, `outer_fit_ok = False`.

### delta_f3

```
delta_f3 = slope_tail − 0.5
```

The reference value **0.5** is the SCM/MOND deep-regime expectation (BETA_REF).

### Disk geometry

```
width_kpc     = 2.5 · Rdisk
thickness_kpc = 0.1 · Rdisk
```

`Rdisk` is the exponential disk scale radius from the SPARC catalog (kpc).  
Both are NaN when `Rdisk` is absent.

---

## NaN policy

| Field | Set to NaN when |
|---|---|
| `logMbar` | both `L36` and `MHI` absent, or computed mass ≤ 0 |
| `Mgas` | `MHI` absent from SPARC catalog |
| `slope_tail` | fewer than 4 points satisfy `r >= 0.7·Rmax` |
| `delta_f3` | `slope_tail` is NaN |
| `env_proxy` | `--env-csv` not provided, or galaxy not in env CSV |
| `width_kpc` | `Rdisk` absent from SPARC catalog |
| `thickness_kpc` | `Rdisk` absent from SPARC catalog |

Galaxies with no usable rotation-curve file (`{galaxy}_rotmod.dat`) are **omitted**
entirely from the output table.

---

## Expected SPARC directory layout

```
data/SPARC/
├── SPARC_Lelli2016c.csv      # galaxy summary table (or .mrt)
├── NGC0024_rotmod.dat        # per-galaxy rotation curves
├── NGC0055_rotmod.dat
└── ...
```

The `_rotmod.dat` files must contain whitespace-separated columns in this order
(no header, comment lines starting with `#` are ignored):

```
r   v_obs   v_obs_err   v_gas   v_disk   v_bul   SBdisk   SBbul
```

Units: `r` in kpc, velocities in km/s.

---

## env_proxy.csv contract

Minimum required columns:

```
galaxy,env_proxy
NGC0024,0.42
NGC0055,0.17
...
```

- `galaxy` must match the names in `SPARC_Lelli2016c.csv` exactly
- `env_proxy` is a continuous scalar (higher = denser environment)
- Galaxies in SPARC but absent from `env_proxy.csv` receive `env_proxy = NaN`

---

## Constants (frozen)

| Name | Value | Role |
|---|---|---|
| `BETA_REF` | 0.5 | Reference slope for `delta_f3` |
| `UPSILON_DEFAULT` | 0.5 | Stellar M/L for `logMbar` |
| `HE_CORRECTION` | 1.33 | HI → total gas correction |
| `OUTER_FRAC` | 0.7 | Fraction of Rmax defining the outer regime |
| `MIN_OUTER_POINTS` | 4 | Minimum points for a valid slope fit |
| `upsilon_disk` | 1.0 | Fixed disk M/L for g_bar in slope_tail |
| `upsilon_bulge` | 1.0 | Fixed bulge M/L for g_bar in slope_tail |
