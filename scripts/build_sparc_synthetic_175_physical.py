#!/usr/bin/env python3
from pathlib import Path

import numpy as np

OUT_DIR = Path("data/SPARC_synthetic/rotmod")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

N_GALAXIES = 175


def generate_galaxy(i):
    # Long radii -> clear outer tail
    r = np.array(
        [
            0.5,
            1.2,
            2.5,
            4.0,
            6.5,
            9.0,
            12.0,
            15.0,
            18.0,
            21.0,
            24.0,
            28.0,
        ]
    )

    # Physical parameters
    v_max = rng.uniform(90, 260)
    r_s = rng.uniform(1.5, 4.5)

    # Rise + transition
    v = v_max * (r / (r + r_s))

    # Outer tail (the key signal for the framework)
    outer = r >= 10
    mode = i % 3

    if mode == 0:
        # Slight decline
        v[outer] -= 0.45 * (r[outer] - 10)
    elif mode == 1:
        # Nearly flat
        v[outer] -= 0.15 * (r[outer] - 10)
    else:
        # Slight persistence
        v[outer] += 0.08 * (r[outer] - 10)

    # Small organic scatter
    v += rng.normal(0, 1.2, len(r))
    v = np.clip(v, 5, None)

    # Baryonic components
    v_gas = v * rng.uniform(0.12, 0.22)
    v_disk = v * rng.uniform(0.65, 0.80) * (r / (r + 1.2))
    v_bulge = v * rng.uniform(0.00, 0.15) * np.exp(-r / 2.2)

    # Observational uncertainties
    e_v = rng.uniform(2, 5, len(r))

    return np.column_stack([r, v, e_v, v_gas, v_disk, v_bulge])


def main():
    for i in range(1, N_GALAXIES + 1):
        data = generate_galaxy(i)
        name = f"G_SYNTH_{i:03d}"

        with open(OUT_DIR / f"{name}_rotmod.dat", "w") as f:
            for row in data:
                f.write(" ".join(f"{x:.6g}" for x in row) + "\n")

    print(f"✅ 175 physical synthetic galaxies generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
