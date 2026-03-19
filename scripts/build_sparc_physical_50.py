#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np

OUT_DIR = Path("data/SPARC/rotmod")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 12345
N_GALAXIES = 50


def generate_physical_galaxy(i: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a synthetic SPARC-like rotation curve with:
    - inner rise
    - transition/knee
    - extended outer tail
    - small galaxy-to-galaxy diversity
    """

    # Long radial baseline with emphasis on outer structure
    r = np.array(
        [0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0, 24.0, 27.0],
        dtype=float,
    )

    # Galaxy-scale parameters
    v_max = rng.uniform(90.0, 260.0)
    r_s = rng.uniform(1.5, 4.5)

    # Inner+transition shape: saturating rise
    v_base = v_max * (r / (r + r_s))

    # Outer-tail behaviour:
    # small plateau / slight decline / slight persistence differences
    tail_mode = i % 3
    v = v_base.copy()

    outer = r >= 10.0
    if tail_mode == 0:
        # Slight decline
        v[outer] -= 0.45 * (r[outer] - 10.0)
    elif tail_mode == 1:
        # Near-flat plateau
        v[outer] -= 0.15 * (r[outer] - 10.0)
    else:
        # Very mild persistence / tiny positive support
        v[outer] += 0.08 * (r[outer] - 10.0)

    # Add small observational scatter
    v += rng.normal(0.0, 1.2, size=len(r))
    v = np.clip(v, 8.0, None)

    # Observational uncertainties
    err_v = rng.uniform(2.0, 4.0, size=len(r))

    # Baryonic components: approximate but structured
    # Gas becomes relatively more important outward
    gas_frac = rng.uniform(0.10, 0.22)
    disk_frac = rng.uniform(0.60, 0.82)
    bulge_frac = rng.uniform(0.00, 0.20)

    vgas = gas_frac * v * (1.0 - np.exp(-r / 4.0))
    vdisk = disk_frac * v * (r / (r + 1.2))
    vbul = bulge_frac * v * np.exp(-r / 2.2)

    # Small noise but keep positive
    vgas += rng.normal(0.0, 0.35, size=len(r))
    vdisk += rng.normal(0.0, 0.45, size=len(r))
    vbul += rng.normal(0.0, 0.25, size=len(r))

    vgas = np.clip(vgas, 0.0, None)
    vdisk = np.clip(vdisk, 0.0, None)
    vbul = np.clip(vbul, 0.0, None)

    return np.column_stack([r, v, err_v, vgas, vdisk, vbul])


def write_rotmod(path: Path, data: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(" ".join(f"{x:.6g}" for x in row) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED)

    for i in range(1, N_GALAXIES + 1):
        data = generate_physical_galaxy(i, rng)
        name = f"G_PHYSICAL_{i:03d}"
        write_rotmod(OUT_DIR / f"{name}_rotmod.dat", data)

    print(f"✅ MOTOR CEBADO: {N_GALAXIES} galaxias físicas con cola externa inyectadas.")


if __name__ == "__main__":
    main()
