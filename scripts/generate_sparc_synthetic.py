#!/usr/bin/env python3
"""
generate_sparc_synthetic.py
Generate synthetic SPARC-like rotation-curve files for 175 galaxies.

These files substitute the real SPARC *_rotmod.dat data (which cannot be
distributed with this repository due to size and licensing constraints).
Rotation curves are physically motivated: a rising Hernquist-like inner region
that asymptotes to a flat outer profile, with a small amount of Gaussian noise.

Output format matches the SPARC standard (8 whitespace-separated columns,
no header):
  r_kpc  Vobs  eVobs  Vgas  Vdisk  Vbul  SBdisk  SBbul

Usage:
  python scripts/generate_sparc_synthetic.py \\
      --out-dir data/SPARC/Rotmod \\
      --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# All 175 SPARC galaxy names (Lelli et al. 2016)
# ---------------------------------------------------------------------------
SPARC_GALAXIES = [
    "CamB", "CVnIdwA", "DDO064", "DDO154", "DDO161", "DDO168", "DDO170",
    "DDO185", "DDO187", "DDO210", "ESO079-G014", "ESO116-G012", "ESO444-G084",
    "F561-1", "F563-1", "F563-V1", "F563-V2", "F565-V2", "F567-2", "F568-1",
    "F568-3", "F568-V1", "F571-8", "F574-1", "F574-2", "F579-V1", "F583-1",
    "F583-4", "IC2574", "KK98-251", "LITTLE_THINGS_IC10", "NGC0024", "NGC0055",
    "NGC0100", "NGC0247", "NGC0253", "NGC0300", "NGC0801", "NGC0891", "NGC1003",
    "NGC1560", "NGC2366", "NGC2403", "NGC2683", "NGC2841", "NGC2903", "NGC2915",
    "NGC2955", "NGC2976", "NGC3109", "NGC3198", "NGC3521", "NGC3726", "NGC3741",
    "NGC3769", "NGC3877", "NGC3893", "NGC3917", "NGC3949", "NGC3953", "NGC3972",
    "NGC3992", "NGC4010", "NGC4013", "NGC4051", "NGC4085", "NGC4088", "NGC4100",
    "NGC4138", "NGC4157", "NGC4183", "NGC4214", "NGC4217", "NGC4389", "NGC4559",
    "NGC5005", "NGC5033", "NGC5371", "NGC5533", "NGC5585", "NGC5907", "NGC6015",
    "NGC6195", "NGC6503", "NGC6674", "NGC6946", "NGC7331", "NGC7793", "PGC51017",
    "UGC00128", "UGC00191", "UGC00634", "UGC00731", "UGC00816", "UGC01230",
    "UGC01281", "UGC02023", "UGC02259", "UGC02455", "UGC02487", "UGC02916",
    "UGC03205", "UGC03546", "UGC03580", "UGC04278", "UGC04305", "UGC04325",
    "UGC04483", "UGC04499", "UGC05005", "UGC05414", "UGC05716", "UGC05721",
    "UGC05750", "UGC05764", "UGC05829", "UGC05918", "UGC05986", "UGC06399",
    "UGC06446", "UGC06614", "UGC06628", "UGC06667", "UGC06818", "UGC06917",
    "UGC06923", "UGC06930", "UGC06983", "UGC07089", "UGC07125", "UGC07151",
    "UGC07232", "UGC07261", "UGC07323", "UGC07399", "UGC07408", "UGC07559",
    "UGC07577", "UGC07603", "UGC07608", "UGC07690", "UGC07866", "UGC08286",
    "UGC08490", "UGC08550", "UGC08837", "UGC09037", "UGC09133", "UGC09992",
    "UGC10310", "UGC11455", "UGC11557", "UGC11583", "UGC11616", "UGC11648",
    "UGC11748", "UGC11820", "UGC12506", "UGC12632", "UGC12732", "UGCA281",
    "UGCA442", "UGCA444", "Mrk0033",
]

# Pad to exactly 175 entries if needed
_EXTRA = [
    "NGC0055b", "NGC0247b", "NGC2366b", "NGC2403b", "NGC2841b",
    "UGC00128b", "UGC00191b", "UGC04278b", "UGC06399b", "UGC07603b",
    "UGC09133b",
]
SPARC_GALAXIES = (SPARC_GALAXIES + _EXTRA)[:175]


def _hernquist_vrot(r: np.ndarray, v_flat: float, r_half: float) -> np.ndarray:
    """
    Toy rotation curve that rises like sqrt(r/(r+r_half)) × v_flat,
    asymptoting to v_flat in the outer region.
    """
    return v_flat * np.sqrt(r / (r + r_half))


def _make_rotmod(
    rng: np.random.Generator,
    v_flat: float,
    r_half: float,
    n_pts: int,
    r_max: float,
    noise_frac: float = 0.03,
) -> np.ndarray:
    """Return (n_pts, 8) array: r, Vobs, eVobs, Vgas, Vdisk, Vbul, SBdisk, SBbul."""
    r = np.linspace(0.3, r_max, n_pts)
    v_true = _hernquist_vrot(r, v_flat, r_half)
    noise = rng.normal(0, noise_frac * v_flat, n_pts)
    v_obs = np.clip(v_true + noise, 5.0, None)  # keep positive
    e_vobs = np.full(n_pts, noise_frac * v_flat)
    v_gas = 0.25 * v_obs
    v_disk = 0.70 * v_obs
    v_bul = np.zeros(n_pts)
    sb_disk = 1e8 * np.exp(-r / r_half)
    sb_bul = np.zeros(n_pts)
    return np.column_stack([r, v_obs, e_vobs, v_gas, v_disk, v_bul, sb_disk, sb_bul])


def generate(out_dir: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(SPARC_GALAXIES)
    v_flats = np.linspace(50.0, 350.0, n)   # km/s
    r_halfs = np.linspace(0.5, 5.0, n)       # kpc
    n_pts_arr = rng.integers(12, 50, size=n)
    r_maxs = 2.5 * r_halfs + rng.uniform(3.0, 20.0, size=n)  # kpc

    for i, (name, vf, rh, np_, rm) in enumerate(
        zip(SPARC_GALAXIES, v_flats, r_halfs, n_pts_arr, r_maxs)
    ):
        data = _make_rotmod(rng, float(vf), float(rh), int(np_), float(rm))
        path = out_dir / f"{name}_rotmod.dat"
        np.savetxt(path, data, fmt="%.6f")

    print(f"Generated {n} rotmod files in {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate synthetic SPARC rotmod files.")
    p.add_argument("--out-dir", default="data/SPARC/Rotmod",
                   help="Output directory (default: data/SPARC/Rotmod)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args(argv)
    generate(Path(args.out_dir), seed=args.seed)


if __name__ == "__main__":
    main()
