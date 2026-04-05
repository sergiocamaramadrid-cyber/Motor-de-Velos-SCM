"""
scripts/scm_fallback_modulation.py — β modulation by failed-SN fallback in SCM.

Computes the perturbation of the external SCM β parameter caused by a
failed-supernova (or partial-collapse) fallback event.  The modulation term
represents the ratio of the event's ram-pressure to the background IGM/CGM
kinetic-energy density, attenuated exponentially with the fallback radius.

Physical model
--------------
    β_mod = 0.5 + (E / (ρ · v² · r³)) · exp(−r / r_decay)

where all quantities are in CGS:

    E      — ejection energy [erg]
    ρ      — local gas density [g cm⁻³]
    v      — characteristic fallback velocity [cm s⁻¹]
    r      — fallback radius [cm]
    r_decay — exponential attenuation length [pc, kept in pc for the exponent]

The 0.5 offset is the MOND deep-regime β (Milgrom 1983), used as the
unperturbed baseline consistent with the rest of the Motor-de-Velos SCM
pipeline.

Reference scenario (M31-2014-DS1)
----------------------------------
    E = 1e46 erg, r = 0.1 pc, v = 50 km/s, ρ = 1e-24 g/cm³ (IGM)

    → modulation is ~1.35e4 at r = 0.1 pc, confirming that a failed SN
      completely dominates the local pressure balance.  The exponential
      factor suppresses the perturbation to negligible levels at galactic
      scales (r ≳ 50 pc for r_decay = 10 pc).

Usage
-----
    python scripts/scm_fallback_modulation.py \\
        --energy 1e32 --radius 5.0 --velocity 20.0 \\
        --density 1e-21 --decay-scale 10.0
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Unit conversion constants (CGS)
# ---------------------------------------------------------------------------

PC_TO_CM: float = 3.085677581e18   # cm per parsec (IAU 2012)
KMS_TO_CMS: float = 1.0e5         # cm/s per km/s

# Unperturbed MOND deep-regime β baseline
BETA_MOND: float = 0.5


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def fallback_modulation(
    energy_erg: float = 1e32,
    r_pc: float = 5.0,
    v_kms: float = 20.0,
    rho_gcm3: float = 1e-21,
    decay_scale_pc: float = 10.0,
) -> tuple[float, float, float]:
    """Compute β modulation from a failed-SN fallback event.

    All input parameters use physically natural units; internal computation
    is performed in CGS throughout.

    Parameters
    ----------
    energy_erg : float
        Ejection energy of the fallback event [erg].  Default 1e32 erg
        (sub-luminous transient regime).
    r_pc : float
        Characteristic fallback radius [pc].  Default 5.0 pc.
    v_kms : float
        Characteristic fallback velocity [km/s].  Default 20.0 km/s.
    rho_gcm3 : float
        Local gas density [g cm⁻³].  Default 1e-21 g/cm³ (CGM regime).
    decay_scale_pc : float
        Exponential attenuation length [pc].  Controls how quickly the
        perturbation decays with radius.  Default 10.0 pc.

    Returns
    -------
    beta_mod : float
        Modified β = BETA_MOND + modulation × decay.
    modulation : float
        Dimensionless pressure ratio E / (ρ v² r³) in CGS.
    decay : float
        Exponential attenuation factor exp(−r_pc / decay_scale_pc).

    Raises
    ------
    ValueError
        If any of energy_erg, r_pc, v_kms, rho_gcm3, or decay_scale_pc
        is non-positive.
    """
    if energy_erg <= 0:
        raise ValueError(f"energy_erg must be positive, got {energy_erg}")
    if r_pc <= 0:
        raise ValueError(f"r_pc must be positive, got {r_pc}")
    if v_kms <= 0:
        raise ValueError(f"v_kms must be positive, got {v_kms}")
    if rho_gcm3 <= 0:
        raise ValueError(f"rho_gcm3 must be positive, got {rho_gcm3}")
    if decay_scale_pc <= 0:
        raise ValueError(f"decay_scale_pc must be positive, got {decay_scale_pc}")

    r_cm: float = r_pc * PC_TO_CM
    v_cms: float = v_kms * KMS_TO_CMS

    modulation: float = energy_erg / (rho_gcm3 * v_cms**2 * r_cm**3)
    decay: float = float(np.exp(-r_pc / decay_scale_pc))
    beta_mod: float = BETA_MOND + modulation * decay

    return beta_mod, modulation, decay


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute SCM β modulation from a failed-SN fallback event.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--energy", type=float, default=1e32,
                   metavar="ERG",
                   help="Ejection energy [erg]")
    p.add_argument("--radius", type=float, default=5.0,
                   metavar="PC",
                   help="Fallback radius [pc]")
    p.add_argument("--velocity", type=float, default=20.0,
                   metavar="KMS",
                   help="Fallback velocity [km/s]")
    p.add_argument("--density", type=float, default=1e-21,
                   metavar="GCM3",
                   help="Local gas density [g/cm³]")
    p.add_argument("--decay-scale", type=float, default=10.0,
                   metavar="PC",
                   help="Exponential decay length [pc]")
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = _build_parser().parse_args(argv)

    beta_mod, modulation, decay = fallback_modulation(
        energy_erg=args.energy,
        r_pc=args.radius,
        v_kms=args.velocity,
        rho_gcm3=args.density,
        decay_scale_pc=args.decay_scale,
    )

    print(f"β modificado:  {beta_mod:.6f}")
    print(f"modulación:    {modulation:.4e}")
    print(f"decay factor:  {decay:.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])
