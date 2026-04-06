#!/usr/bin/env python3
"""
scripts/plot_scm_v5_results.py
SCM‑Motor de Velos V5: 3-panel summary figure

Generates a publication-quality 3-panel plot showing 20 years of
interstellar surfing:
  - Panel 1: distance from the Sun (AU)
  - Panel 2: probe velocity vs solar-wind speed (km/s)
  - Panel 3: net power P_net (W) on a symmetric-log scale

Data can be supplied in two ways:
  1. From the real V5 ODE simulation (via ``--simulate``).  The script
     calls ``scripts.scm_motor_v5_interstellar.run_simulation`` with
     ``--t-days 7300`` (20 years) and converts the output.
  2. Via compact analytic approximations built into ``build_data()``
     (the default, used when ``--simulate`` is not given).  These
     reproduce the qualitative shape reported in the V5 problem statement
     (~680 UA, ~318 km/s terminal, P_net > 0) without requiring the
     10–30 s integration time of the full ODE.

Usage
-----
Quick preview with analytic approximation::

    python scripts/plot_scm_v5_results.py

From real ODE simulation::

    python scripts/plot_scm_v5_results.py --simulate

Save to file::

    python scripts/plot_scm_v5_results.py --out results/v5/scm_v5_results.png

Bugs fixed vs the problem-statement snippet:
  1. ``plt.show()`` was called inside the ``plot_results()`` function, which
     blocks in non-interactive / CI environments.  Moved out of
     ``plot_results()`` and ``main()``.  ``main()`` never calls ``show()``;
     the ``__main__`` block calls it only when the script runs interactively.
  2. No ``main()`` entry-point → not importable in tests. Added
     ``main(argv=None)`` that returns the figure.
  3. Heliopause annotation positioned at a hard-coded y value that depends
     on the scale. Changed to a fraction of the y-axis range.
  4. ``P_net_W`` clipping via ``np.maximum`` left an abrupt discontinuity.
     Replaced with a smooth analytic form that never goes below −24 W.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T_YEARS_DEFAULT = 20.0
_N_POINTS_DEFAULT = 1000
_HELIOPAUSE_LABEL_AU = 120.0   # nominal heliopause position for annotation

# ---------------------------------------------------------------------------
# Analytic approximation of V5 ODE results
# ---------------------------------------------------------------------------

def build_data(
    t_years_total: float = _T_YEARS_DEFAULT,
    n_points: int = _N_POINTS_DEFAULT,
) -> dict:
    """Return smooth analytic approximations of the V5 ODE results.

    The expressions are calibrated so that the terminal values (at 20 yr)
    approximately match the V5 reported outcomes:
      · r ≈ 680 AU,  v ≈ 318 km/s,  P_net > 0 throughout.

    Returns
    -------
    dict with keys: t_years, r_AU, v_kms, v_sw_kms, P_net_W
    """
    t = np.linspace(0.0, t_years_total, n_points)

    # Distance: approximately 680 AU at t = 20 yr
    r_AU = 1.0 + 3.09 * t ** 1.8

    # Probe velocity: starts at 30 km/s, asymptotes toward ~318 km/s
    v_kms = 30.0 + 288.0 * (1.0 - np.exp(-t / 3.5))

    # Solar-wind speed reference (qualitative oscillation)
    v_sw_kms = np.where(
        t < 8.0,
        400.0 - 80.0 * np.sin(2.0 * np.pi * t / 0.5),
        25.0 * np.ones_like(t),        # ISM: ~25 km/s beyond heliopause
    )

    # Net power: large initial surplus, decays to a small positive plateau
    P_net_W = (
        1.4e6 * np.exp(-t / 1.5)           # initial solar-wind power burst
        + 50.0 * np.ones_like(t)           # ISM plateau (always > 0)
    )

    return {
        "t_years": t,
        "r_AU": r_AU,
        "v_kms": v_kms,
        "v_sw_kms": v_sw_kms,
        "P_net_W": P_net_W,
    }


def build_data_from_simulation(
    t_days: float = 7300.0,
    n_steps: int = 10_000,
) -> dict:
    """Run the real V5 ODE and convert output to the plot dict format."""
    from scripts.scm_motor_v5_interstellar import run_simulation  # type: ignore[import]

    res = run_simulation(t_total_days=t_days, n_steps=n_steps)

    # v_sw_kms: use the v_wind_kms array that run_simulation already exposes
    return {
        "t_years": res["t_days"] / 365.25,
        "r_AU": res["r_AU"],
        "v_kms": res["v_kms"],
        "v_sw_kms": res["v_wind_kms"],
        "P_net_W": res["P_net"],
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_results(data: dict, out_path: str | Path | None = None) -> plt.Figure:
    """Create the 3-panel summary figure.

    Parameters
    ----------
    data:
        Dict with keys ``t_years``, ``r_AU``, ``v_kms``, ``v_sw_kms``,
        ``P_net_W``.
    out_path:
        If given, save the figure to this path before returning.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = data["t_years"]
    r_AU = data["r_AU"]
    v_kms = data["v_kms"]
    v_sw_kms = data["v_sw_kms"]
    P_net_W = data["P_net_W"]

    # Estimate time at which the probe crosses the nominal heliopause
    t_hp = t[np.argmin(np.abs(r_AU - _HELIOPAUSE_LABEL_AU))]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # --- Panel 1: distance ---
    ax1.plot(t, r_AU, "b-", lw=2)
    ax1.axvline(t_hp, color="gray", linestyle="--", alpha=0.5)
    r_max = float(np.max(r_AU))
    ax1.text(
        t_hp + 0.3,
        r_max * 0.05,
        "Heliopausa",
        rotation=90,
        fontsize=9,
        va="bottom",
    )
    ax1.set_ylabel("Distancia (UA)")
    ax1.grid(True)

    # --- Panel 2: velocities ---
    ax2.plot(t, v_kms, "r-", lw=2, label="Sonda")
    ax2.plot(t, v_sw_kms, "k--", lw=1, alpha=0.6, label="Viento solar / ISM")
    ax2.set_ylabel("Velocidad (km/s)")
    ax2.legend()
    ax2.grid(True)

    # --- Panel 3: net power ---
    ax3.plot(t, P_net_W, "purple", lw=2)
    ax3.axhline(0, color="gray", linestyle="--")
    ax3.set_ylabel("Potencia neta (W)")
    ax3.set_xlabel("Tiempo (años)")
    # symlog only makes sense when there are negative values; fall back to
    # plain log if everything is positive (avoids matplotlib warnings).
    p_min = float(np.min(P_net_W))
    if p_min < 0:
        ax3.set_yscale("symlog", linthresh=100)
    else:
        ax3.set_yscale("log")
    ax3.grid(True)

    plt.suptitle("SCM‑Motor de Velos V5: 20 años de surf interestelar")
    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> plt.Figure:
    """Entry-point. Returns the figure (does not call ``plt.show()``)."""
    parser = argparse.ArgumentParser(
        description="SCM V5 results — 3-panel summary figure."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use the real V5 ODE simulation instead of analytic approximation.",
    )
    parser.add_argument(
        "--t-days",
        type=float,
        default=7300.0,
        help="Simulation duration in days (only used with --simulate).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10_000,
        help="ODE evaluation steps (only used with --simulate).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=_N_POINTS_DEFAULT,
        help="Number of analytic-approximation points (ignored with --simulate).",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    if args.simulate:
        data = build_data_from_simulation(
            t_days=args.t_days, n_steps=args.n_steps
        )
    else:
        data = build_data(n_points=args.n_points)

    fig = plot_results(data, out_path=args.out)
    if args.out:
        print(f"Figura guardada en {args.out}")
    return fig


if __name__ == "__main__":
    # main() does not call plt.show() so it is safe to call from tests.
    # When run directly from the command line, show() is appropriate.
    fig = main()
    plt.show()
    plt.close(fig)
