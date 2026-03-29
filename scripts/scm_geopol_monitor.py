#!/usr/bin/env python3
"""
SCM Geoeconomic Monitor — v0.1

Calcula el estado del sistema SCM (F3, F3*, F3**)
y genera señales operativas a partir de indicadores clave.

Autor: Sergio Cámara Madrid (Framework SCM)
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# =========================
# CONFIGURACIÓN DE UMBRALES
# =========================

THRESHOLDS = {
    "PMI": {
        "amber": 51.5,
        "red": 50.0
    },
    "NPL": {
        "green": 2.5,
        "amber": 3.0
    },
    "CONSUMER_CONF": {
        "amber": -12,
        "red": -15
    },
    "ENERGY_MWH": {
        "amber": 120,
        "red": 130
    },
    "CREDIT_TIGHTENING": {
        "red": True
    },
    "GRID_LATENCY_MONTHS": {
        "red": 24
    }
}

# =========================
# FUNCIONES DE CLASIFICACIÓN
# =========================

def classify_pmi(value):
    if value < THRESHOLDS["PMI"]["red"]:
        return "RED"
    elif value < THRESHOLDS["PMI"]["amber"]:
        return "AMBER"
    else:
        return "GREEN"


def classify_npl(value):
    if value > THRESHOLDS["NPL"]["amber"]:
        return "RED"
    elif value > THRESHOLDS["NPL"]["green"]:
        return "AMBER"
    else:
        return "GREEN"


def classify_conf(value):
    if value < THRESHOLDS["CONSUMER_CONF"]["red"]:
        return "RED"
    elif value < THRESHOLDS["CONSUMER_CONF"]["amber"]:
        return "AMBER"
    else:
        return "GREEN"


def classify_energy(value):
    if value >= THRESHOLDS["ENERGY_MWH"]["red"]:
        return "RED"
    elif value >= THRESHOLDS["ENERGY_MWH"]["amber"]:
        return "AMBER"
    else:
        return "GREEN"


def classify_credit(is_tightening):
    return "RED" if is_tightening else "GREEN"


def classify_grid(latency):
    return "RED" if latency >= THRESHOLDS["GRID_LATENCY_MONTHS"]["red"] else "GREEN"


# =========================
# MOTOR PRINCIPAL SCM
# =========================

def evaluate_system(data):
    states = {}

    states["PMI"] = classify_pmi(data["PMI"])
    states["NPL"] = classify_npl(data["NPL"])
    states["CONF"] = classify_conf(data["CONF"])
    states["ENERGY"] = classify_energy(data["ENERGY"])
    states["CREDIT"] = classify_credit(data["CREDIT_TIGHTENING"])
    states["GRID"] = classify_grid(data["GRID_LATENCY"])

    red_count = sum(1 for v in states.values() if v == "RED")
    amber_count = sum(1 for v in states.values() if v == "AMBER")

    # =========================
    # REGÍMENES SCM
    # =========================

    if states["GRID"] == "RED":
        regime = "F3** (PHYSICAL CONSTRAINT)"
    elif states["ENERGY"] == "RED":
        regime = "F3* (FRICTION LIMIT)"
    elif red_count >= 3:
        regime = "F3 (RECESSIVE FRICTION)"
    else:
        regime = "NORMAL"

    return states, red_count, amber_count, regime


# =========================
# DECISIÓN SECTORIAL
# =========================

def sector_signal(regime, states):
    if regime == "F3** (PHYSICAL CONSTRAINT)":
        return {
            "BANKS": "HOLD",
            "DEFENSE": "OVERWEIGHT",
            "INFRA_AI": "EXIT",
            "INDUSTRY": "UNDERWEIGHT",
            "CONSUMER": "EXIT"
        }

    if regime == "F3* (FRICTION LIMIT)":
        return {
            "BANKS": "OVERWEIGHT",
            "DEFENSE": "OVERWEIGHT",
            "INFRA_AI": "WATCH",
            "INDUSTRY": "UNDERWEIGHT",
            "CONSUMER": "UNDERWEIGHT"
        }

    if regime == "F3 (RECESSIVE FRICTION)":
        return {
            "BANKS": "OVERWEIGHT",
            "DEFENSE": "HOLD",
            "INFRA_AI": "SELECTIVE",
            "INDUSTRY": "UNDERWEIGHT",
            "CONSUMER": "UNDERWEIGHT"
        }

    return {
        "BANKS": "NEUTRAL",
        "DEFENSE": "NEUTRAL",
        "INFRA_AI": "NEUTRAL",
        "INDUSTRY": "NEUTRAL",
        "CONSUMER": "NEUTRAL"
    }


# =========================
# EJECUCIÓN (BOTÓN ROJO)
# =========================

def run_scm(data):
    states, red_count, amber_count, regime = evaluate_system(data)
    signals = sector_signal(regime, states)

    print("\n=== SCM GEOPOLITICAL MONITOR ===")
    print("States:", states)
    print(f"Red: {red_count} | Amber: {amber_count}")
    print("Regime:", regime)
    print("\nSector Signals:")
    for k, v in signals.items():
        print(f"  {k}: {v}")

    result = {
        "states": states,
        "red_count": red_count,
        "amber_count": amber_count,
        "regime": regime,
        "signals": signals,
        "input_data": data,
    }

    outdir = Path("results/scm_geopol")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "scm_geopol_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # --- weekly history ---
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    history_record = {"timestamp": timestamp, **result}

    # JSONL — one record per line
    jsonl_path = outdir / "scm_geopol_history.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_record) + "\n")

    # CSV — flat row for easy tabular analysis
    csv_path = outdir / "scm_geopol_history.csv"
    csv_row = {
        "timestamp": timestamp,
        "regime": regime,
        "red_count": red_count,
        "amber_count": amber_count,
        **{f"state_{k}": v for k, v in states.items()},
        **{f"signal_{k}": v for k, v in signals.items()},
        **{f"input_{k}": v for k, v in data.items()},
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    return result


# =========================
# EJEMPLO (DATOS ACTUALES)
# =========================

if __name__ == "__main__":
    sample_data = {
        "PMI": 50.5,
        "NPL": 2.18,
        "CONF": -16.3,
        "ENERGY": 135,
        "CREDIT_TIGHTENING": True,
        "GRID_LATENCY": 18  # meses
    }

    run_scm(sample_data)
