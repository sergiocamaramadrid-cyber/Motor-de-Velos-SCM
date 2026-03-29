#!/usr/bin/env python3
"""
SCM Geoeconomic Report Renderer — v0.1

Reads scm_geopol_summary.json and generates a LaTeX weekly report.

Usage:
    python scripts/scm_geopol_monitor.py          # produces JSON
    python scripts/render_scm_geopol_report.py    # produces .tex
    pdflatex -output-directory=results/scm_geopol results/scm_geopol/scm_geopol_report.tex
"""

import json
from pathlib import Path
from datetime import datetime

TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{array}

\title{SCM Geoeconomic Weekly Report}
\author{Framework SCM}
\date{{REPORT_DATE}}

\begin{document}
\maketitle

\section*{Executive Summary}
Current regime: \textbf{{REGIME}}.

System state:
\begin{itemize}
  \item Red indicators: \textbf{{RED_COUNT}}
  \item Amber indicators: \textbf{{AMBER_COUNT}}
  \item Banking stability remains the last systemic anchor while energy and credit conditions constrain investment.
\end{itemize}

\section*{Indicator States}
\begin{longtable}{>{\bfseries}p{4cm}p{4cm}p{5cm}}
\toprule
Indicator & State & Input value \\
\midrule
PMI & {PMI_STATE} & {PMI_VALUE} \\
NPL & {NPL_STATE} & {NPL_VALUE} \\
Consumer Confidence & {CONF_STATE} & {CONF_VALUE} \\
Energy & {ENERGY_STATE} & {ENERGY_VALUE} \\
Credit Tightening & {CREDIT_STATE} & {CREDIT_VALUE} \\
Grid Latency & {GRID_STATE} & {GRID_VALUE} \\
\bottomrule
\end{longtable}

\section*{Sector Signals}
\begin{longtable}{>{\bfseries}p{5cm}p{6cm}}
\toprule
Sector & Signal \\
\midrule
Banks & {BANKS} \\
Defense & {DEFENSE} \\
Infra-AI & {INFRA_AI} \\
Industry & {INDUSTRY} \\
Consumer & {CONSUMER} \\
\bottomrule
\end{longtable}

\section*{Interpretation}
The current weekly output places the system in regime \textbf{{REGIME}}. This implies that the dominant constraint is no longer only macro-financial; the cost structure of investment is beginning to invalidate deployment economics in selected sectors.

\section*{Operational Rule}
\begin{quote}
If energy remains in red and credit conditions stay restrictive while NPL remains green, maintain overweight in banks and defense, keep infra-AI under watch, and underweight industry and consumer exposure.
\end{quote}

\end{document}
"""


def main():
    input_path = Path("results/scm_geopol/scm_geopol_summary.json")
    output_dir = Path("results/scm_geopol")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tex = TEMPLATE
    replacements = {
        "REPORT_DATE": datetime.now().strftime("%Y-%m-%d"),
        "REGIME": data["regime"],
        "RED_COUNT": str(data["red_count"]),
        "AMBER_COUNT": str(data["amber_count"]),
        "PMI_STATE": data["states"]["PMI"],
        "NPL_STATE": data["states"]["NPL"],
        "CONF_STATE": data["states"]["CONF"],
        "ENERGY_STATE": data["states"]["ENERGY"],
        "CREDIT_STATE": data["states"]["CREDIT"],
        "GRID_STATE": data["states"]["GRID"],
        "PMI_VALUE": str(data["input_data"]["PMI"]),
        "NPL_VALUE": str(data["input_data"]["NPL"]),
        "CONF_VALUE": str(data["input_data"]["CONF"]),
        "ENERGY_VALUE": f'{data["input_data"]["ENERGY"]} EUR/MWh',
        "CREDIT_VALUE": str(data["input_data"]["CREDIT_TIGHTENING"]),
        "GRID_VALUE": f'{data["input_data"]["GRID_LATENCY"]} months',
        "BANKS": data["signals"]["BANKS"],
        "DEFENSE": data["signals"]["DEFENSE"],
        "INFRA_AI": data["signals"]["INFRA_AI"],
        "INDUSTRY": data["signals"]["INDUSTRY"],
        "CONSUMER": data["signals"]["CONSUMER"],
    }

    for key, value in replacements.items():
        tex = tex.replace("{" + key + "}", value)

    out_tex = output_dir / "scm_geopol_report.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Written: {out_tex}")


if __name__ == "__main__":
    main()
