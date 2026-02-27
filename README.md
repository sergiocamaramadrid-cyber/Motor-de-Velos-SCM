# Framework SCM – Motor de Velos

> Modelo de Condensación Fluida aplicado a curvas de rotación galácticas  
> Versión actual: **v0.6.1**

[![CI](https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM/actions/workflows/ci.yml/badge.svg)](https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-v0.6.1-blue)](https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Descripción

El Framework SCM (Motor de Velos) es un sistema computacional diseñado para modelar curvas de rotación galácticas mediante un enfoque de presión efectiva calibrada.

El framework implementa:

- Modelo SCM calibrado empíricamente
- Clasificador de regímenes de presión
- Detección automática de inyectores de presión
- Pipeline reproducible y auditable
- Validación estadística multigalaxia

---

## Características principales

- Pipeline reproducible completo
- Validación estadística automatizada
- Integración continua (CI)
- Arquitectura modular
- Reportes estructurados en JSON y CSV
- Sistema de auditoría integrado

---

## Instalación

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
pip install -r requirements.txt
```

---

## Uso básico

```bash
python -m src.scm_analysis \
  --data-dir data/sparc \
  --outdir results/
```

Con detección automática de inyectores:

```bash
python -m src.scm_analysis \
  --custom-data data/galaxy.txt \
  --detect-pressure-injectors \
  --audit-mode high-pressure
```

---

## Estructura del proyecto

```
Motor-de-Velos-SCM/
│
├── src/
├── tests/
├── audits/
├── reports/
├── data/
├── assets/
├── docs/
│
├── README.md
├── CITATION.cff
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Estado del proyecto

Estado actual: **Estable**

Validado en:

- Catálogo SPARC
- Grupo Local
- Grupo M81

Pipeline reproducible y verificado.

---

## Aplicaciones

Este framework puede aplicarse en:

- análisis de curvas de rotación
- modelado físico computacional
- investigación científica
- análisis de sistemas dinámicos complejos

También es relevante para sectores como:

- aeroespacial
- simulación física
- HPC
- modelado matemático avanzado

---

## Citación

Ver archivo: `CITATION.cff`

---

## Licencia

MIT License — ver archivo [`LICENSE`](LICENSE)

---

## Autor

Sergio Cámara Madrid  
Framework SCM – Motor de Velos