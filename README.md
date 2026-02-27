<p>
  <a href="https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM/actions/workflows/ci.yml">
    <img src="https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM/actions/workflows/ci.yml/badge.svg">
  </a>
  <img src="https://img.shields.io/badge/version-v0.6.1-blue">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-informational">
</p>

# Framework SCM – Motor de Velos
**Modelo de Condensación Fluida (SCM)** con pipeline reproducible, auditorías de estabilidad (VIF/κ), y herramientas de validación.

## Qué es
Este repositorio implementa el Framework SCM ("Motor de Velos") como un **pipeline científico auditable** para:
- Ejecutar análisis/ajuste sobre curvas de rotación.
- Generar artefactos de auditoría (VIF, κ, reportes).
- Ejecutar tests automáticamente (CI) para garantizar reproducibilidad.

## Estado (v0.6.1)
- ✅ Pipeline operacional con salida de auditoría.
- ✅ Diagnóstico estructural: VIF / condition number (κ).
- ✅ Integración Continua (GitHub Actions) ejecutando `pytest` en Python 3.10/3.11.
- ✅ Metadatos de citación (CITATION.cff) para "Cite this repository".

## Instalación rápida
```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecución

Ejemplo genérico (ajusta flags según tu CLI real):

```bash
python -m src.scm_analysis --data-dir data/sparc --outdir results/audit_run
```

Outputs esperados (si están habilitados en tu pipeline):

- `results/.../audit/vif_table.csv`
- `results/.../audit/stability_metrics.csv`
- `results/.../audit/quality_status.txt` (si el "quality reporting" está activo)

## Estructura del repo (orientativa)

```
.
├─ src/                  # núcleo del framework
├─ scripts/              # utilidades/auditorías
├─ tests/                # tests unitarios e integración
├─ data/                 # datos de entrada
├─ results/              # artefactos de validación
├─ docs/                 # documentación
└─ .github/workflows/    # CI
```

## Reproducibilidad

Ejecuta pytest localmente:

```bash
pytest -q
```

La CI en GitHub debe reflejar lo mismo (entorno limpio).

## Citar

Usa el botón "Cite this repository" de GitHub (se alimenta de CITATION.cff).

## Licencia

MIT License — ver archivo [`LICENSE`](LICENSE)

## Autor

Sergio Cámara Madrid  
Framework SCM – Motor de Velos