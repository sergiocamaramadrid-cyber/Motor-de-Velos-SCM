# Motor-de-Velos-SCM

## ⚠️ IMPORTANTE: Advertencia sobre Datos

**Los scripts actuales usan DATOS SINTÉTICOS por defecto.** Ver [DATOS_SINTETICOS.md](DATOS_SINTETICOS.md) para detalles completos.

Para análisis con datos reales, proporciona `df_master.csv` y ejecuta `verify_data_quality.py` primero.

---

Resumen
-------
Motor-de-Velos-SCM implementa modelos y análisis de SCM (Supply Chain / causalidad según contexto del proyecto) usados para el trabajo asociado al artículo en `paper/manuscript.tex`. Contiene código para generar análisis principales, sensibilidad y notebooks de validación usando el dataset SPARC (instrucciones de descarga en `data/`).

Características principales
--------------------------
- Modelos y utilidades en `src/`:
  - `scm_models.py` — definición de modelos y funciones principales.
  - `scm_analysis.py` — pipeline/funciones para ejecutar análisis.
  - `sensitivity.py` — análisis de sensibilidad.
- Notebook de validación: `notebooks/SPARC_validation.ipynb`
- Resultados ejemplo (salida del análisis): `results/`
- Recursos para el artículo en `paper/`

Tabla de contenidos
------------------
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Datos (SPARC)](#datos-sparc)
- [Uso rápido](#uso-rápido)
- [Reproducir resultados](#reproducir-resultados)
- [Desarrollo y testing](#desarrollo-y-testing)
- [Contribuir](#contribuir)
- [Citación y licencia](#citación-y-licencia)
- [Contacto](#contacto)

Requisitos
----------
- Python 3.9+ (recomiendo 3.10+)
- Dependencias listadas en `requirements.txt`

Instalación
-----------
1. Clonar el repositorio:
   ```
   git clone <url-del-repo>
   cd Motor-de-Velos-SCM
   ```
2. Crear y activar entorno virtual:
   - Unix / macOS:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows:
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```
3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

Datos (SPARC)
------------
- Las instrucciones para descargar SPARC están en la carpeta `data/`.
- Recomendación de estructura local:
  ```
  data/SPARC/
  ├── raw/
  └── processed/
  ```
- Para reproducir fácilmente, define una variable de entorno (opcional):
  - Unix/macOS:
    ```
    export SPARC_DATA_DIR=$(pwd)/data/SPARC
    ```
  - Windows PowerShell:
    ```
    $env:SPARC_DATA_DIR = (Resolve-Path .\data\SPARC).Path
    ```

Uso rápido
----------
A continuación pasos generales; adapta las rutas/flags según tus scripts:

1. Descargar y colocar los datos SPARC en `data/SPARC` (ver `data/`).
2. Ejecutar análisis principal (ejemplo):
   ```
   python -m src.scm_analysis --data-dir data/SPARC --out results/
   ```
   Si `scm_analysis.py` no expone CLI, ejecuta el script directamente o modifica sus variables de entrada en la parte superior del archivo.
3. Ejecutar análisis de sensibilidad:
   ```
   python -m src.sensitivity --data-dir data/SPARC --out results/sensitivity/
   ```
4. Abrir/ejecutar el notebook de validación:
   ```
   jupyter lab notebooks/SPARC_validation.ipynb
   ```
   o ejecutar en bloque para reproducir:
   ```
   jupyter nbconvert --to notebook --execute notebooks/SPARC_validation.ipynb --ExecutePreprocessor.timeout=600 --output results/SPARC_validation_executed.ipynb
   ```

Reproducir resultados
---------------------
- Los resultados generados por el pipeline se guardan en `results/`. Los archivos clave incluyen:
  - `universal_term_comparison_full.csv`
  - `executive_summary.txt`
  - `top10_universal.tex`
- Para regenerar las figuras usadas en el artículo:
  1. Ejecutar los scripts/analyses necesarios (ver sección Uso rápido).
  2. Ejecutar el notebook o los scripts de visualización que generan `paper/figures`.

Desarrollo y testing
--------------------
- Recomiendo añadir tests con `pytest` en un directorio `tests/`.
- Ejecutar tests:
  ```
  pytest -q
  ```
- Linting / type checking sugerido:
  ```
  flake8 src/ tests/
  mypy src/
  ```
- Para integrar CI (GitHub Actions): añadir workflow que ejecute `pytest` y `flake8` en PR.

Sugerencias de mejora (pendientes)
---------------------------------
- Añadir un script automático para descargar y verificar SPARC (`scripts/download_sparc.py`).
- Añadir CLI formal con `argparse` o `click` para `scm_analysis.py` y `sensitivity.py`.
- Convertir notebook a scripts reproducibles ejecutables en CI.
- Añadir tests unitarios para funciones críticas de `src/`.

Contribuir
----------
1. Hacer fork y crear una rama feature/bugfix.
2. Añadir tests para nuevo comportamiento.
3. Abrir PR con descripción clara y referencia a issue si aplica.

Citación y licencia
-------------------
- Consulta `CITATION.cff` para la forma recomendada de citar este proyecto.
- Licencia en `LICENSE`. Incluye información sobre permisos y limitaciones.

Contacto
--------
- Autor / mantainer: revisa `CITATION.cff` o el historial de commits para los correos/usuarios.
- Para dudas o PRs, abre un issue en el repositorio.

Notas finales
-------------
- Si quieres, puedo:
  - Subir este README al repo y abrir un PR (necesito owner/repo).
  - Traducir al inglés.
  - Crear un issue para rastrear la incorporación del README y otras tareas relacionadas.