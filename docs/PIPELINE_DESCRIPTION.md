# Pipeline description

Pipeline completo recomendado (`scripts/run_full_framework_pipeline.py`):

1. Construir catálogo maestro SPARC.
2. Validar contrato/calidad de datos.
3. Ejecutar `deep_slope_test.py`.
4. Ajustar regresión F3 (`fit_f3_linear_regression.py`).
5. Ejecutar validación OOS.
6. Generar figuras de análisis F3.
7. Guardar resumen final en `results/framework_summary.json`.
