# Deep Regime Test

El test de régimen profundo valida la pendiente log-log esperada en el límite profundo.

Script principal:

```bash
python scripts/deep_slope_test.py --csv results/universal_term_comparison_full.csv --out results
```

Salida principal:

- `results/deep_slope_test.csv`

Interpretación general:

- Pendiente cercana al valor esperado => consistencia con el régimen profundo.
- Desviaciones sistemáticas => posible cambio de régimen o problemas de calidad de datos.
