- **F3_SCM**: pendiente \(d\log_{10}(V_{\mathrm{obs}})/d\log_{10}(r)\) en la cola externa de la curva
  (\(r \ge 0.7\,R_{\max}\)).
- **delta_f3**: observable centrado definido como `F3_SCM - 0.5`.
- **beta**: pendiente \(d\log_{10}(g_{\mathrm{obs}})/d\log_{10}(g_{\mathrm{bar}})\) en régimen profundo
  (\(g_{\mathrm{bar}} < 0.3\,a_0\)).
- **logSigmaHI_out**: proxy ambiental reproducible usado por el pipeline, estimado como la mediana
  de \(\log_{10}(V_{\mathrm{gas}}^2/r)\) en la cola externa.
- **coef(logSigmaHI_out)**: coeficiente ambiental del ajuste lineal frente a `delta_f3` o `beta`.
- **Métricas OOS**: comparación fuera de muestra (70/30) entre modelo completo y baseline medio,
  reportando `R²` y `RMSE` en `results/results_overview.json`.
