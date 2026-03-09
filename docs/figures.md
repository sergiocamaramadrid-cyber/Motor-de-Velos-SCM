# Framework Figures

## Environmental Test

### ΔF3 vs HI Surface Density

This figure tests whether the outer rotation curve slope correlates with environmental gas density.

\[
\Delta F_3 = \frac{d \log V_{\mathrm{obs}}}{d \log r}
\]

A statistically significant slope would indicate that outer galaxy dynamics depend on environmental conditions.

Expected output file:

```text
results/fig_deltaF3_environment.png
```

Interpretation guide:

- `p < 0.01` → strong environmental signal
- `0.01 ≤ p < 0.05` → moderate evidence
- `p ≥ 0.05` → no significant evidence
