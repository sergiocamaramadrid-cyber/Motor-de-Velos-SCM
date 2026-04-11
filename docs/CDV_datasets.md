# SCM–Motor de Velos Framework · Dataset Snapshot

I'm sharing a full snapshot of the CSV datasets from the SCM–Motor de Velos Framework.

The idea is simple:
data should be accessible, reproducible, and open.

If you want to explore, validate, or challenge the results, everything is here.

---

## 📂 Available datasets

### 📁 `data/` (input data)

- `data/little_things_global.csv` → LITTLE THINGS global catalog
- `data/raw/lt_masses.csv` → Stellar/gas masses
- `data/raw/lt_metals.csv` → Metallicities
- `data/raw/cigan2021_tdust.csv` → Dust temperatures (Cigan+2021)
- Rotation curves (Oh+2015):
  - `DDO69_rot.csv`
  - `DDO70_rot.csv`
  - `DDO75_rot.csv`
  - `DDO210_rot.csv`

---

### 📁 `results/` (pipeline outputs)

- `results/per_galaxy_summary.csv` → Per-galaxy summary
- `results/f3_catalog_synthetic_flat.csv` → Synthetic F3 catalog
- `results/universal_term_comparison_full.csv` → Model comparison
- `results/blind_test_lt/` → Blind test predictions + summary
- `results/lt_dust_hinge/lt_hinge_dust_results.csv` → Dust analysis
- `results/diagnostics/compare_nu_models_175/compare_nu_models.csv`
- `results/diagnostics/deep_slope_test/deep_slope_test.csv`

---

## 🔗 Repository

👉 https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM

Raw file access:
👉 `https://raw.githubusercontent.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM/main/<file-path>`

---

## 🧠 Notes

Additional datasets such as:
- `galaxy_catalog_with_env.csv`
- `mw_cepheids.csv`

are generated during the pipeline execution and may not be stored directly in the repository.

---

## 🟢 Philosophy

> «Knowledge should be open.
> Not locked behind subscriptions.»

This work has been built with the help of multiple AI systems:
Gemini, ChatGPT, Grok, DeepSeek

and is shared in the same spirit.

---

If you use the data, test it.
If you test it, try to break it.
That's how this becomes real science.
