# Usage Guide

This guide demonstrates how to use the analysis output functionality.

## Running the Analysis Script

To execute the analysis and see both required outputs:

```bash
# From the project root directory
python src/analysis_output.py
```

This will display:
1. OOS dictionary statistics for key 350 (delta_rmse metrics)
2. Sensitivity analysis results table

## Using the Jupyter Notebook

For interactive analysis:

```bash
# Launch Jupyter
jupyter notebook notebooks/analysis_output.ipynb

# Or with JupyterLab
jupyter lab notebooks/analysis_output.ipynb
```

## Expected Outputs

### Output 1: OOS Dictionary Analysis
Shows descriptive statistics (count, mean, std, min, quartiles, max) for:
- `delta_rmse_out`: Out-of-sample RMSE differences
- `delta_rmse_vs_scm`: RMSE differences compared to SCM model

### Output 2: Sensitivity Analysis
Displays a table with columns:
- `thresh_kpc`: Threshold value in kpc units
- `n_close_pairs`: Number of close pairs
- `n_valid`: Number of valid samples
- `med_delta_rmse`: Median delta RMSE
- `med_delta_rmse_vs_scm`: Median delta RMSE vs SCM
- `wilcoxon_p_one_less`: Wilcoxon test p-value (one-sided, less)
- `max_vif`: Maximum Variance Inflation Factor

## Customization

The script uses sample data for demonstration. In production:
1. Replace the `create_oos_dict()` function to load actual OOS analysis results
2. Replace the `create_sens_df()` function to load actual sensitivity analysis results
3. Modify the data generation logic to match your specific analysis pipeline

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scipy matplotlib seaborn jupyter
```
