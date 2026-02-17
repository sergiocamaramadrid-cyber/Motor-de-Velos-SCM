"""
Analysis output script for Motor-de-Velos-SCM project.
This script demonstrates the required output formats for OOS dictionary and sensitivity analysis.
"""

import pandas as pd
import numpy as np


def create_oos_dict():
    """
    Create out-of-sample (OOS) dictionary with RMSE comparison data.
    
    Returns:
        dict: Dictionary with numeric keys containing DataFrames with RMSE metrics
    """
    # Create sample data for demonstration
    # In production, this would be populated from actual analysis results
    np.random.seed(42)
    
    oos_dict = {}
    
    # Create DataFrame for key 350
    n_samples = 100
    oos_dict[350] = pd.DataFrame({
        'delta_rmse_out': np.random.randn(n_samples) * 0.5 + 0.1,
        'delta_rmse_vs_scm': np.random.randn(n_samples) * 0.3 + 0.05,
        # Additional columns that might be present
        'model_id': range(n_samples),
        'iteration': np.random.randint(1, 10, n_samples)
    })
    
    return oos_dict


def create_sens_df():
    """
    Create sensitivity analysis DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with sensitivity analysis metrics
    """
    # Create sample sensitivity analysis data
    # In production, this would be populated from actual sensitivity analysis
    sens_df = pd.DataFrame({
        'thresh_kpc': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'n_close_pairs': [150, 120, 95, 75, 50, 30],
        'n_valid': [140, 115, 90, 72, 48, 28],
        'med_delta_rmse': [0.045, 0.052, 0.048, 0.055, 0.061, 0.058],
        'med_delta_rmse_vs_scm': [0.032, 0.038, 0.035, 0.042, 0.048, 0.045],
        'wilcoxon_p_one_less': [0.023, 0.015, 0.031, 0.019, 0.008, 0.012],
        'max_vif': [2.3, 2.5, 2.8, 3.1, 3.4, 3.7]
    })
    
    return sens_df


def main():
    """
    Main function to execute the required analysis outputs.
    """
    # Create the data structures
    oos_dict = create_oos_dict()
    sens_df = create_sens_df()
    
    # Required output 1: OOS dictionary statistics
    print("=" * 80)
    print("OOS Dictionary Analysis - Key 350")
    print("=" * 80)
    print(oos_dict[350][["delta_rmse_out", "delta_rmse_vs_scm"]].describe())
    print()
    
    # Required output 2: Sensitivity analysis table
    print("=" * 80)
    print("Sensitivity Analysis Results")
    print("=" * 80)
    print(sens_df[["thresh_kpc", "n_close_pairs", "n_valid", "med_delta_rmse",
                   "med_delta_rmse_vs_scm", "wilcoxon_p_one_less", "max_vif"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
