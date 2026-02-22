import pandas as pd

def _pick(dataframe, cols):
    """Pick specific columns from a DataFrame."""
    return dataframe[cols]

def export_structural_residuals_v01():
    """Export structural residuals to a CSV file after merging two datasets."""
    # Load the data
    df_master = pd.read_csv('scm_results_final/diagnostics/df_master.csv')
    df_metrics = pd.read_csv('scm_results_final/oos_validation/per_galaxy_metrics.csv')

    # Merging dataframes (adjust as needed for the actual merge logic)
    df_merged = pd.merge(df_master, df_metrics, on='common_column', how='inner')  # Replace 'common_column' with the actual column

    # Save the merged dataframe to CSV
    df_merged.to_csv('scm_results_final/diagnostics/structural_residuals_v01.csv', index=False)

if __name__ == "__main__":
    export_structural_residuals_v01()