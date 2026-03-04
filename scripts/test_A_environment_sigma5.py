# Test A — Diferencial de entorno (SCM): F3 vs Σ5 (proxy densidad local) 

# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# Simulated data for F3 and Sigma5 environments
F3_data = np.random.normal(loc=0.5, scale=0.1, size=30)
Sigma5_data = np.random.normal(loc=0.6, scale=0.1, size=30)

# Create a DataFrame for the data
results = pd.DataFrame({'F3': F3_data, 'Sigma5': Sigma5_data})

# Calculate differences
results['diff'] = results['Sigma5'] - results['F3']

# Perform Wilcoxon signed-rank test
statistic, p_value = wilcoxon(results['F3'], results['Sigma5'])

# Output the results
print('Wilcoxon test statistic:', statistic)
print('P-value:', p_value)