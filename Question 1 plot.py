import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Set a seed for reproducibility
np.random.seed(42)

# Sample size
n = 2023

# Simulate variables
log_faminc = np.random.normal(10, 0.5, n)  # Log family income
fatheduc = np.random.normal(12, 2, n)  # Father's education in years
errors = np.random.normal(0, 0.1, n)  # Errors term

# True model: log(bweight) = β0 + β1*log(faminc) + β2*fatheduc + error
beta_0 = 2.0
beta_1 = 0.3
beta_2 = 0.1

# Simulate birth weight
log_bweight = beta_0 + beta_1 * log_faminc + beta_2 * fatheduc + errors

# Create a DataFrame
data = pd.DataFrame({
    'log_bweight': log_bweight,
    'log_faminc': log_faminc,
    'fatheduc': fatheduc
})

# Fit the correct model
model_true = smf.ols('log_bweight ~ log_faminc + fatheduc', data=data).fit()

# Fit the model with omitted variable
model_omitted = smf.ols('log_bweight ~ log_faminc', data=data).fit()

# Scatter plot with regression lines
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_faminc', y='log_bweight', data=data, label='Data', alpha=0.5)
sns.lineplot(x=data['log_faminc'], y=model_omitted.fittedvalues, color='red', label='Without fatheduc')
sns.lineplot(x=data['log_faminc'], y=model_true.fittedvalues, color='blue', label='With fatheduc')
plt.xlabel('Log Family Income')
plt.ylabel('Log Birth Weight')
plt.title('Regression Lines with and without Father\'s Education')
plt.legend()
plt.show()

# Residual plots with the same scale for y-axis
plt.figure(figsize=(12, 5))

# Determine the limits for the y-axis
y_min = min(min(model_omitted.resid), min(model_true.resid)) - 0.1
y_max = max(max(model_omitted.resid), max(model_true.resid)) + 0.1

plt.subplot(1, 2, 1)
plt.scatter(data['log_faminc'], model_omitted.resid, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Log Family Income')
plt.ylabel('Residuals')
plt.title('Residuals without Father\'s Education')
plt.ylim(y_min, y_max)

plt.subplot(1, 2, 2)
plt.scatter(data['log_faminc'], model_true.resid, alpha=0.5)
plt.axhline(0, color='blue', linestyle='--')
plt.xlabel('Log Family Income')
plt.ylabel('Residuals')
plt.title('Residuals with Father\'s Education')
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()
