import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm



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

# Question 1a: Omitted Variable Bias
# Fit Different Regression Models
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit the correct model
model_true = smf.ols('log_bweight ~ log_faminc + fatheduc', data=data).fit()
print(model_true.summary())


# Fit the model with omitted variable
model_omitted = smf.ols('log_bweight ~ log_faminc', data=data).fit()
print(model_omitted.summary())

with open('regression_results.txt', 'w') as f:
    f.write(model_omitted.summary().as_text())
    f.write(model_true.summary().as_text())


# Illustrate the Impact of Omitted Variable Bias
beta_1_true = model_true.params['log_faminc']
beta_1_omitted = model_omitted.params['log_faminc']
bias = beta_1_omitted - beta_1_true

print("Question 1a: Omitted Variable Bias")
print(f"True β1: {beta_1_true}")
print(f"β1 with omitted variable: {beta_1_omitted}")
print(f"Bias in β1: {bias}")

# Compare coefficients and other statistics
print('Coefficient for log_faminc without fatheduc:', model_omitted.params['log_faminc'])
print('Coefficient for log_faminc with fatheduc:', model_true.params['log_faminc'])
print('R-squared without fatheduc:', model_omitted.rsquared)
print('R-squared with fatheduc:', model_true.rsquared)

# Question 1b: Hypothesis Testing
# Hypothesis test
beta_1_hat = model_true.params['log_faminc']
beta_2_hat = model_true.params['fatheduc']
se_beta_1 = model_true.bse['log_faminc']
se_beta_2 = model_true.bse['fatheduc']

# Calculate covariance between beta_1_hat and beta_2_hat
cov_matrix = model_true.cov_params()
cov_beta_1_beta_2 = cov_matrix.loc['log_faminc', 'fatheduc']

# t statistic
t_stat = (0.2 * beta_1_hat - beta_2_hat) / np.sqrt(0.2**2 * se_beta_1**2 + se_beta_2**2 - 2 * 0.2 * cov_beta_1_beta_2)
print(f"t statistic: {t_stat}")

# Two-tailed test
p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
print("----- Question 1b: Hypothesis Testing -----")
print(f"p-value: {p_value:.10f}")

# Fit the model with robust standard errors
model_true_robust = smf.ols('log_bweight ~ log_faminc + fatheduc', data=data).fit(cov_type='HC3')
print(model_true_robust.summary())

# Compare the standard errors
se_comparison = pd.DataFrame({
    'Regular SE': model_true.bse,
    'Robust SE': model_true_robust.bse
})

print(se_comparison)


