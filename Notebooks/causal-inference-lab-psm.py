# %% Imports
import pandas as pd
import numpy as np
np.float = float  # Patch for deprecated np.float used by some older packages
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm 
from scipy.stats import ttest_ind

# %% Load data 
# Read the dataset
file_path = 'C:/Users/kanellop/Downloads/lalonde.csv'
lalonde = pd.read_csv(file_path)

# Basic data checks
# (rows, columns)
print(lalonde.shape)
# Data types of each column
print(lalonde.dtypes)
# Summary stats of numeric columns
print(lalonde.describe())     

# %% Check covariate balance before matching
# Define t-test function for continuous variables
def perform_ttest(group1, group2):
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_val

# Pre-treatment covariates to compare
pre_treatment_vars = ['age', 'educ', 're74', 're75']

# Separate treated and control groups
treated = lalonde[lalonde['treat'] == 1]
control = lalonde[lalonde['treat'] == 0]

# Run t-tests
ttest_results = []
for var in pre_treatment_vars:
    t_stat, p_val = perform_ttest(treated[var], control[var])
    ttest_results.append({'Variable': var, 'T-Statistic': round(t_stat, 2), 'P-Value': f"{p_val:.3f}"})

ttest_results_df = pd.DataFrame(ttest_results)
# Age and earnings before treatment differ significantly — suggests selection bias.
ttest_results_df

# %% Check binary variable balance with z-tests
from statsmodels.stats.proportion import proportions_ztest

# Define z-test function
def perform_ztest(count1, nobs1, count2, nobs2):
    z_stat, p_val = proportions_ztest([count1, count2], [nobs1, nobs2])
    return z_stat, p_val

# Binary covariates
binary_vars = ['hispan', 'black', 'married']

# Run z-tests
ztest_results = []
for var in binary_vars:
    count_treated = treated[var].sum()
    count_control = control[var].sum()
    nobs_treated = treated[var].count()
    nobs_control = control[var].count()
    z_stat, p_val = perform_ztest(count_treated, nobs_treated, count_control, nobs_control)
    ztest_results.append({'Variable': var, 'Z-Statistic': round(z_stat, 2), 'P-Value': f"{p_val:.3f}"})

ztest_results = pd.DataFrame(ztest_results)
# Binary characteristics also differ significantly pre-matching
ztest_results

# %% Run naive regression (no matching)
reg_unmatched = sm.OLS.from_formula('re78 ~ treat + age + educ + married + hispan + black + re74 + re75', data=lalonde).fit()
print(reg_unmatched.summary())

# %% Propensity score matching
# Define covariates to use for propensity score estimation
covariates = ['age', 'educ', 'black', 'hispan', 'married', 're74', 're75']
X = lalonde[covariates]
y = lalonde['treat']

# Step 1: Estimate propensity scores using logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
lalonde['pscore'] = model.predict_proba(X)[:, 1]

# Step 2: Match each treated unit to nearest control (1-to-1)
treated = lalonde[lalonde['treat'] == 1]
control = lalonde[lalonde['treat'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['pscore']])
distances, indices = nn.kneighbors(treated[['pscore']])

# Step 3: Apply caliper (max distance threshold)
caliper = 0.05
keep = distances.flatten() <= caliper

matched_treated = treated[keep]
matched_control = control.iloc[indices.flatten()[keep]]
matched_control.index = matched_treated.index  # Align for merging

# Combine matched data
matched = pd.concat([matched_treated, matched_control])
print("Matched sample size:", matched.shape[0])

# %% Check covariate balance after matching
def check_balance(df1, df2, covs):
    for var in covs:
        t, p = ttest_ind(df1[var], df2[var])
        print(f"{var:>10s}: p = {p:.3f}")

print("\nBalance BEFORE matching:")
check_balance(treated, control, covariates)

print("\nBalance AFTER matching:")
check_balance(matched_treated, matched_control, covariates)

# %% Run regression on matched sample
reg_matched = sm.OLS.from_formula('re78 ~ treat + age + educ + married + hispan + black + re74 + re75', data=matched).fit()
print(reg_matched.summary())
print(reg_unmatched.summary())

# %% Clean up all user-defined variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]