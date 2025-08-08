import pandas as pd
import scipy.stats as stats
import numpy as np
from linearmodels.panel import PanelOLS
from linearmodels import OLS
from linearmodels.panel import RandomEffects
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('Results/final/results/simulated_complexity_2015_ISJ_True.csv').astype({'best_rho': 'str', 'best_nu': 'f4'})
print(data.describe())
data['rho_dummy'] = data['best_rho'].map({'leontief': 0, '-9': 1, '-3': 2, '0': 3, '1': 4})
data['nu_dummy'] = data['best_nu'].map({0.5: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4})
data['pred_eci'] = (data['pred_eci'] - data['pred_eci'].mean()) / data['pred_eci'].std()
regression = pd.read_csv('Results/final/data/regression_dataset.csv').sort_values('ISO3')
regression_2005 = regression[regression['year'] == 2015].rename(columns={'ISO3':'country'})

regression_2005 = regression_2005.merge(data[['country', 'rho_dummy', 'nu_dummy', 'actual_eci', 'pred_eci']], on=['country'], how='right')

regression_2005 = regression_2005.drop_duplicates(['country', 'year']).set_index(['country', 'year'])
regression_2005['log_gdp_pc'] = np.log(regression_2005['GDP per capita (constant 2015 US$)']+ 1e-9)
regression_2005['log_gdp_sq'] = (regression_2005['log_gdp_pc'] - regression_2005['log_gdp_pc'].mean()) ** 2
regression_2005['pred_eci_sq'] = (regression_2005['pred_eci'] - regression_2005['pred_eci'].mean()) ** 2
regression_2005['actual_eci_sq'] = (regression_2005['actual_eci'] - regression_2005['actual_eci'].mean()) ** 2

regression_2005 = regression_2005[['log_gdp_pc', 'pop', 'rho_dummy', 'nu_dummy', 'inv_GDP', 'exports_GDP', 'actual_eci', 'pred_eci', 'log_gdp_sq', 'pred_eci_sq', 'actual_eci_sq']].dropna()
X = regression_2005[
        ['log_gdp_pc', 'log_gdp_sq', 'pop',  'inv_GDP', 'exports_GDP']]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("VIF:", dict(zip(X.columns, vif)))  # Values >5-10 indicate issues

y = regression_2005['nu_dummy']

model_logit = OrderedModel(y, X, distr='logit')  # For probit, use distr='probit'
result_logit = model_logit.fit(method='bfgs')  # Use optimization method

# View results
print(result_logit.summary())

print('Pseudo R-squared: {}'.format(result_logit.prsquared))
print('LLR: {}'.format(result_logit.llr))
print('LLR p-value: {}'.format(result_logit.llr_pvalue))