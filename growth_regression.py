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

data = pd.read_csv('Results/final/results/simulated_complexity_2015_ISJ_True.csv')
data['pred_eci'] = (data['pred_eci'] - data['pred_eci'].mean()) / data['pred_eci'].std()
diversity = pd.read_csv('Results/final/data/country_diversity.csv')
diversity = diversity[diversity['year'] == 2015]
regression = pd.read_csv('Results/final/data/regression_dataset.csv').sort_values('ISO3')
regression_2005 = regression[regression['year'] == 2015].rename(columns={'ISO3':'country'})

regression = pd.read_csv('Results/final/data/regression_dataset.csv').sort_values('ISO3')
regression_2005 = regression[regression['year'] == 2015].rename(columns={'ISO3':'country'})

growth_data = regression[['ISO3', 'year', 'GDP per capita growth (annual %)']].rename(columns={'ISO3':'country', 'GDP per capita growth (annual %)': 'growth'}).dropna()
growth_data = growth_data[growth_data['year'].between(2016,2020)].groupby(['country']).mean().reset_index() 

# Make dataset
regression_2005 = regression_2005.merge(data[['country', 'pred_eci', 'pred_num_caps', 'actual_eci']], on=['country'], how='right')
regression_2005 = regression_2005.merge(growth_data[['country', 'growth']], on=['country'], how='right')
regression_2005 = regression_2005.merge(diversity[['diversity', 'country']], on=['country'], how='right')

regression_2005 = regression_2005.drop_duplicates(['country', 'year']).set_index(['country', 'year'])
regression_2005['log_gdp_pc'] = np.log(regression_2005['GDP per capita (constant 2015 US$)']+ 1e-9)

# Correlation (R-squared)

'''corr_df = regression_2005[['diversity', 'pred_num_caps', 'actual_eci', 'pred_eci', 'log_gdp_pc', 'exports_GDP',   'pop', 'inv_GDP', 'growth'
]].corr()
corr_df.to_csv('Results/final/results/corr_df_2015.csv')
'''
# Regression

regression_2005 = regression_2005[
        ['pred_eci', 'actual_eci', 'pop', 'pred_num_caps', 'inv_GDP',
        'exports_GDP',
       'log_gdp_pc',
       'diversity',
       'growth'
       ]
].dropna()

X = regression_2005[
        ['exports_GDP', 'inv_GDP', 'actual_eci', 'log_gdp_pc']
]
y = regression_2005['growth']

import statsmodels.api as sm
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit(cov_type='HC1')
print(model_sm.summary())