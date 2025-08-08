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

data = pd.read_csv('Results/final/results/simulated_complexity_2000_ISJ_False.csv')
data['pred_eci'] = (data['pred_eci'] - data['pred_eci'].mean()) / data['pred_eci'].std()
diversity = pd.read_csv('Results/final/data/country_diversity.csv')
diversity = diversity[diversity['year'] == 2000]
regression = pd.read_csv('Results/final/data/regression_dataset.csv').sort_values('ISO3')
regression_year = regression[regression['year'] == 2000].rename(columns={'ISO3':'country'})

visualization_data = data.merge(regression_year[['GDP per capita (constant 2015 US$)', 'country']], on=['country'], how='right')
visualization_data = visualization_data.merge(diversity[['diversity', 'country']], on=['country'], how='right')
visualization_data['log_gdp_pc'] = np.log(visualization_data['GDP per capita (constant 2015 US$)'])
visualization_data = visualization_data.dropna()
print(visualization_data)

# Correlation heatmap

corr = pd.read_csv('Results/final/results/corr_df_2000.csv')
corr = corr.set_index('Unnamed: 0')
print(corr)
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(
    corr,
    annot=True,          # Annotate cells with the correlation values
    fmt=".2f",           # Format annotations to two decimal places
    cmap='coolwarm',     # Use a diverging colormap
    linewidths=.5,       # Add lines between cells
    vmin=-1, vmax=1      # Ensure the color scale is fixed from -1 to 1
)
names = [
    'Diversity',
    'Num. of capabilities', 
    'ECI (Method of Reflections)', 
    'Average capability complexity',
    'Log GDP per capita (USD)',
    'Export-GDP ratio',
    'Population',
    'Investment-GDP ratio',
    'Avg. GDP growth rate'
]
heatmap.set_xticklabels(names, rotation=45, ha='right')
heatmap.set_yticklabels(names, rotation=0)
plt.tight_layout()
plt.show()


# Correlation between measures and visualization

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(13, 9))
scatter = ax.scatter(
    visualization_data['actual_eci'],
    visualization_data['pred_eci'],
    s=np.sqrt(visualization_data['diversity']) * 15,  # Square root purely to avoid clutter
    c=visualization_data['log_gdp_pc'],         
    cmap='plasma',                
    alpha=0.7,
    edgecolor='k',
    linewidth=0.5
)
lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1]),
]

sns.regplot(
    x='actual_eci', 
    y='pred_eci', 
    data=visualization_data,
    ax=ax,
    scatter=False,  
    color='red',
    line_kws={'linestyle':'--', 'linewidth':1.5, 'label': 'Best Fit (OLS)'}
)
lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1]),
]
ax.plot(lims, lims, 'k-.', alpha=0.7, lw=1.5, label='Identity Line (y=x)')

ax.legend()

slope, intercept, r_value, p_value, std_err = stats.linregress(visualization_data['actual_eci'], visualization_data['pred_eci'])

ax.text(0.05, 0.95, f'$R^2 = {r_value ** 2:.2f}$',
        transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


countries_to_label = ['ARG', 'BRA', 'AGO']
texts = []
for code in countries_to_label:
    if code in visualization_data['country'].values:
        country_data = visualization_data[visualization_data['country'] == code].iloc[0]
        texts.append(
            ax.text(country_data['actual_eci'], country_data['pred_eci'], code, fontsize=15, color='red')
        )

countries_to_label = ['KOR', 'IND', 'BWA']
for code in countries_to_label:
    if code in visualization_data['country'].values:
        country_data = visualization_data[visualization_data['country'] == code].iloc[0]
        texts.append(
            ax.text(country_data['actual_eci'], country_data['pred_eci'], code, fontsize=15, color='darkgreen')
        )

countries_to_label = ['IRQ', 'SAU', 'IRN']
for code in countries_to_label:
    if code in visualization_data['country'].values:
        country_data = visualization_data[visualization_data['country'] == code].iloc[0]
        texts.append(
            ax.text(country_data['actual_eci'], country_data['pred_eci'], code, fontsize=15, color='orange')
        )

countries_to_label = ['USA', 'JPN', 'DEU']
for code in countries_to_label:
    if code in visualization_data['country'].values:
        country_data = visualization_data[visualization_data['country'] == code].iloc[0]
        texts.append(
            ax.text(country_data['actual_eci'], country_data['pred_eci'], code, fontsize=15, color='navy')
        )



adjust_text(texts,
        force_points=(0.5, 0.5),  # Increase repulsion from data points
            force_text=(0.75, 1.25),   # Increase repulsion between labels (especially vertically)
            
            arrowprops=dict(
                arrowstyle="-",  # A simple line, not an arrow
                color='gray',    # A subtle gray color
                lw=0.5,          # A thin line
                alpha=0.8
            ),

            expand_points=(1.5, 1.5), # Add a bit more padding around points
            expand_text=(1.2, 1.2),   # Add more padding around the text boxes
            lim=200                   # Increase the number of iterations to find a good solution
)
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Log GDP per capita (USD)', fontsize=12)
handles, labels = [], []
for div_val in [50, 200, 800, 2000]:
    handles.append(plt.scatter([], [], s=np.sqrt(div_val) * 15, c='gray', edgecolor='k'))
    labels.append(f'{div_val} products')
size_legend = ax.legend(handles, labels, title='Country Diversity',
                        loc='lower right', scatterpoints=1, 
                        frameon=True, labelspacing=1.5, title_fontsize=12, fontsize=10)

ax.add_artist(size_legend)

ax.legend(handles=[ax.get_lines()[0]], loc='upper left', fontsize=12)
ax.set_xlabel('ECI (Method of Reflections)', fontsize=14)
ax.set_ylabel('Average capability complexity', fontsize=14)

plt.tight_layout()
plt.show()

# Growth regression

'''regression = pd.read_csv('Results/final/data/regression_dataset.csv').sort_values('ISO3')
regression_2005 = regression[regression['year'] == 2005].rename(columns={'ISO3':'country'})

growth_data = regression[['ISO3', 'year', 'GDP per capita growth (annual %)']].rename(columns={'ISO3':'country', 'GDP per capita growth (annual %)': 'growth'})
growth_data = growth_data[growth_data['year'].between(2006,2025)] # 10-year average
growth_data = growth_data.groupby(['country']).mean().reset_index()
growth_data.to_csv('Results/final/growth.csv')

regression_2005 = regression_2005.merge(data[['country', 'pred_eci', 'pred_num_caps', 'actual_eci']], on=['country'], how='right')
regression_2005 = regression_2005.merge(growth_data[['country', 'growth']], on=['country'], how='right')
regression_2005 = regression_2005.drop_duplicates(['country', 'year']).set_index(['country', 'year'])
regression_2005['log_gdp_pc'] = np.log(regression_2005['GDP per capita (constant 2015 US$)']+ 1e-9)

regression_2005 = regression_2005[
        ['pred_eci', 'actual_eci', 'pop', 'pred_num_caps', 'inv_GDP',
        'exports_GDP',
        'Government Effectiveness: Estimate',
       'log_gdp_pc',
       'growth'
       ]
].dropna()

# Regression

X = regression_2005[
        ['exports_GDP', 'inv_GDP', 'pop', 'pred_eci', 'log_gdp_pc']
]
y = regression_2005['growth']

import statsmodels.api as sm
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit(cov_type='HC1')
print(model_sm.summary())'''