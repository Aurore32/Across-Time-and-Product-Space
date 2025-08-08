import pandas as pd
from ecomplexity import ecomplexity, proximity

data = pd.read_csv('Results/data/product_aggregated_exports.csv', low_memory=False)
data = data[['exporter','product','value','year']]
# Calculate complexity

for year in [2000,2005,2010,2015]:
    year_data = data[data['year'] == year]
    trade_cols = {'time': 'year', 'loc':'exporter', 'prod':'product', 'val':'value'}
    eci_df = proximity(year_data, trade_cols)
    eci_df.to_csv('Results/final/data/product_space_{}.csv'.format(year))   