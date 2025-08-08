import pandas as pd
import igraph as ig
import leidenalg
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import powerlaw
import scipy.stats as stats

mpl.rcParams['font.family'] = 'Arial'


categories = ['Animal products', 'Vegetable products', 'Fats', 'Prepared foodstuffs and nicotine', 'Mineral products', 'Chemical products',
              'Rubber and plastic', 'Leather and hides', 'Wood products', 'Textiles', 'Footwear', 'Stone-based products', 'Precious stones and pearls',
              'Metal products', 'Machinery', 'Transportation', 'Cinematography and optical products', 'Arms and ammunition',  'Miscellaneous manufactured products',
              'Fine art']


import numpy as np
from scipy.optimize import minimize

def code_to_chapter(code):
    if 1 <= code <= 5:
        return 'Animal products'
    elif 6 <= code <= 14:
        return 'Vegetable products'
    elif code == 15:
        return 'Fats'
    elif 16 <= code <= 24:
        return 'Prepared foodstuffs and nicotine'
    elif 25 <= code <= 27:
        return 'Mineral products'
    elif 28 <= code <= 38:
        return 'Chemical products'
    elif 39 <= code <= 40:
        return 'Rubber and plastic'
    elif 41 <= code <= 43:
        return 'Leather and hides'
    elif 44 <= code <= 49:
        return 'Wood products'
    elif 50 <= code <= 63:
        return 'Textiles'
    elif 64 <= code <= 67:
        return 'Footwear'
    elif 68 <= code <= 70:
        return 'Stone-based products'
    elif code == 71:
        return 'Precious stones and pearls'
    elif 72 <= code <= 83:
        return 'Metal products'
    elif 84 <= code <= 85:
        return 'Machinery'
    elif 86 <= code <= 89:
        return 'Transportation'
    elif 90 <= code <= 92:
        return 'Cinematography and optical products'
    elif code == 93:
        return 'Arms and ammunition'
    elif 94 <= code <= 96:
        return 'Miscellaneous manufactured products'
    elif code == 97:
        return 'Fine art'
    else:
        return 'Unclassified'


start = time.time()
product_space = pd.read_csv('Results/final/data/product_space_2015.csv', 
                           usecols=['product_1', 'product_2', 'proximity']).astype({'product_1': 'str', 'product_2': 'str'})
print('Number of products: {}'.format(len(product_space['product_1'].unique())))

product_space['product_1'] = ['0' + code if len(code) == 5 else code for code in product_space['product_1']]
product_space['product_2'] = ['0' + code if len(code) == 5 else code for code in product_space['product_2']]

products = pd.read_csv('Results/final/data/product_codes_HS92_V202501.csv', dtype='str')
pci_data = pd.read_csv('Results/final/data/HS92_pci.csv').astype({'product_code': 'str'})
pci_data['product_code'] = ['0' + code if len(code) == 5 else code for code in pci_data['product_code']]

pci_data = pci_data[pci_data['year'] == 2005].drop_duplicates('product_code')

print(f"Data loaded in {time.time()-start:.2f}s")

unique_products = product_space['product_1'].unique()
print(unique_products)
code_dict = {p: i for p, i in zip(pci_data['product_name'], pci_data['product_code'])}
pci_dict = {p: pci for p, pci in zip(pci_data['product_code'], pci_data['pci'])}
n_nodes = len(unique_products)

'''MIN_PROXIMITY = 0.1  # Adjust based on your data distribution
filtered_edges = product_space[(product_space['proximity'] >= MIN_PROXIMITY)]
filtered_edges: np.array = product_space.pivot(columns='product_1', index='product_2', values='proximity').values
np.fill_diagonal(filtered_edges, 0)

edges = []
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        edges.append((i, j, filtered_edges[i,j]))'''

all_products = pd.unique(product_space['product_1'])
product_to_idx = {name: idx for idx, name in enumerate(all_products)}
n_nodes = len(product_to_idx)

MIN_PROXIMITY = 0
filtered = product_space[
    (product_space['proximity'] > 0) &
    (product_space['product_1'] != product_space['product_2'])
]

edges = []
diffs = []
weights = []
seen = set()  

for _, row in filtered.iterrows():
    u = product_to_idx[row['product_1']]
    v = product_to_idx[row['product_2']]

    key = (min(u, v), max(u, v))
    
    if key not in seen:
        seen.add(key)
        edges.append((u, v))
        weights.append(row['proximity'])
        diffs.append(abs(pci_dict[row['product_1']] - pci_dict[row['product_2']]))

G = ig.Graph(n=n_nodes, edges=edges, directed=False)
G.es['weight'] = weights
G.es['pci_diff'] = diffs

print(f"Nodes: {G.vcount()}, Edges: {G.ecount()}, Density: {(2 * G.ecount()) / (G.vcount() * (G.vcount() - 1))}, Clustering coefficient: {G.transitivity_undirected()}")
print(f"Self-loops: {any(G.is_loop())}")  # Should be False
print(f"Directed: {G.is_directed()}")     # Should be False

print("Adding attributes...")
G.vs['name'] = unique_products
G.vs['pci'] = [pci_dict.get(code, np.nan) for code in unique_products]
G.vs['chapter_code'] = [code_to_chapter(int(code[0:2])) for code in unique_products]
G.vs['strength'] = G.strength(weights='weight') 
G.vs['strength'] = np.array(G.vs['strength']) / np.array(G.vs['strength']).max()
G.vs['degree'] = G.degree()
G.vs['centrality'] = G.eigenvector_centrality(directed=False, scale=True, weights='weight')
G.vs['pagerank_centrality'] = G.personalized_pagerank(directed=False, weights='weight')
G.vs['coreness'] = G.coreness()


def fit_betabinom(data, n_trials): 
    # Negative log-likelihood function to minimize
    def nll(params):
        alpha, beta = params
        valid_data = data[data <= n_trials]
        log_probs = stats.betabinom.logpmf(valid_data, n_trials, alpha, beta)
        return -np.sum(log_probs)

    init_params = [1.0, 1.0]
    
    # Bounds (alpha and beta must be positive)
    bounds = [(1e-6, None), (1e-6, None)]
    mean = np.mean(data)
    var = np.var(data, ddof=0)
    p0 = max(mean / n_trials, 1e-6)
    v0 = max(p0 * (1-p0), 1e-6)
    r = (n_trials * v0) / (var - n_trials * v0) if var > n_trials*v0 else 1.0
    init_alpha = p0 * r
    init_beta = (1-p0) * r
    result = minimize(nll, [init_alpha, init_beta], bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        return (result.x[0], result.x[1])
    else:
        print('Noob')
        return (init_alpha, init_beta)


def goodness_of_fit(data):
    params_normal = stats.norm.fit(data)
    params_lognormal = stats.lognorm.fit(data, floc=0)  # floc=0 fixes location for lognorm
    params_weibull = stats.weibull_min.fit(data, floc=0)  # floc=0 for Weibull
    print("Normal params (μ, σ):", params_normal)
    print("Log-normal params (shape, loc, scale):", params_lognormal)
    print("Weibull params (shape, loc, scale):", params_weibull)
    # KS Test for each distribution
    ks_normal = stats.kstest(data, 'norm', args=params_normal)
    ks_lognormal = stats.kstest(data, 'lognorm', args=params_lognormal)
    ks_weibull = stats.kstest(data, 'weibull_min', args=params_weibull)

    print("\nKS Test Results:")
    print(f"Normal: statistic={ks_normal.statistic:.4f}, p-value={ks_normal.pvalue:.4f}")
    print(f"Log-normal: statistic={ks_lognormal.statistic:.4f}, p-value={ks_lognormal.pvalue:.4f}")
    print(f"Weibull: statistic={ks_weibull.statistic:.4f}, p-value={ks_weibull.pvalue:.4f}")

g_sub = G.copy()
g_sub.delete_edges(g_sub.es.select(weight_eq=0))

sns.kdeplot(g_sub.vs['degree'], bw_method='scott')
plt.show()


n_nodes = len(g_sub.vs['degree'])
max_deg = n_nodes - 1  
a, b = fit_betabinom(np.array(g_sub.vs['degree']), max_deg)

x = np.arange(0, max_deg + 1)
pmf = stats.betabinom.pmf(x, max_deg, a, b)
pmf /= pmf.sum()  # Ensures sum=1
expected = pmf * n_nodes  # Expected counts
bins = np.arange(0, max_deg + 2)
observed, _ = np.histogram(g_sub.vs['degree'], bins=bins, density=False)

observed = observed[:len(expected)]
mask = expected >= 5

observed = observed[mask]
expected = expected[mask]
expected = expected * (observed.sum() / expected.sum())  # Force exact match

print(f"Observed sum: {observed.sum()}, Expected sum: {expected.sum()}")

chi2, p = stats.chisquare(observed, expected)
print(f"Chi-squared p-value: {p:.4f}")

fig, ax = plt.subplots(2,2, figsize=(20, 10))

plt.subplot(2,2,1)
sns.kdeplot(G.vs['strength'], label='Node strength')
sns.kdeplot(G.vs['centrality'], label='Eigenvector centrality')
plt.legend()
plt.xlabel('Centrality value (0-1)')

plt.subplot(2,2,2)
sns.kdeplot(g_sub.es['weight'], label='Weight', bw_method='scott')
plt.xlabel('Edge weight')

plt.subplot(2,2,3)
sns.histplot(g_sub.vs['degree'], binwidth=100)
plt.xlabel('Node degree (unweighted)')

plt.subplot(2,2,4)
import statsmodels.api as sm
plt.title("Q-Q Plot, Beta-Binomial Fit for Degree")
theoretical_quantiles = stats.betabinom.ppf(np.linspace(0.01, 0.99, 100), 5008, a, b, loc=0)
sm.qqplot(np.array(g_sub.vs['degree']), dist=stats.betabinom(max_deg, a, b, loc=0), line='45')


plt.tight_layout()
plt.show()

print('Weight: mean = {}, median = {}, mode = {}, IQR = {}, skewmess = {}, kurtosis = {}'.format(np.mean(G.es['weight']),
                                                                                                   np.median(G.es['weight']),
                                                                                                   stats.mode(G.es['weight']),
                                                                                                   np.percentile(G.es['weight'], 75) - np.percentile(G.es['weight'], 25),
                                                                                                   stats.skew(G.es['weight']),
                                                                                                   stats.kurtosis(G.es['weight'])
                                                                                                   ))
print('Centrality: mean = {}, median = {}, mode = {}, IQR = {}, skewmess = {}, kurtosis = {}'.format(np.mean(G.vs['centrality']),
                                                                                                   np.median(G.vs['centrality']),
                                                                                                   stats.mode(G.vs['centrality']),
                                                                                                   np.percentile(G.vs['centrality'], 75) - np.percentile(G.vs['centrality'], 25),
                                                                                                   stats.skew(G.vs['centrality']),
                                                                                                   stats.kurtosis(G.vs['centrality'])
                                                                                                   ))
print('Degree: mean = {}, median = {}, mode = {}, IQR = {}, skewmess = {}, kurtosis = {}'.format(np.mean(G.vs['degree']),
                                                                                                   np.median(G.vs['degree']),
                                                                                                   stats.mode(G.vs['degree']),
                                                                                                   np.percentile(G.vs['degree'], 75) - np.percentile(G.vs['centrality'], 25),
                                                                                                   stats.skew(G.vs['degree']),
                                                                                                   stats.kurtosis(G.vs['degree'])
                                                                                                   ))
print('Centrality-PCI correlation: {}'.format(stats.pearsonr(G.vs['pci'], G.vs['centrality'])[0]))
print('PCI Diff-weight correlation: {}'.format(stats.pearsonr(G.es['pci_diff'], G.es['weight'])))
