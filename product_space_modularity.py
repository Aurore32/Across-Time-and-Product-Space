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
pci_vals = pci_data['pci']
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

G = ig.Graph(n=n_nodes, edges=edges, directed=False)
G.es['weight'] = weights

'''edge_weights = G.es['weight']
weight_threshold = np.percentile(edge_weights, 90)
G = G.subgraph_edges(G.es.select(weight_ge=weight_threshold))
'''

print(f"Nodes: {G.vcount()}, Edges: {G.ecount()}")

G.vs['name'] = unique_products
G.vs['pci'] = [pci_dict.get(name, np.nan) for name in unique_products]
G.vs['chapter_code'] = [int(name[0:2]) for name in unique_products]
G.vs['category'] = [categories.index(code_to_chapter(int(name[0:2]))) for name in unique_products]

G.vs['centrality'] = G.eigenvector_centrality(directed=False, scale=True, weights='weight')
G.vs['pagerank_centrality'] = G.personalized_pagerank(directed=False, weights='weight')
G.vs['coreness'] = G.coreness()

percentiles = [0,10,20,30,40,50,60,70,80,90,95,99]
leiden = []
pcis = []
product_classes = []    

for i in [0,10,20,30,40,50,60,70,80,90,95,99]:
    edge_weights = G.es['weight']
    weight_threshold = np.percentile(edge_weights, i)
    subg = G.subgraph_edges(G.es.select(weight_ge=weight_threshold))

    def classify_into_bins(values, bin_edges):
        values = np.asarray(values)
        
        indices = np.searchsorted(bin_edges, values, side='left') - 1
        clipped_indices = np.clip(indices, 0, len(bin_edges) - 2)
        return clipped_indices

    partition = leidenalg.find_partition(
        subg,
        leidenalg.ModularityVertexPartition,
        weights='weight',
        n_iterations=-1,
        seed=42
    )

    print(f"Clustering completed in {time.time()-start:.2f}s")
    print(partition.summary())

    subg.vs['cluster'] = partition.membership
    num_clusters = len(list(set(partition.membership)))

    leiden.append(subg.modularity(subg.vs['cluster'], weights='weight', directed=False))
    product_classes.append(subg.modularity(subg.vs['category'], weights='weight', directed=False))
    pcis_temp = []
    for q in range(2,20):
        bins, binedges = pd.qcut(pci_vals, q=q, labels=False, duplicates="drop", retbins=True)
        
        subg.vs['pci_class'] = classify_into_bins(subg.vs['pci'], binedges)
        pcis_temp.append(subg.modularity(subg.vs['pci_class'], weights='weight', directed=False))
    pcis.append(max(pcis_temp))

print(leiden)
print(pcis)
print(product_classes)

plt.plot(percentiles, leiden, '.-b', label='Leiden algorithm')
plt.plot(percentiles, pcis, '^--g', label='PCI')
plt.plot(percentiles, product_classes, 'D-.r', label='HS92 product classes')
plt.legend()
plt.xlabel('Top percentile of edges')
plt.ylabel('Modularity')
plt.show()

