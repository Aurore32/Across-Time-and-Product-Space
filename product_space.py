import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import pandas as pd
import igraph as ig
import leidenalg
from sklearn.mixture import GaussianMixture
import scipy.stats as st
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from tqdm import tqdm

import warnings
from joblib import Parallel, delayed
from cma import CMAEvolutionStrategy
from scipy import optimize

def get_empirical_dists(year, scale):
    print('Reading data...')
    product_space = pd.read_csv('Results/final/data/product_space_{}.csv'.format(year), 
                            usecols=['product_1', 'product_2', 'proximity'])

    unique_products = product_space['product_1'].unique()
    product_to_idx = {p: i for i, p in enumerate(unique_products)}
    n_nodes = len(unique_products)

    filtered = product_space[
        (product_space['proximity'] > 0) &
        (product_space['product_1'] != product_space['product_2'])
    ]
    unfiltered = product_space[
        (product_space['product_1'] != product_space['product_2'])
    ]
    global percentile
    percentile = len(filtered) / len(unfiltered)
    
    edges = []
    weights = []
    seen = set()  
    print('Building network...')

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

    print(f"Nodes: {G.vcount()}, Edges: {G.ecount()}, Density: {(2 * G.ecount()) / (G.vcount() * (G.vcount() - 1))}")
    print(f"Self-loops: {any(G.is_loop())}")  # Should be False
    print(f"Directed: {G.is_directed()}")     # Should be False

    print("Adding attributes...")
    G.vs['name'] = unique_products
    G.vs['degree'] = np.array(G.degree()) / n_nodes * scale
    G.vs['centrality'] = G.eigenvector_centrality(directed=False, scale=True, weights='weight')
    return np.random.choice(G.vs['degree'], scale), np.random.choice(G.vs['centrality'], scale), np.random.choice(G.es['weight'], scale)

class ProductSpace:
    def __init__(self, pcis,
                 max_caps,
                 low_cluster_size, high_cluster_size,
                 low_within_coef, low_between_coef, low_high_coef,
                 high_within_coef, high_between_coef, high_low_coef,
                 dist_type,
                 param,
                 n_components):

        max_caps = round(max_caps)
        low_cluster_size = round(low_cluster_size)
        n_components = round(n_components)
        high_cluster_size = round(high_cluster_size)
        _, p_value, best_gmm = self.fit_mixture_distribution(pcis, n_components)
        if p_value < 0.05:
            raise ValueError('Noob')
        self.weights = best_gmm.weights_
        means = best_gmm.means_.flatten()
        stds = np.sqrt(best_gmm.covariances_.flatten())
        
        _, _, gmm2 = self.fit_mixture_distribution(pcis, 2)
        periphery_threshold = gmm2.means_.flatten()[0]

        self.gmm = best_gmm
        self.gmm_list = []
        self.max_caps = max_caps

        scaler = (max_caps) / (max(pcis) - min(pcis))

        self.low_cluster_list = []
        self.high_cluster_list = []
        for i in range(n_components):
            self.gmm_list.append(st.norm(loc=(means[i] - min(pcis)) * scaler, scale=stds[i] * scaler))
            print(f'Mean of GMM {i}: {(means[i] - min(pcis)) * scaler}, STD of GMM {i}: {stds[i] * scaler}, Weight of GMM {i}: {self.weights[i]}')
            if means[i] < periphery_threshold:
                self.low_cluster_list.append(i)
            else:
                self.high_cluster_list.append(i)
        
        low_cluster_num = len(self.low_cluster_list)
        high_cluster_num = len(self.high_cluster_list)

        # Cluster sizes
        low_cluster_sizes = [low_cluster_size] * low_cluster_num
        high_cluster_sizes = [high_cluster_size] * high_cluster_num
        self.low_sizes = low_cluster_sizes
        self.high_sizes = high_cluster_sizes
        
        # Initialize clusters
        self.total_capabilities = sum(low_cluster_sizes) + sum(high_cluster_sizes)
        self.clusters = []
        self.cluster_types = {}
        self.cluster_indices = {}

        self.low_clusters = []
        self.high_clusters = []

        self.proximity_matrix = np.zeros((self.total_capabilities, self.total_capabilities))
        
        counter = 0
        # Low-complexity clusters
        for size in low_cluster_sizes:
            cluster_indices = list(range(counter, counter + size))
            self.low_clusters += cluster_indices
            self.clusters.append(cluster_indices)
            self.cluster_types[tuple(cluster_indices)] = 'low'
            for index in cluster_indices:
                self.cluster_indices[index] = 'low'
            counter += size
        
        for size in high_cluster_sizes:
            cluster_indices = list(range(counter, counter + size))
            self.high_clusters += cluster_indices
            self.clusters.append(cluster_indices)
            self.cluster_types[tuple(cluster_indices)] = 'high'
            for index in cluster_indices:
                self.cluster_indices[index] = 'high'
            counter += size

        for cluster in self.clusters:
            cluster_array = np.asarray(cluster)
            if self.cluster_types[tuple(cluster)] == 'low':          
                other_low_clusters = [index for index in self.low_clusters if index not in cluster]
                rows = cluster_array[:, None]  # Make it a column vector
                cols = cluster_array
                self.set_submatrix(rows, cols, dist_type, low_within_coef, param)

                rows = cluster_array[:, None]
                cols = other_low_clusters
                self.set_submatrix(rows, cols, dist_type, low_between_coef, param)

                rows = cluster_array[:, None]
                cols = self.high_clusters
                self.set_submatrix(rows, cols, dist_type, low_high_coef, param)
           
            elif self.cluster_types[tuple(cluster)] == 'high':
                other_high_clusters = [index for index in self.high_clusters if index not in cluster]
                rows = cluster_array[:, None]
                cols = cluster_array
                self.set_submatrix(rows, cols, dist_type, high_within_coef, param)

                rows = cluster_array[:, None]
                cols = other_high_clusters
                self.set_submatrix(rows, cols, dist_type, high_between_coef, param)

                rows = cluster_array[:, None]
                cols = self.low_clusters
                self.set_submatrix(rows, cols, dist_type, high_low_coef, param)
            

            else:
                print('Error')
                raise Exception
            
        np.fill_diagonal(self.proximity_matrix, 1)

        plt.show()
        print(self.proximity_matrix)
    
    def set_submatrix(self, rows, cols, dist_type, mean, param):
        def objective_func(scale):
            calculated_mean = stats.truncexpon.mean(b=b/scale, scale=scale)
            return calculated_mean - mean
        if dist_type == 'constant':
            self.proximity_matrix[rows, cols] = mean
        elif dist_type == 'beta':
            a = mean * param
            b = (1 - mean) * param
            dist = stats.beta
            self.proximity_matrix[rows, cols] = dist.rvs(a, b, size=(len(rows),len(cols)))
        elif dist_type == 'exp':
            b = 1
            
            try:
                sol = optimize.root_scalar(objective_func, bracket=[1e-9, 100], method='brentq')
                self.proximity_matrix[rows, cols] = stats.truncexpon.rvs(b=b/sol.root, scale=sol.root, size=(len(rows),len(cols)))

            except ValueError:
                raise RuntimeError(f"Could not find a scale parameter for target_mean={mean}. "
                                "The mean might be too close to the boundaries 0 or 1.")


    def map_pci_to_caps(self, pci_value):
        sorted_pci = np.sort(self.pcis)
        rank = np.searchsorted(sorted_pci, pci_value) / len(sorted_pci)
        return max(1, int(rank * (self.max_caps - 1) + 1))
                   
    def fit_mixture_distribution(self, data, n_components):
        data = np.asarray(data).reshape(-1,1)
        bics = []
        models = []
        for n in range(1,11):
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            gmm.fit(data)
            models.append(gmm)
            bics.append(gmm.bic(data))
        best_gmm = models[np.argmin(bics)]
        gmm_score = best_gmm.bic(data)

        '''print(gmm_score)
        print(best_gmm.n_components)
        
        plt.figure(figsize=(15,10))
        plt.subplot(1,2,1)
        plt.title('PCI Distribution + GMM Fit')
        sns.histplot(data, kde=True, stat='density', color='skyblue')
        x = np.linspace(min(data), max(data), 1000)
        logprob = best_gmm.score_samples(x.reshape(-1, 1))
        pdf = np.exp(logprob)
        plt.plot(x, pdf, '-r', lw=2, label=f'GMM ({best_gmm.n_components} components)')
        plt.legend()

        n = len(data)
        p = (np.arange(n) + 0.5) / n  # Probabilities [0.5/n, 1.5/n, ..., (n-0.5)/n]
        sample_quantiles = np.sort(data.ravel())
        gmm_sample = best_gmm.sample(1000000)[0].flatten()
        theoretical_quantiles = np.quantile(gmm_sample, p)
        np.sort(theoretical_quantiles)

        plt.subplot(1, 2, 2)
        plt.title('Q-Q Plot')
        plt.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, color='blue')
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
        plt.legend()'''

        # plt.show()

        statistic, p_value = st.kstest(data, lambda x: self.gmm_cdf(np.array([x]), best_gmm)[0])
        return best_gmm.n_components, p_value, best_gmm

    def gmm_cdf(self, x, gmm):
        cdf_vals = np.zeros_like(x, dtype=float)
        for weight, mean, std in zip(gmm.weights_, gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())):
            cdf_vals += weight * st.norm.cdf(x, loc=mean, scale=std)
        return cdf_vals

    def create_product(self):
        
        cluster_type = np.random.choice(list(range(len(self.weights))), p=self.weights) 
        base_cluster = self.clusters[cluster_type]
        starting = np.random.choice(base_cluster, p=np.ones(len(base_cluster)) / len(base_cluster))
        complexity = round(self.gmm_list[cluster_type].rvs(size=1)[0])
        complexity = min(complexity, self.max_caps)
        complexity = max(complexity, 1)
        
        product = [starting]
        capabilities = list(range(self.total_capabilities))
        for index in product:
            capabilities.remove(index)

        for _ in range(complexity - 1):   
            proximity_vector = self.proximity_matrix[product, :].mean(axis=0)[capabilities]
            if all(proximity_vector == 0):
                break
            elif not capabilities:
                break

            proximity_vector = proximity_vector / np.sum(proximity_vector)
            new = np.random.choice(capabilities, p=proximity_vector)
            product.append(new)
            capabilities.remove(new)

        return product, cluster_type

    def create_country(self, cluster_type, complexity):
        
        base_cluster = self.clusters[cluster_type]
        capabilities = list(range(self.total_capabilities))
        starting = np.random.choice(base_cluster, p=np.ones(len(base_cluster)) / len(base_cluster))
        country = [starting]
        for index in country:
            capabilities.remove(index)

        for _ in range(complexity - 1):   
            proximity_vector = self.proximity_matrix[country, :].mean(axis=0)[capabilities]
            if all(proximity_vector == 0):
                break
            elif not capabilities:
                break

            proximity_vector = proximity_vector / np.sum(proximity_vector)
            new = np.random.choice(capabilities, p=proximity_vector)
            country.append(new)
            capabilities.remove(new)

        return country

    def create_product_wrapper(self):
        return self.create_product()

    def calculate_proximity(self, product1, product2):
        submatrix = self.proximity_matrix[np.ix_(product1, product2)]
        return np.mean(submatrix)

    def calculate_proximity_asymmetrical(self, product1, product2):
        ## This is the cost of moving from product 1 to product 2
        set_product1 = set(product1)
        set_product2 = set(product2)
        if set_product1.issubset(set_product2):
            return 1
        else:
            common_capabilities = set_product1 & set_product2
            missing_capabilities = set_product2.difference(set_product1)
            proximity = np.sum(self.proximity_matrix[product1,:][:,list(missing_capabilities)].mean(axis=0))
            return (len(list(common_capabilities)) + proximity) / (len(product2))
            
    def build_product_space(self, n_products):
        results = [self.create_product() for _ in range(n_products)]
        results.sort(key=lambda x: len(x[0]))
        n_total = len(results)
        products = [res[0] for res in results]
        self.products = products
        cluster_types = [res[1] for res in results]
        complexities = [len(p) for p in products]
        self.pci_cluster = cluster_types
        product_space = np.identity(n_total)
        for i in tqdm(range(n_total)):
            for j in range(i+1, n_total):
                prox = self.calculate_proximity(products[i], products[j])
                product_space[i, j] = prox
                product_space[j, i] = prox

        '''product_space_1 = np.identity(n_total)
        for i in tqdm(range(n_total)):
            for j in range(i+1, n_total):
                prox = self.calculate_proximity_asymmetrical(products[i], products[j])
                product_space_1[i, j] = prox
                product_space_1[j, i] = prox

        product_space_2 = np.identity(n_total)
        for i in tqdm(range(n_total)):
            for j in range(i+1, n_total):
                prox = self.calculate_proximity_asymmetrical(products[j], products[i])
                product_space_2[i, j] = prox
                product_space_2[j, i] = prox
            
        product_space = np.minimum(product_space_1, product_space_2)'''
            
        product_space /= np.max(product_space)
        
        products_1 = []
        products_2 = []
        products_1_complexities = []
        products_2_complexities = []
        proximities = []

        for i in tqdm(range(len(products))):
            for j in range(len(products)):
                products_1.append(i)
                products_2.append(j)
                products_1_complexities.append(len(products[i]))
                products_2_complexities.append(len(products[j]))
                proximities.append(product_space[i,j])
        
        self.product_space_df = pd.DataFrame([])
        self.product_space_df['product_1'] = products_1
        self.product_space_df['product_2'] = products_2
        self.product_space_df['product_1_pci'] = products_1_complexities
        self.product_space_df['product_2_pci'] = products_2_complexities
        self.product_space_df['proximity'] = proximities

        self.pci_dict = {self.product_space_df['product_1'][i] : self.product_space_df['product_1_pci'][i] for i in range(len(self.product_space_df))}

        self.product_space_df_pci = pd.pivot(self.product_space_df, index='product_1', columns='product_2', values='proximity')

        return product_space, complexities, cluster_types

    def recalc_product_space(self, products):
        n_total = len(products)
        product_space = np.identity(n_total)
        for i in tqdm(range(n_total)):
            for j in range(i+1, n_total):
                prox = self.calculate_proximity(products[i], products[j])
                product_space[i, j] = prox
                product_space[j, i] = prox
            
        product_space /= np.max(product_space)
        
        return product_space
    
    
    def build_asymmetrical_product_space(self, n_products):
        results = [self.create_product() for _ in range(n_products)]
        results.sort(key=lambda x: len(x[0]))
        n_total = len(results)
        products = [res[0] for res in results]
        cluster_types = [res[1] for res in results]
        print(cluster_types)
        complexities = [len(p) for p in products]

        self.pci_cluster = cluster_types

        product_space = np.identity(n_total)
        for i in tqdm(range(n_total)):
            for j in range(n_total):
                prox = self.calculate_proximity_asymmetrical(products[i], products[j])
                product_space[i, j] = prox
            
        product_space /= np.max(product_space)

        return product_space, complexities, cluster_types
        
    def build_network(self, product_space, complexities, cluster_types):
        global percentile
        edges = []
        weights = []
        diffs = []

        for i in range(product_space.shape[0]):
            for j in range(i+1, product_space.shape[0]):
                if product_space[i,j] > 0:
                    edges.append((i, j))
                    weights.append(product_space[i, j])
                    diffs.append(abs(complexities[i] - complexities[j]))

        G = ig.Graph(n=product_space.shape[0], edges=edges, directed=False)
        G.es['weight'] = weights
        G.es['pci_diff'] = diffs
        G.vs['pci'] = complexities
        G.vs['pci_cluster'] = cluster_types
        G.vs['centrality'] = G.eigenvector_centrality(scale=True, weights='weight', directed=False)
        min_weight = np.percentile(G.es['weight'], (1-percentile) * 100)
        print(min_weight)
        G.vs['degree'] = G.subgraph_edges(G.es.select(weight_gt=min_weight)).degree()
        return G
    
    def cluster_network(self, G):
        partition = leidenalg.find_partition(
            G,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=1,
            n_iterations=-1,
            seed=42
        )
        print(partition.summary())
        G.vs['cluster'] = partition.membership
        print('Modularity from Leiden clustering: {}'.format(G.modularity(partition.membership, weights='weight', directed=False)))
        return G.modularity(partition.membership, weights='weight', directed=False)

    def pci_cluster_network(self, G):
        def assign_quantiles(arr, n):
            if n <= 0:
                raise ValueError("n must be a positive integer")
            arr = np.asarray(arr)
            if n == 1:
                return np.ones(arr.shape, dtype=int)
            percentiles = np.linspace(0, 100, n + 1)[1:-1]
            bin_edges = np.percentile(arr, percentiles)
            bin_indices = np.digitize(arr, bin_edges, right=True) + 1
            return bin_indices
        pci_modularities = []
        for i in range(2,21):
            G.vs['pci_cluster'] = assign_quantiles(G.vs['pci'], i)
            pci_modularities.append(G.modularity(G.vs['pci_cluster'], weights='weight', directed=False))
        
        
        print('Best number of PCI bins, {}; Max modularity from PCI clustering, {}'.format(np.argmax(pci_modularities) + 2, pci_modularities[np.argmax(pci_modularities)]))

        return pci_modularities[np.argmax(pci_modularities)]
        '''print(f'Modularity by PCI cluster: {G.modularity(G.vs['pci_cluster'], weights='weight', directed=False)}')
        return G.modularity(G.vs['pci_cluster'], weights='weight', directed=False)'''
        '''def assign_quantiles(arr, n):
            if n <= 0:
                raise ValueError("n must be a positive integer")
            arr = np.asarray(arr)
            if n == 1:
                return np.ones(arr.shape, dtype=int)
            percentiles = np.linspace(0, 100, n + 1)[1:-1]
            bin_edges = np.percentile(arr, percentiles)
            bin_indices = np.digitize(arr, bin_edges, right=True) + 1
            return bin_indices
        pci_modularities = []
        for i in range(2,21):
            G.vs['pci_cluster'] = assign_quantiles(G.vs['pci'], i)
            pci_modularities.append(G.modularity(G.vs['pci_cluster'], weights='weight', directed=False))
        
        
        print('Best number of PCI bins, {}; Max modularity from PCI clustering, {}'.format(np.argmax(pci_modularities) + 2, pci_modularities[np.argmax(pci_modularities)]))

        return pci_modularities[np.argmax(pci_modularities)]'''

    def centrality_distribution(self, G):
        print('Centrality-PCI correlation: {}'.format(st.pearsonr(G.vs['pci'], G.vs['centrality'])))
        return st.pearsonr(G.vs['pci'], G.vs['centrality'])[0]

    def proximity_distribution(self, G):
        all_weights = G.es['weight']
        print('Proximity mode: {}, Proximity skew: {}'.format(stats.mode(all_weights).mode, stats.skew(all_weights)))
        return stats.mode(all_weights).mode, stats.skew(all_weights)
    
    def pci_assortativity(self, complexities, num_bins=50):
        bins = np.array(range(min(complexities), max(complexities), 2))
        pci_to_bin = {product: np.digitize(pci, bins) for product, pci in self.pci_dict.items()}
        new_df = self.product_space_df_pci
        new_df.index = pd.MultiIndex.from_tuples([(product, pci_to_bin[product]) for product in self.product_space_df_pci.index], names=['product', 'pci_class'])
        new_df.columns = pd.MultiIndex.from_tuples([(product, pci_to_bin[product]) for product in self.product_space_df_pci.columns], names=['product', 'pci_class'])
        bin_df = new_df.groupby(level='pci_class', axis=0).mean().groupby(level='pci_class', axis=1).mean()
        bin_df = bin_df.rename_axis(index='pci_class_1', columns='pci_class_2').stack().reset_index(name='proximity')
        bin_df['pci_1'] = [min(complexities) + 2 * (pci_class - 1) for pci_class in bin_df['pci_class_1']]
        bin_df['pci_2'] = [min(complexities) + 2 * (pci_class - 1) for pci_class in bin_df['pci_class_2']]
        bin_df['pci_diff'] = np.abs(np.array(bin_df['pci_1']) - np.array(bin_df['pci_2']))

        print('PCI assortativity coefficient: {}'.format(st.pearsonr(bin_df['pci_diff'], bin_df['proximity'])))
        return st.pearsonr(bin_df['pci_diff'], bin_df['proximity'])[0]      

    def evaluate_network(self, product_space, complexities, cluster_types):
        critical_value = 1.36 / np.sqrt(1000)
        G = self.build_network(product_space, complexities, cluster_types)
        centrality_D = st.ks_2samp(centrality_dist, G.vs['centrality'])[0]
        weight_D = st.ks_2samp(weight_dist, np.random.choice(G.es['weight'], 1000))[0]
        degree_D = st.ks_2samp(degree_dist, G.vs['degree'])[0]

        print('D-statistic for centrality distributions: {}'.format(centrality_D))
        print('D-statistic for weight distributions: {}'.format(weight_D))
        print('D-statistic for degree distributions: {}'.format(degree_D))
        results = {
        'modularity': self.cluster_network(G),
        'pci_modularity': self.pci_cluster_network(G),
        'pci_centrality_corr': self.centrality_distribution(G),
        'centrality_dist': centrality_D,
        'weight_dist': weight_D,
        'degree_dist': degree_D,
        'pci_assortativity': self.pci_assortativity(complexities)
        }

        targets = {
        'modularity': (0.15, 0.1, 'exact'),       # Target, tolerance, direction (min/max/exact)
        'pci_modularity': (0.1, 0.05, 'exact'),
        'pci_assortativity': (-0.3, 0.05, 'max'),  # More negative is better
        'pci_centrality_corr': (0.3, 0.05, 'min'),
        'centrality_dist': (critical_value, 0.02, 'max'),
        'weight_dist': (critical_value, 0.02, 'max'),
        'degree_dist': (critical_value, 0.02, 'max')
    }
        weights = {
        'modularity': 1.0,
        'pci_modularity': 1.0,
        'pci_assortativity': 1.0, 
        'pci_centrality_corr': 1.0,
        'centrality_dist': 1.0,
        'weight_dist': 1.0,
        'degree_dist': 1.0
    }
    
        total_loss = weight_D
        
        '''for metric, (target, tol, direction) in targets.items():
            actual = results[metric]
            weight = weights[metric]
            
            if direction == 'min':
                if actual < target - tol:
                    diff = max(2, (target - actual) / target)
                    total_loss += weight * (diff ** 2)
            
            elif direction == 'max':
                if actual > target + tol:
                    diff = max(2, (actual - target) / target)
                    total_loss += weight * (diff ** 2)
            
            else:  # 'exact'
                diff = abs(actual - target) / target
                total_loss += weight * (diff ** 2)'''
        
        return total_loss, results
    
    def network_visualization_suite(self, product_space, complexities, cluster_types, visualize, year):
        G = self.build_network(product_space, complexities, cluster_types)
        G.vs['type'] = cluster_types
        results = {
        'year': year,
        'modularity': self.cluster_network(G),
        'pci_modularity': self.pci_cluster_network(G),
        'pci_centrality_corr': self.centrality_distribution(G),
        'pci_assortativity': st.pearsonr(G.es['pci_diff'], G.es['weight'])[0],
        'weight_mode': st.mode(G.es['weight'])[0],
        'weight_skew': st.skew(G.es['weight']),
        'weight_avg': np.mean(G.es['weight']),
        'centrality_mode': st.mode(G.vs['centrality'])[0],
        'centrality_skew': st.skew(G.vs['centrality']),
        'centrality_avg': np.mean(G.vs['centrality'])
        }

        if visualize:
            print(results)
            sns.heatmap(self.proximity_matrix, cmap='viridis', cbar=True)
            plt.show()

            weight_threshold = np.percentile(G.es['weight'], 90)
            g_sub = G.subgraph_edges(G.es.select(weight_ge=weight_threshold))
            layout = g_sub.layout_fruchterman_reingold(
                niter=1000,
                weights='weight'
            )

            cmap = plt.get_cmap('tab20')
            category_colors = [mpl.colors.rgb2hex(cmap(c)) for c in G.vs['type']]

            size_min, size_max = 5,30
            sizes = size_min + (size_max - size_min) * (np.array(G.vs['centrality']) - min(G.vs['centrality'])) / (max(G.vs['centrality']) - min(G.vs['centrality']))
            sizes = sizes.tolist()
            fig, ax = plt.subplots(figsize=(20, 16))
            ig.plot(
                g_sub,
                target=ax,
                layout=layout,
                vertex_size=sizes,
                vertex_color=category_colors,
                vertex_frame_width=0.5,
                vertex_frame_color='white',
                edge_width=0.3,
                edge_color='rgba(50, 50, 50, 0.1)',
                bbox=(0, 0, 1200, 1200)
            )
            legend_handles = []
            categories = list(set(G.vs['type']))
            for i in range(len(categories)):
                legend_handles.append(plt.Line2D(
                    [0], [0], 
                    marker='o', 
                    color='w',
                    markerfacecolor=cmap(i),
                    markersize=10,
                    label='Product Class {}'.format(categories[i])
                ))
            import matplotlib.lines as mlines  # For proxy artists

            # First legend (existing)
            legend1 = plt.legend(
                handles=legend_handles,
                loc='center left',
                bbox_to_anchor=(1, 0.7),
                title="Product Categories"
            )
            plt.gca().add_artist(legend1)

            plt.show()
            plt.subplot(1,3,1)
            sns.histplot(G.subgraph_edges(G.es.select(weight_ge=0.1)).degree(), kde=True, label='Simulated')
            plt.legend()    
            plt.title('Degree')
            plt.subplot(1,3,2)
            sns.histplot(G.vs['centrality'], kde=True, label='Simulated')
            plt.title('Centrality')
            plt.legend()
            plt.subplot(1,3,3)
            plt.title('Weight')
            sns.histplot(np.random.choice(G.es['weight'], 1000), kde=True, label='Simulated')
            plt.legend()
            plt.show()

        return results


    def visualize_network(self, G: ig.Graph):
        weight_threshold = np.percentile(G.es['weight'], 90)
        g_sub = G.subgraph_edges(G.es.select(weight_ge=weight_threshold))
        layout = g_sub.layout_fruchterman_reingold(
            niter=1000,
            weights='weight'
        )

        all_pci = G.vs['pci']
        pci_min = min(pci for pci in all_pci if not np.isnan(pci))
        pci_max = max(pci for pci in all_pci if not np.isnan(pci))
        norm = mcolors.Normalize(vmin=pci_min, vmax=pci_max)
        cmap = cm.ScalarMappable(norm=norm, cmap=cm.jet)

        size_min, size_max = 5,30
        sizes = size_min + (size_max - size_min) * (np.array(G.vs['centrality']) - min(G.vs['centrality'])) / (max(G.vs['centrality']) - min(G.vs['centrality']))
        sizes = sizes.tolist()

        category_colors = [cmap.to_rgba(c) for c in G.vs['pci']]

        fig, ax = plt.subplots(figsize=(20, 16))
        cbar = fig.colorbar(cmap, ax=ax, shrink=0.8)
        legend_handles = []
        for centrality_value in np.linspace(min(G.vs['centrality']), max(G.vs['centrality']), 4):
            # Calculate the corresponding marker size using the same formula as above
            legend_size = size_min + (size_max - size_min) * (centrality_value - min(G.vs['centrality'])) / (max(G.vs['centrality']) - min(G.vs['centrality']))
            
            handle = ax.scatter([], [], # Empty data points
                                s=legend_size, 
                                color='gray',  # Use a neutral color for the legend
                                label=f'{centrality_value:.2f}')
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, 
                  title='Centrality',
                  loc='upper right',
                  fontsize=12,
                  title_fontsize=14,
                  labelspacing=1.5, # Increase spacing between items
                  frameon=True,
                  facecolor='white',
                  edgecolor='black',
                  scatterpoints=1) # Ensure one marker per label
        cbar.set_label('Number of capabilities (PCI)', fontsize=14, weight='bold')
        cbar.ax.tick_params(labelsize=12)

        ig.plot(
            g_sub,
            target=ax,
            layout=layout,
            vertex_size=sizes,
            vertex_color=category_colors,
            vertex_frame_width=0.5,
            vertex_frame_color='white',
            edge_width=0.3,
            edge_color='rgba(50, 50, 50, 0.1)',
            bbox=(0, 0, 1200, 1200)
        )

    def heatmap_pci_assortativity(self, complexities):
        bins = np.array(range(min(complexities), max(complexities), 2))
        pci_to_bin = {product: np.digitize(pci, bins) for product, pci in self.pci_dict.items()}
        new_df = self.product_space_df_pci
        new_df.index = pd.MultiIndex.from_tuples([(product, pci_to_bin[product]) for product in self.product_space_df_pci.index], names=['product', 'pci_class'])
        new_df.columns = pd.MultiIndex.from_tuples([(product, pci_to_bin[product]) for product in self.product_space_df_pci.columns], names=['product', 'pci_class'])
        bin_df = new_df.groupby(level='pci_class', axis=0).mean().groupby(level='pci_class', axis=1).mean()
        sns.heatmap(bin_df, cmap='jet')
        plt.show()
    
    def visualize_network_dists(self, G: ig.Graph):
        global percentile
        plt.subplot(1,3,1)
        sns.histplot(G.vs['degree'], kde=True, label='Simulated')
        sns.histplot(degree_dist, color='orange', label='Empirical', kde=True)
        plt.legend()    
        plt.title('Degree')
        plt.subplot(1,3,2)
        sns.histplot(G.vs['centrality'], kde=True, label='Simulated')
        sns.histplot(centrality_dist, color='orange', label='Empirical', kde=True)
        plt.title('Centrality')
        plt.legend()
        plt.subplot(1,3,3)
        plt.title('Weight')
        sns.histplot(np.random.choice(G.es['weight'], 1000), kde=True, label='Simulated')
        sns.histplot(weight_dist[weight_dist > 0], color='orange', label='Empirical', kde=True)
        plt.legend()
        plt.show()

'''ps = ProductSpace(
    pcis=pci_data_2005,
    max_caps=100,
    low_cluster_num=2,
    high_cluster_num=2,
    low_cluster_size=20,
    high_cluster_size=20,
    low_within_ratio=1,
    low_between_ratio=0,
    low_high_ratio=0,
    high_within_ratio=0.4,
    high_between_ratio=0.3, 
    high_low_ratio=0.3
)
product_space, complexities = ps.build_network(500)'''
'''
Gen 50: Best Loss=0.171, Params=[0.80206855 0.010074   0.04695638 0.80161548 0.01066959 0.01425405]
Best Results:
{'modularity': 0.21877473156697397, 'pci_modularity': 0.10387437971324379, 'pci_centrality_corr': -0.15562770847163201, 'centrality_dist': 0.726, 'weight_dist': 0.171, 'degree_dist': 0.288, 'pci_assortativity': -0.5831690885511154}
'''

def optimize_product_space(params):
    low_within_coef, low_between_coef, low_high_coef, high_within_coef, high_between_coef, high_low_coef = tuple(params)
    if any(param < 0 for param in params):
        return 69420
    try:
        print()
        print('---------------------------------')
        print('Parameters: low_within_coef={}, low_between_coef={}, low_high_coef={}, high_within_coef={}, high_between_coef={}, high_low_coef={}'.format(
           low_within_coef, low_between_coef, low_high_coef, high_within_coef, high_between_coef, high_low_coef)
        )
        ps = ProductSpace(pcis=pci_data, max_caps=100, 
                        low_cluster_size=20, high_cluster_size=20,
                        low_within_coef=low_within_coef, low_between_coef=low_between_coef, low_high_coef=low_high_coef,
                        high_within_coef=high_within_coef, high_between_coef=high_between_coef, high_low_coef=high_low_coef,
                        param=1000,
                        dist_type='constant',
                        n_components=10)
        product_space, complexities, cluster_types = ps.build_product_space(1000)
        loss, results = ps.evaluate_network(product_space, complexities, cluster_types)
        print('Loss: {}'.format(loss))
        return loss, results
    except Exception as e:
        print('Error: {}'.format(e))
        return 69420
    

def ga_optimize():
    # Parameter bounds
    bounds = [[0.8, 0.01, 0.01, 0.8, 0.01, 0.01], [0.99, 0.1, 0.1, 0.99, 0.2, 0.2]] # 2 to 3 is actually the range of log kappa

    # CMA-ES initialization
    x0 = [0.9, 0.05, 0.05, 0.9, 0.05, 0.05]  # Center point
    sigma0 = 0.5  # Initial step size
    
    # Optimization with restarts
    best_loss = float('inf')
    best_params = None
    best_results = {}
    
    for _ in [1]:
        es = CMAEvolutionStrategy(x0, sigma0, inopts={
            'bounds': bounds,
            'maxiter': 50,
            'popsize': 20  # Population size
        })
        
        while not es.stop():
            
            solutions = es.ask()
            losses = []
            results = []
            
            with Parallel(n_jobs=-1) as parallel:
                outputs = parallel(delayed(optimize_product_space)(x) 
                            for x in solutions)
            for loss, result in outputs:
                losses.append(loss)
                results.append(result)
            
            es.tell(solutions, losses)
            es.logger.add()  # For logging
            
            # Track best solution
            current_best = min(losses)
            if current_best < best_loss:
                best_loss = current_best
                best_params = solutions[np.argmin(losses)]
                best_results = results[np.argmin(losses)]
            
            print(f"Gen {es.countiter}: Best Loss={es.best.f}, Params={es.best.x}")
            print(f"Best Results:")
            print(best_results)
        
        # Prepare for restart
        x0 = best_params  # Start from current best
        sigma0 *= 0.5    # Reduce step size
    
    return best_params, best_loss

if __name__ == '__main__':
    warnings.simplefilter('ignore', category=FutureWarning)
    params = [0.98999995, 0.0795094, 0.01187099, 0.97032985, 0.01005868, 0.02077327]
    pci_data = pd.read_csv('Results/final/data/HS92_pci.csv')
    pci_data = np.array(pci_data[pci_data['year'] == 2005]['pci'])
    degree_dist, centrality_dist, weight_dist = get_empirical_dists(2005, 1000) # scale = number of products
    ps = ProductSpace(pcis=pci_data, max_caps=100,
                      low_cluster_size=25, high_cluster_size=25,
                      low_within_coef=params[0],
                      low_between_coef=params[1],
                      low_high_coef=params[2],
                      high_within_coef=params[3],
                      high_between_coef=params[4],
                      high_low_coef=params[5],
                      dist_type='beta', param=400, n_components=8)
    product_space, complexities, cluster_types = ps.build_product_space(1000)
    loss = ps.evaluate_network(product_space, complexities, cluster_types)
    print('Loss = {}'.format(loss))

    G = ps.build_network(product_space, complexities, cluster_types)
    ps.visualize_network(G)
    plt.show()
    ps.visualize_network_dists(G)
    plt.show()
    '''best_params, best_loss = ga_optimize()
    print()
    print()
    print()
    print('-------------------------------------------')
    print('Best parameters: {}'.format(best_params))
    print('Best loss: {}'.format(best_loss))'''

    '''ps = ProductSpace(pcis=pci_data, max_caps=100,
                      low_cluster_size=25, high_cluster_size=25,
                      **params, 
                      dist_type='constant', param=10 ** 2.54075719, n_components=8)
    product_space, complexities, cluster_types = ps.build_product_space(1000)
    loss = ps.evaluate_network(product_space, complexities, cluster_types)
    print('Loss = {}'.format(loss))

    G = ps.build_network(product_space, complexities, cluster_types)
    ps.visualize_network(G)
    plt.show()
    ps.visualize_network_dists(G)
    plt.show()'''

    '''best_params, best_loss = ga_optimize()
    print()
    print()
    print()
    print('-------------------------------------------')
    print('Best parameters: {}'.format(best_params))
    print('Best loss: {}'.format(best_loss))
'''
# opt_params = [5.95292229e+01 2.57316435e+01 2.43713462e+01 8.65436768e-01 6.16177293e-02 8.63409413e-01 5.01028341e-02 7.77679473e+00]
# Asymmetric: Params=[58.83777633 26.15119269 24.51747751  0.84562998  0.08342726  0.81807037 0.07694384  9.70723719]

# 2000 constant: [0.98954199 0.09639131 0.01005841 0.98870606 0.01000218 0.01770049]
# 2005 constant: Gen 50: Best Loss=0.039, Params=[0.98999995 0.0795094  0.01187099 0.97032985 0.01005868 0.02077327]
# 2005 beta: [0.98213012 0.09488212 0.01002641 0.97349211 0.01000112 0.02292742]

# Very happy with these results! Think we can stop here :)

'''
lb = [20, 5, 5, 0.7, 0, 0.7, 0, 2]
ub = [100, 50, 50, 1, 0.3, 1, 0.3, 10]

# Run optimization
x_opt, f_opt = pso(
    optimize_product_space, 
    lb, ub, 
    swarmsize=20, 
    maxiter=100,
    phip=0.5, phig=0.5  # Cognitive/social weights
)
print(x_opt, f_opt)
'''

'''ps = ProductSpace(pcis=pci_data_2005, max_caps=28.038085576161745, low_cluster_num=3.089078350765352, high_cluster_num=2.0860482665388926, low_cluster_size=23.7696489314977, high_cluster_size=27.216346997946665, low_within_ratio=0.15018260646716505, low_between_ratio=0.7323526050449244, low_high_ratio=1-0.7323526050449244-0.15018260646716505, high_within_ratio=0.06762349122736308, high_between_ratio=0.664021215031614, high_low_ratio=1-0.664021215031614-0.06762349122736308)
product_space, complexities = ps.build_product_space(500)
print(list(set(complexities)))
G = ps.build_network(product_space, complexities)
loss = ps.evaluate_network(product_space, complexities)
print('Loss: {}'.format(loss))
ps.visualize_network(G)
ps.heatmap_pci_assortativity(complexities)'''

'''space = [
    (20, 150),
    (1,5),
    (1,5),
    (5,50),
    (5,50),
    (0.7,1),
    (0.01,0.3),
    (0.5,1),
    (0.01,0.5)
]

lb = [25, 3, 2, 20, 25, 0.1, 0.6, 0.01, 0.4]
ub = [35, 3.1, 2.1, 25, 35, 0.2, 0.8, 0.1, 0.8]

# Run optimization
x_opt, f_opt = pso(
    optimize_product_space, 
    lb, ub, 
    swarmsize=20, 
    maxiter=100,
    phip=0.5, phig=0.5)  # Cognitive/social weights
print()
print('----------------------------------')
print(x_opt)
'''
'''# Run optimization
x_opt, f_opt = pso(
    optimize_product_space, 
    lb, ub, 
    swarmsize=20, 
    maxiter=100,
    phip=0.5, phig=0.5  # Cognitive/social weights
)'''
'''optimizer = CMAEvolutionStrategy(x0=[29.78726424313216, 2.5639788571991198, 2.1846975379836593, 22.18662777869461, 28.74491162394756, 0.15835579044531148, 0.6668056570181827, 0.06685075017056609, 0.460106445023304], sigma0=0.5, inopts={'popsize': 15, 'maxiter': 1000})
while not optimizer.stop():
    solutions = optimizer.ask()
    losses = [optimize_product_space(x) for x in solutions]
    optimizer.tell(solutions, losses)
    optimizer.logger.add()  # Track progress
optimizer.result_pretty()  # Best parameters
'''
'''space = [
    (20, 100),
    (1,5),
    (1,5),
    (5,50),
    (5,50),
    (0.7,1),
    (0.01,0.3),
    (0.5,1),
    (0.01,0.5)
]

lb = [20, 1, 1, 5, 5, 0.7, 0.01, 0.5, 0.01]
ub = [100, 2, 2, 50, 50, 1, 0.3, 1, 0.5]

# Run optimization
x_opt, f_opt = pso(
    optimize_product_space, 
    lb, ub, 
    swarmsize=20, 
    maxiter=50,
    phip=1.5, phig=1.5  # Cognitive/social weights
)'''


'''
optimizer = CMAEvolutionStrategy(x0=[50,2,3,20,20,0.7,0.2,0.5,0.4], sigma0=0.5, inopts={'popsize': 15, 'maxiter': 1000})
while not optimizer.stop():
    solutions = optimizer.ask()
    losses = [optimize_product_space(x) for x in solutions]
    optimizer.tell(solutions, losses)
    optimizer.logger.add()  # Track progress
optimizer.result_pretty()  # Best parameters
'''

'''
0.05482298157996076
D-statistic for centrality distributions: 0.307
D-statistic for weight distributions: 0.062
D-statistic for degree distributions: 0.055
Clustering with 1000 elements and 6 clusters
Modularity from Leiden clustering: 0.12466825202110111
Best number of PCI bins, 5; Max modularity from PCI clustering, 0.06755978146487383
Centrality-PCI correlation: PearsonRResult(statistic=0.2490788381449159, pvalue=1.3167612054037387e-15)
PCI assortativity coefficient: PearsonRResult(statistic=-0.41209196258002084, pvalue=2.841525733997e-73)
Loss = 0.062
0.05482298157996076


Gen 17: Best Loss=0.033, Params=[0.80008951 0.06253514 0.01276981 0.80006697 0.01170295 0.01175191]
Best Results:
{'modularity': 0.1832861098535279, 'pci_modularity': 0.07741752036971994, 'pci_centrality_corr': 0.21303731281289948, 'centrality_dist': 0.48, 'weight_dist': 0.033, 'degree_dist': 0.134, 'pci_assortativity': -0.5269094244495357}

'''