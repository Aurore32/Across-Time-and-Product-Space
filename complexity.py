import pandas as pd
import scipy.stats as st
from sklearn.mixture import GaussianMixture
import numpy as np
import igraph as ig

from product_space import ProductSpace
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN

import cma
from tqdm import tqdm
import country_converter as cc 

from KDEpy import FFTKDE, NaiveKDE, TreeKDE
from multiprocessing import Pool, cpu_count
from functools import partial

coco = cc.CountryConverter()

class Complexity:
    def __init__(self, ps, pci_gmm, pci_data, eci_data, product_space, cluster_types, products, max_complexity, export_df, pci_df, eci_df):
        self.ps = ps
        self.pci_gmm = pci_gmm
        self.pci_data = pci_data
        self.eci_data = eci_data
        self.product_space = product_space
        self.products = products
        self.complexities = np.array([len(p) for p in self.products])
        
        self.export_df = export_df
        self.pci_df = pci_df
        self.eci_df = eci_df

        self.capability_complexities = []
        num_caps = self.ps.proximity_matrix.shape[0]
        avg_complexity = np.average(self.complexities)
        
        for i in range(num_caps):
            product_indices = [idx for idx, p_caps in enumerate(self.products) if i in p_caps]
            if not product_indices:
                self.capability_complexities.append(avg_complexity)
            else:
                cap_complexity = np.mean(self.complexities[product_indices])
                self.capability_complexities.append(cap_complexity)
        
        self.capability_complexities = np.asarray(self.capability_complexities)
        self.n_products = product_space.shape[0]
        self.cluster_types = cluster_types
        self.max_complexity = max_complexity

    def calc_complexity(self, start_year, bw_method, tune_params):
        eci_year_df = self.eci_df[self.eci_df['year'] == start_year]
        pci_year_df = self.pci_df[self.pci_df['year'] == start_year]
        export_year_df = self.export_df[self.export_df['time'] == start_year]
        sorted_list = self.eci_df[self.eci_df['year']==start_year].sort_values('eci', ascending=True)['exporter_name'].unique().tolist()
        all_countries_in_year = self.export_df[(self.export_df['time'] == start_year)]['exporter_name'].unique().tolist()
        countries = [country for country in sorted_list if country in all_countries_in_year]
        num_processes = cpu_count()
        print('Number of processes: {}'.format(num_processes))
        worker_args = [
            (
                country, self, export_year_df, pci_year_df,
                eci_year_df, start_year, bw_method, tune_params, (i % num_processes) + 1
            )
            for i, country in enumerate(countries)
            ]
        print('Starting model...')
        with Pool(processes=num_processes) as pool:
            # The main TQDM bar now wraps the starmap call
            results = list(tqdm(
                pool.imap(process_country, worker_args), 
                total=len(countries),
                desc="Overall Progress"
            ))


        results = [res for res in results if res is not None]
        df = pd.DataFrame(results)
        df = df.dropna()
        return df

    def optimize_params(self, country, eci, initial_capabilities, country_vector, n_trials, rho_params, nu_params, position):
        best_loss = float('inf')
        best_capabilities = []
        best_r = None
        best_n = None
        all_results = []
        param_pairs = [(r, n) for r in rho_params for n in nu_params]

        param_iterator = tqdm(
            param_pairs, 
            desc=f"Optimizing {country[:10]:<10}", # Pad country name for alignment
            position=position,
            leave=False 
        )

        for r in rho_params:
            for n in nu_params:
                trial_losses = []
                trial_capabilities = []
                for _ in range(n_trials):
                    caps, _ = self.find_capabilities(initial_capabilities, country_vector, r, n)
                    loss = -self._log_loss(caps, country_vector, r, n)[0]
                    trial_capabilities.append(caps)
                    trial_losses.append(loss)

                min_trial_loss = min(trial_losses)
                best_trial_caps = trial_capabilities[np.argmin(trial_losses)]
                
                all_results.append({'country': country, 'eci': eci, 'rho': r, 'nu': n, 'loss': min_trial_loss})
                
                if min_trial_loss < best_loss:
                    best_loss = min_trial_loss
                    best_capabilities = best_trial_caps
                    best_r, best_n = r, n
        
        return best_capabilities, best_loss, best_r, best_n, all_results

    def _calculate_initial_country_vector(self, country_capabilities, rho, nu):
        def calculate_proximity(country_capabilities, product, rho):
            submatrix = self.ps.proximity_matrix[np.ix_(country_capabilities, product)]
            mean_prox_to_each_cap = np.mean(submatrix, axis=0) # Avg prox from country caps to each product cap

            if rho == 'leontief':
                return np.min(mean_prox_to_each_cap)
            elif rho == 0:
                return np.exp(np.mean(np.log(mean_prox_to_each_cap + 1e-9)))
            else:
                return np.mean(mean_prox_to_each_cap ** rho)

        if rho == 0 or rho == 'leontief':
            vector = np.array([calculate_proximity(country_capabilities, product, rho) ** nu for product in self.products])
        else:
            vector = np.array([calculate_proximity(country_capabilities, product, rho) ** (nu/rho) for product in self.products])

        # Normalize vector
        vec_sum = vector.sum()
        return vector / vec_sum if vec_sum > 0 else np.full_like(vector, 1.0 / len(vector))

    def find_capabilities(self, prev_capabilities, country_vector, rho, nu):
        current_score, _, _, _ = self._log_loss(prev_capabilities, country_vector, rho, nu)
        current_capabilities = prev_capabilities.copy()

        best_capabilities, best_score = current_capabilities, current_score
        
        temp = 1.0
        cooling_rate = 0.95
        
        for _ in range(100): # Simulated Annealing loop
            candidate = self._perturb_capabilities(current_capabilities, temp)
            candidate_score, _, _, _ = self._log_loss(candidate, country_vector, rho, nu)

            # Metropolis-Hastings acceptance criterion
            if candidate_score > current_score or np.random.random() < np.exp((candidate_score - current_score) / temp):
                current_capabilities, current_score = candidate, candidate_score
                
                if candidate_score > best_score:
                    best_capabilities, best_score = candidate, candidate_score
            
            temp *= cooling_rate
        
        return best_capabilities, best_score

    def _log_loss(self, current_capabilities, country_vector, rho, nu):
        simulated_vector = self._calculate_initial_country_vector(current_capabilities, rho, nu)     
        
        eps = 1e-10
        p = np.clip(country_vector, eps, 1)
        q = np.clip(simulated_vector, eps, 1)
        p /= p.sum()
        q /= q.sum()

        kl_div = np.sum(p * np.log(p / q))
        log_like = -kl_div
        
        # Priors can be tuned or removed if they don't add value
        # log_prior = np.log(self.ps.calculate_proximity(current_capabilities, prev_capabilities) + eps)
        # hamming = len(set(prev_capabilities) ^ set(current_capabilities)) / len(prev_capabilities) if len(prev_capabilities) > 0 else 0
        # log_temp = -0.25 * hamming 
        
        return log_like, log_like, 0.0, 0.0 # Returning 0 for priors for simplicity

    def _perturb_capabilities(self, current_capabilities, T):
        """
        Perturbs the set of capabilities for the simulated annealing process.
        This version is vectorized for performance.
        """
        num_total_caps = self.ps.proximity_matrix.shape[0]
        if len(current_capabilities) == 0:
            proximity = np.ones(num_total_caps)
        else:
            mean_prox_to_current = np.mean(self.ps.proximity_matrix[:, current_capabilities], axis=1)
            proximity = np.zeros(num_total_caps)
            is_in_current = np.zeros(num_total_caps, dtype=bool)
            is_in_current[current_capabilities] = True
            proximity[is_in_current] = 1.0 - mean_prox_to_current[is_in_current]
            proximity[~is_in_current] = mean_prox_to_current[~is_in_current]

        prob_sum = proximity.sum()
        if prob_sum == 0: # Avoid division by zero
            capability_probs = np.full(num_total_caps, 1.0 / num_total_caps)
        else:
            capability_probs = proximity / prob_sum

        n_flips = max(1, int(round(5 * T)))
        
        flipped_indices = np.random.choice(
            num_total_caps, 
            n_flips, 
            p=capability_probs, 
            replace=False
        )
        
        new_caps_set = set(current_capabilities)
        for i in flipped_indices:
            if i in new_caps_set:
                new_caps_set.remove(i)
            else:
                new_caps_set.add(i)
        
        return np.array(list(new_caps_set), dtype=int)

def process_country(args):
    return _process_country(*args)

def _process_country(country, complexity_obj, export_df, pci_df, eci_df, start_year, bw_method, tune_params, position):
    products = pci_df[pci_df['year'] == start_year].sort_values('product_name')['product_name'].unique().tolist()
    pci = pci_df[pci_df['year'] == start_year].sort_values('product_name')['pci'].to_numpy()
    pci = (pci - min(pci)) * complexity_obj.max_complexity / (max(pci) - min(pci))
    
    uniform = np.ones(len(complexity_obj.complexities)) / len(complexity_obj.complexities)

    country_exports = export_df[(export_df['time'] == start_year) & (export_df['exporter_name'] == country) 
                                               & (export_df['product_name'].isin(products))].sort_values('product_name')
    country_exports = country_exports.merge(pd.DataFrame(products, columns=['product_name']), on='product_name', how='right')
    country_exports = country_exports[['product_name', 'value']].fillna(0)
    country_exports = country_exports['value'].to_numpy()

    if country_exports.sum() == 0:
        return None # Skip countries with no data

    country_exports = country_exports / country_exports.sum()
    
    actual_eci = eci_df[(eci_df['year'] == start_year) & (eci_df['exporter_name'] == country)]['eci'].iloc[0]
    
    simulated_sample = np.random.choice(pci, 5000, p=country_exports)
    
    try:
        kde = TreeKDE(kernel='gaussian', bw=bw_method).fit(simulated_sample)
    except:
        return None
    
    vector = kde.evaluate(complexity_obj.complexities)
    vector = vector / sum(vector)
    
    naive_loss = np.sum(vector * np.log(vector / uniform)) # Worst-case KL divergence

    # Initial guess for capabilities is the set required by the most likely product
    mode_product_idx = np.argmax(vector)
    country_capabilities = complexity_obj.products[mode_product_idx]

    if tune_params:
        best_rho, best_nu = 1, 1
        for i in range(1): 
            # Optimize rho
            country_capabilities, loss, r, n, _ = complexity_obj.optimize_params(country, actual_eci, country_capabilities, vector, 5, [1, 0, -3, -9, 'leontief'], [best_nu], position)
            best_rho, best_nu = r, n
            # Optimize nu
            country_capabilities, loss, r, n, _ = complexity_obj.optimize_params(country, actual_eci, country_capabilities, vector, 5, [best_rho], [0.5, 1, 2, 3, 4], position)
            best_rho, best_nu = r, n
    else:
        country_capabilities, loss, r, n, _ = complexity_obj.optimize_params(country, actual_eci, country_capabilities, vector, 3, [1], [1], position)
        best_rho, best_nu = r, n

    # Final data dictionary for this country
    pred_eci = np.average(complexity_obj.capability_complexities[country_capabilities]) if len(country_capabilities) > 0 else 0
    data = {
        'country': country, 
        'kl_divergence': loss,
        'clarity': 1 - (loss / naive_loss) if naive_loss > 0 else 1.0,
        'pred_num_caps': len(country_capabilities), 
        'pred_eci': pred_eci,
        'actual_eci': actual_eci, 
        'best_rho': best_rho, 
        'best_nu': best_nu
    }
    return data


def simulate_complexity(year, n_products, bw_method, tuning, params):
    pci_data_year = pci_data[pci_data['year'] == year]['pci'].to_numpy()
    eci_data_year = eci_data[eci_data['year'] == year]['eci'].to_numpy()            

    ps = ProductSpace(pcis=pci_data_year, max_caps=100,
                    low_cluster_size=25, high_cluster_size=25,
                    low_within_coef=params[0], low_between_coef=params[1],
                    low_high_coef=params[2], high_within_coef=params[3],
                    high_between_coef=params[4], high_low_coef=params[5],
                    dist_type='beta', param=1000, n_components=8)

    pci_gmm = ps.gmm
    product_space, complexities, cluster_types = ps.build_product_space(n_products)
    simulator = Complexity(
            ps=ps, pci_gmm=pci_gmm,    
            pci_data=pci_data_year, eci_data=eci_data_year, 
            product_space=product_space, cluster_types=cluster_types,
            products=ps.products, max_complexity=100,
            export_df=export_data, pci_df=pci_data, eci_df=eci_data
    )

    df = simulator.calc_complexity(year, bw_method, tuning)
    df.to_csv(f'Results/final/results/simulated_complexity_{year}_{bw_method}_{tuning}.csv', index=False)
    print(f"Simulation complete. Results saved to Results/final/results/simulated_complexity_{year}_{bw_method}_{tuning}_new.csv")

if __name__ == "__main__":
    print('Loading data...')
    export_data = pd.read_csv('Results/final/data/HS92_complexity.csv', usecols=['time','exporter_name','product_name','value','rca','pci'])
    pci_data = pd.read_csv('Results/final/data/HS92_pci.csv', dtype={'pci': 'float32', 'year': 'int16'})
    eci_data = pd.read_csv('Results/final/data/HS92_eci.csv', dtype={'eci': 'float32', 'year': 'int16'})
    print('Finished loading data! Simulation start!')
    params_2000 = [0.988, 0.098, 0.01002641, 0.988, 0.01000112, 0.017]
    params_2010 = [0.989, 0.099, 0.011, 0.990, 0.010, 0.010]
    params_2015 = [0.981, 0.100, 0.010, 0.989, 0.010, 0.011]
    params_2005 = [0.98213012, 0.09488212, 0.01002641, 0.97349211, 0.01000112, 0.02292742]
    simulate_complexity(year=2010, n_products=1000, bw_method='ISJ', tuning=True, params=params_2010)
