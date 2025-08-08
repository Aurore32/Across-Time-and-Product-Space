# Across-Time-and-Product-Space
Code and all data files for the working paper "Across Time and (Product) Space: A Capability-Centric Model of Relatedness and Economic Complexity"!

The following provides a road-map for the data files in the repository:
- Stored in /data
  - country_diversity.csv: Diversity values calculated for all countries, 1995-2023 (HS92)
  - HS92_complexity.csv: root data file for all complexity calculations. Contains all country-product export flows (exporter = ..., product = ..., value = ...) and ECI and PCI values for countries and products, 1995-2023.
    This file is very large; if you wish to use the code provided, please download the file from https://www.dropbox.com/scl/fi/qf2gosrkwbnntjczuz31a/HS92_complexity.csv?rlkey=44mzapp94xef4ran3tgdgzw3c&st=x16q0tbl&dl=0 (or calculate your own!)
  - HS92_eci.csv: ECI data for countries, 1995-2023.
  - HS92_pci.csv: PCI data for countries, 1995-2023.
  - product_aggregated_exports.csv: Contains all country-product export flows (exporter = ..., product = ..., value = ...), without ECI and PCI values. Needed for calculating the Product Space and complexity.
    This file is very large; unfortunately, I was unable to upload this to Dropbox (because, well, I'm poor as heck ;-;), but it's embedded in HS92_complexity.csv - just remove all the complexity columns and voila!
  - product_codes_HS92_V202501.csv: Contains all product-code + product-name pairs.
  - product_space_*.csv (* = 2000, 2005, 2010, 2015): Contains proximity values in the product space, formatted as (product_1_code = ..., product_2_code = ..., proximity = ...).
    These files are very large; if you wish to use the code provided, please calculate your own versions using make_product_space.py. Rename them according to the scheme above.
  - regression_dataset.csv: Dataset of World Bank + Global Macroeconomic Dataset indicators. Not processed in any way.
- Stored in /results
  - corr_df_*.csv (* = 2000, 2005, 2010, 2015): correlation coefficients for different indices.
  - simulated_complexity_*_ISJ_**.csv (* = 2000, 2005, 2010, 2015; ** = True, False): Calculated capability sets + complexity indices for all countries in year *, where rho and nu either vary (** = True) or do not vary and are set at 1 (* = False). ISJ is the KDE bandwidth estimator method (Improved Sheather-Jones).
- .py code files
  - ces_regression.py: Regression against rho and nu.
  - complexity.py: Estimating capability sets for countries.
  - growth_regression.py: Regression against average growth.
  - make_complexity.py: Calculate ECI and PCI. Uses the ecomplexity Python package.
  - make_product_space.py: Calculate the Product Space. Uses the ecomplexity Python package.
  - product_space_modularity.py: Modularity statistics for the Product Space and simulated versions.
  - product_space_properties.py: Topological properties of the Product Space (degree, weight, centrality).
  - product_space.py: Simulation of the model for the Product Space.
