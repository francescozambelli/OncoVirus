# OncoVirus
Repository for notebooks and data related to the article "Unraveling the network signatures of oncogenicity in virus-human protein-protein interactions".

The core of the analysis and the data production is contained in the 4 notebooks:
- **0_networks_generation**: code for the production of the PPI networks associated to each virus in the database starting from the human PPI interaction collected from BIOGrid vv. ??
- **1_data_production**: here are defined the virus combinations to build the combiantion sets used for the multilayer network construction. Then topological quantities relevant for the analysis are computed and saved. In particular they are: Largest Viable Component size, page-rank node percolation critical point, number of modules and modularity from DCSBM.
- **2_statistical_analysis**: the distribution of the quantities produced before are now analyzed in order to see compatibilities between the different combination sets. The plots for figures 3,4,5 from the article are produced here.
- **3_datadrive_approach**: a perceptron is trained to discriminate the presence of an oncogenic layer of a multilayer by analyzing the results of the page-rank centrality. The weights are then analyzed to contribution of the proteins in the classification task.
