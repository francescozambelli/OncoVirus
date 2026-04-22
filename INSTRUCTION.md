# GOAL
The goal is to convert the analysis contained in the notebooks 0_network_generation, 1_data_production, 2_statistical_analysis into modular python scripts. Insted of using the scripts contained in MuxVizPy, use the muxvizpy conda environment and the muxvizpy library stored in there. The goal is to reproduce the results using this new workflow.

# ORGANIZATION
The PPI networks for each virus are build by subsetting the full Human Protein Interaction Network built using the BIOSTR_homo_sapiens data. The subsetting is done by taking the set of all the proteins of a given virus and by taking all the human proteins that interact with at least one of the viral proteins (look into the 0_network_generation.ipynb file to understand the logic).

The multilayers are build by combining the PPI networks of the different viruses. The chosen combinations are stored in data/MultilayerIndex

Tell also which are the necessary files to complete the work, and propose a meaningful organization of the work tree.

NOTE: for now do not consider the datadrive_approach part.