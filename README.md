# OncoVirus

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Research](https://img.shields.io/badge/research-Complex%20Systems-orange.svg)

Pipeline for analysis related to: *"Unraveling the network signatures of oncogenicity in virus-human protein-protein interactions"* by Zambelli, Francesco, Vera Pancaldi, and Manlio De Domenico (Entropy 2025).

## Table of Contents
- [Project Summary](#project-summary)
- [Technical Requirements](#technical-requirements)
- [Data Sources](#data-sources)
- [Pipeline Usage](#pipeline-usage)
- [Citation](#citation)
- [License](#license)

## Project Summary
Study network signatures of oncogenic vs. non-oncogenic viruses. Build multilayer PPI networks. Extract topological features (LVC, Percolation, DCSBM Modularity). Use ML (SVM/UMAP) to classify oncogenic sets.

## Technical Requirements
- **Environment**: Conda `muxvizpy`
- **Core Libraries**:
  - `muxvizpy` (Multilayer analysis)
  - `graph-tool` (Graph algorithms)
  - `pandas`, `numpy`, `scipy` (Data processing)
  - `seaborn`, `matplotlib` (Visualization)
  - `scikit-learn`, `umap-learn` (Machine Learning)

## Data Sources
- **Human Interactome**: BIOSTR edges/nodes (derived from BioGrid).
- **Virus Metadata**: `viruses_metadata.csv` (classification and categorization).
- **Oncogenes**: `cancerGeneList.tsv` (for statistical validation).
- **Viral Targets**: `Virus_data_Enriched_0.7_Neigh_0/` (raw virus-target mappings).

## Pipeline Usage
Run steps via `run_pipeline.py`:

```bash
# 1. Generate viral PPI subnetworks
python run_pipeline.py --step gen-networks

# 2. Create combination index sets
python run_pipeline.py --step gen-indexes

# 3. Compute topological metrics (Multilayer)
python run_pipeline.py --step produce-data

# 4. Statistical analysis & ML
python run_pipeline.py --step analyze
```

## Legacy Notebooks
Original research logic preserved in:
- `0_networks_generation.ipynb`
- `1_data_production.ipynb`
- `2_statistical_analysis.ipynb`
- `3_datadrive_approach.ipynb`
## Citation
If you use this code or data, please cite:
> Zambelli, F.; Pancaldi, V.; De Domenico, M. Unraveling the Network Signatures of Oncogenicity in Virus–Human Protein–Protein Interactions. Entropy 2025, 27, 1248. https://doi.org/10.3390/e27121248

## License
Distributed under the MIT License. See `LICENSE` for more information.
