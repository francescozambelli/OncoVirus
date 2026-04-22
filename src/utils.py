import pandas as pd
import numpy as np
import graph_tool as gt
import os

def load_node_map(path="data/raw/node_map.csv"):
    """
    Load mapping between protein symbols and numeric indices.

    Parameters
    ----------
    path : str, optional
        Path to the node map CSV file.

    Returns
    -------
    dict
        A dictionary mapping protein symbols to 0-indexed integer IDs.
    """
    node_map_df = pd.read_csv(path)
    node_map_dict = {k: (v-1) for k, v in zip(node_map_df["Prot"], node_map_df["Index"])}
    return node_map_dict

def load_virus_metadata(path="data/raw/viruses_metadata.csv"):
    """
    Load virus metadata including oncogenic status and classification.

    Parameters
    ----------
    path : str, optional
        Path to the virus metadata CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing virus properties.
    """
    return pd.read_csv(path, header=0, sep=";")

def load_human_ppi(edges_path="data/raw/BIOSTR_homo_sapiens.edges", nodes_path="data/raw/BIOSTR_homo_sapiens.nodes"):
    """
    Load the human Protein-Protein Interaction (PPI) network from BIOSTR files.

    Parameters
    ----------
    edges_path : str, optional
        Path to the edges file.
    nodes_path : str, optional
        Path to the nodes file.

    Returns
    -------
    tuple (graph_tool.Graph, pandas.DataFrame, dict, dict)
        The human interactome graph, nodes DataFrame, symbol-to-index map, and index-to-symbol map.
    """
    human_ppi = pd.read_csv(edges_path, sep=" ", header=None)
    human_ppi.columns = ["source", "target", "weight"]
    
    human_nodes = pd.read_csv(nodes_path, sep=" ")
    human_map = dict(zip(human_nodes['nodeSymbol'], np.arange(len(human_nodes))))
    human_map_rev = dict(zip(human_nodes["nodeID"], human_nodes["nodeSymbol"]))
    
    human_g = gt.Graph(directed=False)
    human_g.add_edge_list(human_ppi.values)
    
    return human_g, human_nodes, human_map, human_map_rev

def ensure_dir(directory):
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    directory : str
        The path to the directory.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
