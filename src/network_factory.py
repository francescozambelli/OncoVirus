import numpy as np
import pandas as pd
import graph_tool as gt
from operator import itemgetter
from tqdm import tqdm
import os
from src.utils import load_human_ppi, load_virus_metadata, ensure_dir

def create_synthetic_virus(human_g, virus_targeted_proteins, human_map, human_nodes_len):
    """
    Extract subnetwork of nodes directly targeted by virus and their first neighbors.

    Parameters
    ----------
    human_g : graph_tool.Graph
        Full human Protein-Protein Interaction (PPI) network.
    virus_targeted_proteins : list of str
        List of human protein symbols targeted by the virus.
    human_map : dict
        Mapping from protein symbols to integer node indices in human_g.
    human_nodes_len : int
        Total number of nodes in the human interactome.

    Returns
    -------
    tuple (graph_tool.Graph, numpy.ndarray)
        The pruned viral PPI subnetwork and the original indices of its nodes.
    """
    try:
        vtp = itemgetter(*virus_targeted_proteins)(human_map)
    except KeyError as e:
        # Some proteins might not be in the human map
        valid_proteins = [p for p in virus_targeted_proteins if p in human_map]
        if not valid_proteins:
            return None, None
        vtp = itemgetter(*valid_proteins)(human_map)

    first_neigh_nodes_rep = []
    if isinstance(vtp, (int, np.int64)):
        vtp = [vtp]
    
    for vi in vtp:
        first_neigh_nodes_rep.append(human_g.get_all_neighbors(vi))
    
    if not first_neigh_nodes_rep:
        first_neigh_nodes = np.unique(vtp)
    else:
        first_neigh_nodes = np.unique(np.concatenate([vtp, np.concatenate(first_neigh_nodes_rep)]))

    neighbors_mask = np.isin(np.arange(human_nodes_len), first_neigh_nodes)

    # Label nodes to retrieve original indices later
    if "labels" not in human_g.vertex_properties:
        labels = human_g.new_vertex_property("int", np.arange(human_nodes_len))
        human_g.vertex_properties["labels"] = labels
    
    gw = gt.GraphView(human_g, vfilt=neighbors_mask)
    gf = gt.Graph(gw, prune=True)
    original_index_nodes = gf.vp["labels"].get_array()
    
    return gf, original_index_nodes

def generate_all_viral_networks(raw_data_dir="data/raw", output_dir="data/processed/SyntheticViruses"):
    """
    Iterate over viruses in metadata and generate their respective PPI subnetworks.

    Parameters
    ----------
    raw_data_dir : str, optional
        Path to directory containing human interactome and virus metadata.
    output_dir : str, optional
        Directory where generated viral networks will be stored.
    """
    human_g, human_nodes, human_map, human_map_rev = load_human_ppi(
        os.path.join(raw_data_dir, "BIOSTR_homo_sapiens.edges"),
        os.path.join(raw_data_dir, "BIOSTR_homo_sapiens.nodes")
    )
    virus_metadata = load_virus_metadata(os.path.join(raw_data_dir, "viruses_metadata.csv"))
    
    # Target folder for original virus data
    # Note: Using the original path as requested or where the files are currently
    source_target_folder = "data/Virus_data_Enriched_0.7_Neigh_0/"
    
    neigh_ord_1_indexes = virus_metadata[virus_metadata["neigh_order"] == 1].index.values
    
    ensure_dir(os.path.join(output_dir, "original"))

    for vi in tqdm(neigh_ord_1_indexes, desc="Generating Viral Networks"):
        virus_name = virus_metadata.loc[vi, "virus"]
        nodes_file = os.path.join(source_target_folder, virus_name, "nodes.csv")
        
        if not os.path.exists(nodes_file):
            print(f"Warning: Missing nodes file for {virus_name}")
            continue
            
        read_nodes = pd.read_csv(nodes_file)
        virus_targeted_proteins = list(read_nodes[read_nodes["type"] == 1].node)
        
        g0, n0 = create_synthetic_virus(human_g, virus_targeted_proteins, human_map, len(human_nodes))
        
        if g0 is None:
            continue
            
        save_path = os.path.join(output_dir, "original", virus_name)
        ensure_dir(save_path)
        
        provv_dict = dict(zip(np.arange(len(n0)), n0))
        g_df = pd.DataFrame(g0.get_edges())
        g_df["source"] = g_df[0].map(provv_dict).map(human_map_rev)
        g_df["target"] = g_df[1].map(provv_dict).map(human_map_rev)
        
        np.savetxt(X=np.array(n0), fname=os.path.join(save_path, "nodes.txt"), fmt="%d")
        g_df[["source", "target"]].to_csv(os.path.join(save_path, "edges.csv"), index=False)

if __name__ == "__main__":
    generate_all_viral_networks()
