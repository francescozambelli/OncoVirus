import numpy as np
import pandas as pd
import itertools
import os
from operator import itemgetter
from tqdm import tqdm
from src.utils import load_virus_metadata, ensure_dir

try:
    import MuxVizPy as mxp
except ImportError:
    try:
        import muxvizpy as mxp
    except ImportError:
        print("Warning: muxvizpy library not found. Topological analysis will fail.")
        mxp = None

from graph_tool.all import vertex_percolation

class VirusMultiplex_from_dirlist():
    """
    Build a multilayer PPI network from a list of virus directories.

    Each directory must contain an 'edges.csv' file. The class handles
    node mapping across layers and builds the multilayer structure.

    Parameters
    ----------
    dirlist : list of str
        List of paths to directories containing viral PPI edges.

    Attributes
    ----------
    g_list : list of graph_tool.Graph
        List of individual per-layer PPI graphs.
    g_multi : graph_tool.Graph
        Combined multilayer graph with edge 'weight' property indicating layer.
    node_map : dict
        Mapping from protein symbols to unique node indices.
    Layers : int
        Number of viruses (layers) in the multiplex.
    Nodes : int
        Total number of unique proteins in the multiplex.
    """
    def __init__(self, dirlist):
        import graph_tool as gt
        from operator import itemgetter
        self.dirlist = dirlist 
        self.mux_ppi = pd.DataFrame()
        for idir in self.dirlist:
            if not os.path.exists(idir+"/edges.csv"):
                continue
            human_ppi = pd.read_csv(idir+"/edges.csv", header=0, sep=",")
            human_ppi.columns = ["source", "target"]
            human_ppi["layer"] = idir.split("/")[-1]
            self.mux_ppi = pd.concat([self.mux_ppi, human_ppi])

        if self.mux_ppi.empty:
            self.g_list = []
            self.g_multi = None
            self.node_map = {}
            self.Layers = 0
            self.Nodes = 0
            return

        self.mux_ppi = self.mux_ppi.reset_index(drop=True)
        self.Layers = self.mux_ppi["layer"].unique().shape[0]
        self.layer_map = {self.mux_ppi["layer"].unique()[i]: i for i in range(self.Layers) }

        # mapping node names to ids
        self.unique_nodes = pd.concat([self.mux_ppi["source"], self.mux_ppi["target"]]).unique()
        self.node_map = {self.unique_nodes[i]: i for i in range(self.unique_nodes.shape[0]) }

        self.Nodes = self.unique_nodes.shape[0]
        self.Edges = self.mux_ppi.shape[0]

        self.mux_ppi["nodeA"] = itemgetter(*self.mux_ppi["source"].to_numpy())(self.node_map)
        self.mux_ppi["nodeB"] = itemgetter(*self.mux_ppi["target"].to_numpy())(self.node_map)
        self.mux_ppi["l"] = itemgetter(*self.mux_ppi["layer"].to_numpy())(self.layer_map)

        self.g_list = []
        for l in range(self.Layers):
            tmp = self.mux_ppi[self.mux_ppi["l"]==l].reset_index()[["nodeA", "nodeB"]]
            tmp = tmp.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
            g = gt.Graph(directed=False)
            g.add_edge_list(tmp.values)
            g.add_vertex(self.Nodes-g.num_vertices())
            self.g_list.append(g)

        tmp_multi = self.mux_ppi.reset_index()[["nodeA", "nodeB"]]
        tmp_multi = tmp_multi.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
        self.g_multi = gt.Graph(directed=False)
        self.g_multi.add_edge_list(tmp_multi.values)
        ep_weight_lay = []
        for l in range(self.Layers):
            count = self.mux_ppi["l"].value_counts().get(l, 0)
            ep_weight_lay.extend([l] * count)
        self.g_multi.edge_properties["weight"] = self.g_multi.new_edge_property("int", ep_weight_lay)

def generate_combination_indexes(raw_data_dir="data/raw", output_dir="data/processed/MultilayerIndexes", n_iters=256, seed=100):
    """
    Generate reproducible combination sets for oncogenic vs. non-oncogenic comparison.

    Creates random sets of 4 viruses with varying proportions of oncogenic
    viruses (N, N1O, N2O, N3O, O). Results are saved as index files.

    Parameters
    ----------
    raw_data_dir : str, optional
        Path to directory containing virus metadata.
    output_dir : str, optional
        Directory where index files will be saved.
    n_iters : int, optional
        Number of random combinations to generate per set.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple (dict, numpy.ndarray)
        Dictionary of generated index lists and the array of virus names.
    """
    virus_metadata = load_virus_metadata(os.path.join(raw_data_dir, "viruses_metadata.csv"))
    
    # Identify viruses based on SyntheticViruses folder structure (to ensure consistency)
    synthetic_base = "data/processed/SyntheticViruses/original"
    if not os.path.exists(synthetic_base):
        print("Error: Synthetic viruses must be generated first.")
        return
        
    virus_names = np.sort(os.listdir(synthetic_base))
    virus_onco = virus_metadata[virus_metadata["isOncogenic"] == True].virus.unique()
    virus_nonco = virus_metadata[virus_metadata["isOncogenic"] == False].virus.unique()

    virus_onco_idx = np.where(np.isin(virus_names, virus_onco))[0]
    virus_nonco_idx = np.where(np.isin(virus_names, virus_nonco))[0]

    np.random.seed(seed)
    ensure_dir(output_dir)

    # N: 4 non-oncogenic
    n_idx = np.array([np.random.choice(virus_nonco_idx, 4, replace=False) for _ in range(n_iters)])
    
    # N1O: 3 non-onco + 1 onco
    n1o_idx = []
    for _ in range(n_iters):
        onco_pick = np.random.choice(virus_onco_idx, 1, replace=False)
        nonco_pick = np.random.choice(virus_nonco_idx, 3, replace=False)
        n1o_idx.append(np.concatenate([nonco_pick, onco_pick]))
    n1o_idx = np.array(n1o_idx)

    # N2O: 2 non-onco + 2 onco
    n2o_idx = []
    for _ in range(n_iters):
        onco_pick = np.random.choice(virus_onco_idx, 2, replace=False)
        nonco_pick = np.random.choice(virus_nonco_idx, 2, replace=False)
        n2o_idx.append(np.concatenate([nonco_pick, onco_pick]))
    n2o_idx = np.array(n2o_idx)

    # N3O: 1 non-onco + 3 onco
    n3o_idx = []
    for _ in range(n_iters):
        onco_pick = np.random.choice(virus_onco_idx, 3, replace=False)
        nonco_pick = np.random.choice(virus_nonco_idx, 1, replace=False)
        n3o_idx.append(np.concatenate([nonco_pick, onco_pick]))
    n3o_idx = np.array(n3o_idx)

    # O: All combinations of 4 onco viruses (there are 8 total, so 8C4 = 70)
    comb = list(itertools.combinations(range(len(virus_onco_idx)), 4))
    o_idx = np.array([list(virus_onco_idx[list(c)]) for c in comb])
    if len(o_idx) > n_iters:
        o_idx = o_idx[np.random.choice(len(o_idx), n_iters, replace=False)]

    lists = {"n": n_idx, "n1o": n1o_idx, "n2o": n2o_idx, "n3o": n3o_idx, "o": o_idx}
    for name, data in lists.items():
        save_path = os.path.join(output_dir, f"{name}.txt")
        np.savetxt(save_path, data, fmt="%d")
        print(f"  Saved {name.upper()} indexes ({len(data)} combinations) to {save_path}")
    
    return lists, virus_names

def run_topological_analysis(index_dir="data/processed/MultilayerIndexes", 
                            synthetic_dir="data/processed/SyntheticViruses/original",
                            results_dir="data/results"):
    """
    Run topological analysis for all generated combination sets.

    Computes Largest Viable Component (LVC), Percolation Critical Points,
    and DCSBM Modularity for each multilayer combination.

    Parameters
    ----------
    index_dir : str, optional
        Directory containing combination index files.
    synthetic_dir : str, optional
        Directory containing individual virus PPI networks.
    results_dir : str, optional
        Directory where analysis results will be saved.
    """
    if mxp is None:
        raise ImportError("MuxVizPy is required for topological analysis.")

    virus_names = np.sort(os.listdir(synthetic_dir))
    virus_dict = {i: vname for i, vname in enumerate(virus_names)}
    
    sets = ["n", "n1o", "n2o", "n3o", "o"]
    
    for s_name in sets:
        print(f"\n- Processing set: {s_name.upper()}")
        idx_file = os.path.join(index_dir, f"{s_name}.txt")
        if not os.path.exists(idx_file):
            continue
        
        idx_list = np.loadtxt(idx_file, dtype=int)
        
        lvc_sizes = []
        perc_points = []
        mods_list = []
        mody_list = []

        for i in tqdm(range(len(idx_list)), desc=f"Analyzing set {s_name}"):
            paths = [os.path.join(synthetic_dir, virus_dict[idx]) for idx in idx_list[i]]
            net = VirusMultiplex_from_dirlist(paths)
            
            if net.Layers == 0: continue

            # LVC
            lvc_curr = mxp.topology.get_multi_LVC(net.g_list, printt=False)
            lvc_sizes.append(len(lvc_curr) if isinstance(lvc_curr, (list, np.ndarray)) else 1)
            
            # Percolation
            tensor = mxp.utils.parsing.get_node_tensor_from_network_list(net.g_list)
            order = mxp.versatility.get_multi_RW_centrality_edge_colored(tensor)
            order = order.sort_values("vers")["phy nodes"].to_numpy()
            g_agg = mxp.utils.parsing.get_aggregate_network(tensor)
            
            perc_agg_2 = vertex_percolation(g_agg, order, second=True)[0]
            max_perc = np.argmax(perc_agg_2) / len(perc_agg_2) if len(perc_agg_2) > 0 else 0
            perc_points.append(max_perc)
            
            # DCSBM
            mod_res = mxp.mesoscale.get_mod(g_multi=net.g_multi, n_iter=1)
            mods_list.append(mod_res[0])
            mody_list.append(mod_res[1])

        # Save results
        ensure_dir(os.path.join(results_dir, "LVC"))
        ensure_dir(os.path.join(results_dir, "percolation"))
        ensure_dir(os.path.join(results_dir, "block_structure"))
        
        np.savetxt(os.path.join(results_dir, "LVC", f"{s_name}.txt"), lvc_sizes, fmt="%d")
        np.savetxt(os.path.join(results_dir, "percolation", f"{s_name}.txt"), perc_points, fmt="%.5f")
        np.savetxt(os.path.join(results_dir, "block_structure", f"{s_name}_modules.txt"), mods_list, fmt="%d")
        np.savetxt(os.path.join(results_dir, "block_structure", f"{s_name}_modularity.txt"), mody_list, fmt="%.5f")

if __name__ == "__main__":
    # Test combination generation
    generate_combination_indexes()
