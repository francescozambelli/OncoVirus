import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wilcoxon
from umap import UMAP
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
from src.utils import ensure_dir

def load_results(results_dir="data/results"):
    """
    Load topological analysis results from text files.

    Parameters
    ----------
    results_dir : str, optional
        Path to directory containing result subfolders (LVC, percolation, etc.).

    Returns
    -------
    tuple (dict, dict, dict, dict)
        Dictionaries for LVC sizes, critical points, module counts, and modularity scores.
    """
    lvcs, crit_points, mods, mody = {}, {}, {}, {}
    sets = ["n", "n1o", "n2o", "n3o", "o"]
    
    for s in sets:
        lvc_path = os.path.join(results_dir, "LVC", f"{s}.txt")
        if os.path.exists(lvc_path):
            lvcs[s] = np.loadtxt(lvc_path, dtype=int)
            
        perc_path = os.path.join(results_dir, "percolation", f"{s}.txt")
        if os.path.exists(perc_path):
            crit_points[s] = np.loadtxt(perc_path, dtype=float)
            
        mod_path = os.path.join(results_dir, "block_structure", f"{s}_modules.txt")
        if os.path.exists(mod_path):
            mods[s] = np.loadtxt(mod_path, dtype=int)
            
        mody_path = os.path.join(results_dir, "block_structure", f"{s}_modularity.txt")
        if os.path.exists(mody_path):
            mody[s] = np.loadtxt(mody_path, dtype=float)
        
        if s in lvcs:
            print(f"  Loaded {len(lvcs[s])} results for set {s.upper()}")
            
    return lvcs, crit_points, mods, mody

def plot_distributions(lvcs, crit_points, mods, mody, output_path="data/results/distribution_plots.png"):
    """
    Generate boxplots comparing topological metrics across virus sets.

    Parameters
    ----------
    lvcs, crit_points, mods, mody : dict
        Dictionaries containing calculated metrics for each set (N, N1O, etc.).
    output_path : str, optional
        Path where the generated figure will be saved.
    """
    plt.figure(figsize=(16, 8))
    
    metrics = [
        (lvcs, "LVC size", 1),
        (crit_points, "Critical point", 2),
        (mods, "Number of modules", 3),
        (mody, "Modularity", 4)
    ]
    
    for data_dict, title, pos in metrics:
        if not data_dict: continue
        plt.subplot(1, 4, pos)
        
        # Prepare data for seaborn
        plot_data = []
        for label, values in data_dict.items():
            for v in values:
                plot_data.append({"Set": label.upper(), "Value": v})
        
        df = pd.DataFrame(plot_data)
        sns.boxplot(data=df, x="Value", y="Set", hue="Set", palette="plasma", orient="horizontal", legend=False)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.grid(ls="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Distribution plots saved to {output_path}")

def run_ml_classification(lvcs, crit_points, mods, mody):
    """
    Perform UMAP dimensionality reduction and SVM classification.

    Trains an SVM classifier to distinguish between non-oncogenic (N)
    and oncogenic (O) sets based on their topological features.

    Parameters
    ----------
    lvcs, crit_points, mods, mody : dict
        Dictionaries containing calculated metrics.

    Returns
    -------
    float
        The accuracy score of the SVM classifier on the training set.
    """
    # Combine features for UMAP and SVM
    # We focus on 'n' and 'o' sets as in the original notebook for classification
    if "n" not in lvcs or "o" not in lvcs:
        print("Missing 'n' or 'o' sets for classification.")
        return

    # Truncate to the smallest common size if necessary
    min_len = min(len(lvcs["n"]), len(lvcs["o"]))
    
    features_n = np.column_stack([
        lvcs["n"][:min_len], 
        crit_points["n"][:min_len], 
        mods["n"][:min_len], 
        mody["n"][:min_len]
    ])
    
    features_o = np.column_stack([
        lvcs["o"][:min_len], 
        crit_points["o"][:min_len], 
        mods["o"][:min_len], 
        mody["o"][:min_len]
    ])
    
    X = np.vstack([features_n, features_o])
    y = np.array([0] * min_len + [1] * min_len) # 0 for N, 1 for O
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP
    reducer = UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    # SVM
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_scaled, y)
    score = clf.score(X_scaled, y)
    print(f"SVM classification accuracy (on full training set): {score:.4f}")
    
    # Plot UMAP
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:min_len, 0], embedding[:min_len, 1], label="N (Non-onco)", alpha=0.6)
    plt.scatter(embedding[min_len:, 0], embedding[min_len:, 1], label="O (Onco)", alpha=0.6)
    plt.title("UMAP projection of topological features")
    plt.legend()
    save_path = "data/results/umap_projection.png"
    plt.savefig(save_path)
    print(f"UMAP projection plot saved to {save_path}")
    
    return score

if __name__ == "__main__":
    l, c, m, my = load_results()
    if l:
        plot_distributions(l, c, m, my)
        run_ml_classification(l, c, m, my)
