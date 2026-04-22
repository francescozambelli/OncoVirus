import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path just in case
sys.path.append(os.path.abspath("src"))

# Defer imports to avoid early ModuleNotFoundError
# from src.network_factory import generate_all_viral_networks
# from src.analyzer import generate_combination_indexes, run_topological_analysis
# from src.statistics import load_results, plot_distributions, run_ml_classification

def main():
    parser = argparse.ArgumentParser(description="OncoVirus Analysis Pipeline")
    parser.add_argument("--step", choices=["all", "gen-networks", "gen-indexes", "produce-data", "analyze"], 
                        default="all", help="Pipeline step to run")
    parser.add_argument("--iters", type=int, default=256, help="Number of iterations for combination sets")
    
    args = parser.parse_args()

    if args.step in ["all", "gen-networks"]:
        print("\n>>> Step 1: Generating Viral PPI Networks...")
        from network_factory import generate_all_viral_networks
        generate_all_viral_networks()

    if args.step in ["all", "gen-indexes"]:
        print("\n>>> Step 2: Generating Combination Indexes...")
        from analyzer import generate_combination_indexes
        generate_combination_indexes(n_iters=args.iters)

    if args.step in ["all", "produce-data"]:
        print("\n>>> Step 3: Running Topological Analysis (this might take a while)...")
        from analyzer import run_topological_analysis
        try:
            run_topological_analysis()
        except ImportError as e:
            print(f"Error: {e}")
            if args.step != "all": sys.exit(1)

    if args.step in ["all", "analyze"]:
        print("\n>>> Step 4: Running Statistical Analysis and ML...")
        from statistics import load_results, plot_distributions, run_ml_classification
        lvcs, crit_points, mods, mody = load_results()
        if lvcs:
            plot_distributions(lvcs, crit_points, mods, mody)
            run_ml_classification(lvcs, crit_points, mods, mody)
        else:
            print("Error: No results found to analyze. Run 'produce-data' first.")

    print("\nPipeline execution finished.")

if __name__ == "__main__":
    main()
