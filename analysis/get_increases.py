import pandas as pd
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

csvs = ['dvd-ablations', 'sdvd-ablations', 'mvd-ablations', 'smvd-ablations']
FONTSIZE = 16
# Load CSV
def get_increase(csv):
    df = pd.read_csv(f"{str(csv)}.csv")

    input_cols = [c for c in df.columns if c not in ['ID'] and not c.endswith(" mean") and not c.endswith(" std")]

    def encode_inputs(row):
        encoded = ""
        binary_key = ""
        for i, col in enumerate(input_cols):
            letter = col[0].upper()
            if bool(row[col]):
                encoded += letter
                binary_key += "1"
            else:
                binary_key += "0"
        return encoded if encoded else "", binary_key
    
    df[['Augmentations', 'BinaryKey']] = df.apply(lambda row: pd.Series(encode_inputs(row)), axis=1)

    # Make sure components are sorted alphabetically so comparisons are consistent
    df["Augmentations"] = df["Augmentations"].apply(lambda x: ''.join(sorted(x)))

    # Dictionary to store marginal contributions
    component_diffs = {c: [] for c in ["SD", "R", "C", "E"]}
    component_diffs_std = {c: [] for c in ["SD", "R", "C", "E"]}
    all_diffs = []
    all_diffs_std = []

    # Build a quick lookup for component sets
    value_lookup = dict(zip(df["Augmentations"], df["DSC mean"]))
    std_lookup = dict(zip(df["Augmentations"], df["DSC std"]))

    # For each row, test marginal contribution of each component
    for comp_str, value in value_lookup.items():
        comps = set(comp_str)
        for c in ["R", "C", "E"]:
            if c in comps:
                smaller_set = ''.join(sorted(comps - {c}))
                if smaller_set in value_lookup:
                    diff = 100*(value - value_lookup[smaller_set])
                    diff_std = np.sqrt((100*std_lookup[comp_str])**2 + (100*std_lookup[smaller_set])**2)
                    
                    component_diffs[c].append(diff)
                    component_diffs_std[c].append(diff_std)
                    all_diffs.append(diff)
                    all_diffs_std.append(diff_std)

            if 'S' in comps and 'D' in comps:
                smaller_set = ''.join(sorted(comps - {'S','D'}))
                if smaller_set in value_lookup:
                    diff = 100*(value - value_lookup[smaller_set])
                    diff_std = np.sqrt((100*std_lookup[comp_str])**2 + (100*std_lookup[smaller_set])**2)
                    
                    component_diffs["SD"].append(diff)
                    component_diffs_std["SD"].append(diff_std)
                    all_diffs.append(diff)
                    all_diffs_std.append(diff_std)

    stats = {}
    for c in ["SD", "R", "C", "E"]:
        diffs = component_diffs[c]
        stds = component_diffs_std[c]
        if diffs:
            mean = np.mean(diffs)
            propagated_std = np.sqrt(np.sum(np.array(stds)**2)) / len(stds)
            stats[c] = {"mean": mean, "std": propagated_std}
        else:
            stats[c] = {"mean": None, "std": None}

    # Compute overall average and propagated std
    if all_diffs:
        total_mean = np.mean(all_diffs)
        total_std = np.sqrt(np.sum(np.array(all_diffs_std)**2)) / len(all_diffs_std)
    else:
        total_mean = None
        total_std = None

    print("Component-wise average diff and propagated std:")
    for c, res in stats.items():
        print(f"{c}: mean = {res['mean']} + {res['std']}")

    print("\nOverall total diff average:", total_mean, '+', total_std)
    return stats

if __name__ == "__main__":
    base = Path('analysis/results')

    # Combine data into a structure for plotting
    methods = {
        "DVD-CL": get_increase(base / 'dvd-ablations'),
        "MVD-CL": get_increase(base / 'mvd-ablations'),
        "S-DVD-CL": get_increase(base / 'sdvd-ablations'),
        "S-MVD-CL": get_increase(base / 'smvd-ablations'),
    }

    components = ['Counterfactuals', 'Rotation', 'Cropping', 'Effects']
    n_components = len(components)
    n_methods = len(methods)

    # Prepare the bar positions
    bar_width = 0.2
    x = np.arange(n_components)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each method as a separate bar set
    for i, (method_name, effects) in enumerate(methods.items()):
        means = {key: val['mean'] for key, val in effects.items()}
        stds = {key: val['std'] for key, val in effects.items()}
        offsets = x + i * bar_width
        plt.bar(offsets, means.values(), width=bar_width, label=method_name)

    # Set x-ticks to be in the middle of grouped bars
    plt.xticks(x + (n_methods - 1) * bar_width / 2, components, fontsize=FONTSIZE-2)
    plt.yticks(fontsize=FONTSIZE-2)
    # Labels and title
    plt.xlabel("Augmentation", fontsize=FONTSIZE)
    plt.ylabel("Average DSC Increase", fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE-2)

    # Add grid for readability
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Display plot
    plt.tight_layout()
    plt.savefig('analysis/diffs.png')