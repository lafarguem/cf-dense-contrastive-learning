import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from tqdm.auto import tqdm

# Enable LaTeX rendering globally
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.unicode_minus": False,
})

FONTSIZE = 16

# ---------------------- SETTINGS ----------------------
decimal_places = {"DSC": (1, 3, 3, 100), "Hausdorff": (2, 1, 3, 1), "ASD": (1, 1, 2, 1)}
metric_direction = {
    "DSC": 1,        # higher is better
    "Hausdorff": -1,  # lower is better
    "ASD": -1         # lower is better
}
order_by = ['None', 'No augmentations', 'SimCLR', 'VADeR', 'Segmentation', 'SSDCL', 'DVD-CL\\textsuperscript{\\textdagger}',
            'MVD-CL\\textsuperscript{\\textdagger}', 'S-DVD-CL\\textsuperscript{\\textdagger}',
            'S-MVD-CL\\textsuperscript{\\textdagger}', 'DVD-CL', 'MVD-CL', 'S-DVD-CL', 'S-MVD-CL']

# output_groups = [
#     ['DSC', 'DSC (nf)', 'DSC (pe)', 'Hausdorff 95', 'ASD'],
#     ['Hausdorff 95 (nf)', 'Hausdorff 95 (pe)', 'ASD (nf)', 'ASD (pe)']
# ]

output_groups = [
    ['DSC', 'DSC (nf)', 'DSC (pe)'],
    ['Hausdorff 95', 'ASD']
]
metrics_to_plot = ['DSC', 'DSC (nf)', 'DSC (pe)', 'Hausdorff 95', 'ASD']

def sort_binary_df(df, col="BinaryKey", weights = (1,1,1,1,1), priority=(1, 2, 3, 5, 4)):
    df = df.copy()
    def one_sum(binary):
        total = 0
        for i, weight in enumerate(weights):
            total += int(binary[i])*weight
        return total

    df["_ones"] = df[col].apply(one_sum)

    for i in range(5):
        df[f"_b{i+1}"] = df[col].str[i].astype(int)

    sort_cols = ["_ones"] + [f"_b{i}" for i in priority]
    ascending_flags = [True] + [False] * len(priority)

    df = df.sort_values(by=sort_cols, ascending=ascending_flags).reset_index(drop=True)

    df = df.drop(columns=sort_cols)

    return df

def sort_binary_numbers(priority=(1, 2, 3, 5, 4)):
    def sort_key(num):
        # Convert string like "10101" → list of ints [1,0,1,0,1]
        bits = [int(b) for b in num]
        # First key: number of ones
        ones = sum(bits)
        # Second key: negative bits in priority order
        prio_bits = tuple(-bits[p-1] for p in priority)
        return (ones, prio_bits)

    return sort_key
    return sorted(numbers, key=sort_key)



grouped_plots = False  # <--- TOGGLE: True = grouped seaborn plots, False = individual PNGs
sns.set_theme(style="whitegrid", context="paper")  # Make seaborn plots cleaner

def bool_to_truth(x):
    if x is True:
        return "Yes"
    if x is False:
        return "No"
    if x == "No augmentations":
        return "None"
    if x == 'No pre-training':
        return "None"
    return x

# ------------------------------------------------------
# ABLATION PLOTS
# ------------------------------------------------------
def save_ablation(csv_path, output_path_prefix, output_groups, relative_to, baseline_performance):
    """Generate bar plots (or grouped seaborn plots) for ablation studies."""
    df = pd.read_csv(csv_path)

    # Encode augmentations into readable labels
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
        return encoded if encoded else "-", binary_key
    
    df[['Augmentations', 'BinaryKey']] = df.apply(lambda row: pd.Series(encode_inputs(row)), axis=1)

    df = sort_binary_df(df)

    # Generate grouped or individual plots per metric group
    for group_idx, group in enumerate(output_groups):
        baseline_means = {}
        melted_data = []

        for metric in group:
            mean_col = f"{metric} mean"
            std_col = f"{metric} std"
            if mean_col not in df.columns or std_col not in df.columns:
                continue

            coeff = 1
            for m, val in decimal_places.items():
                if metric.startswith(m):
                    if metric_direction.get(m,1) > 0:
                        legend_loc = 'lower right'
                    else:
                        legend_loc = 'upper right'
                    coeff = val[3]
                    break
            baseline = df[mean_col].iloc[0]
            baseline_means[metric] = baseline
            # means = coeff * (df[mean_col] - baseline)
            means = coeff*(df[mean_col] - relative_to[metric])
            stds = coeff*df[std_col]


            baseline_name = 'No augmentations'
            # Collect data for seaborn grouped plot
            start = 0
            # for i, label in enumerate(df['Augmentations']):
            #     # if i == 0 and label in ['None', 'No pre-training']:
            #     #     start = 1
            #     if label not in ['None', 'No pre-training']:
            #         melted_data.append({
            #             "Augmentation": label,
            #             "Metric": metric,
            #             "Mean $\\Delta$": means.iloc[i],
            #             "STD": stds.iloc[i]
            #         })

            colors_line = [('green','dashed'), ('red',(5, (10,3))), ('purple', (0, (2,3))), ('orange','dashdot')]

            if not grouped_plots:
                plt.figure(figsize=(8, 5))
                plt.bar(df['Augmentations'].iloc[start:], means.iloc[start:], yerr=stds.iloc[start:], capsize=4,
                        color='skyblue', edgecolor='black')
                plt.axhline(0, color='gray', linestyle='--', linewidth=1)
                for i,(name,val) in enumerate(baseline_performance.items()):
                    color, linestyle = colors_line[i%len(colors_line)]
                    plt.axhline(coeff*(val[metric] - relative_to[metric]), color=color, linestyle=linestyle, linewidth=1,label=name)
                plt.ylabel(f"{metric} ($\\Delta$ vs No Pre-training)", fontsize=FONTSIZE+2)
                plt.legend(fontsize=FONTSIZE-1, loc=legend_loc)
                plt.xticks(rotation=45, fontsize=FONTSIZE)
                plt.yticks(fontsize=FONTSIZE)
                plt.xlabel('Augmentations', fontsize=FONTSIZE+2)
                plt.tight_layout()
                output_path = f"{output_path_prefix}_{metric.replace(' ', '_')}.png"
                plt.savefig(output_path, dpi=300)
                plt.close()

        # If grouped_plots = True → make a combined seaborn plot
        if grouped_plots and melted_data:
            melted_df = pd.DataFrame(melted_data)
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(data=melted_df, x="Augmentation", y="Mean $\\Delta$",
                             hue="Metric", errorbar=None, capsize=0.1, palette="Set2")
            bars = ax.patches[:len(melted_df)]
            for i, bar in enumerate(bars):
                std = melted_df.iloc[i]["STD"]
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.errorbar(x, y, yerr=std, fmt="none", ecolor="black", capsize=4)
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            plt.xticks(rotation=45)
            plt.legend(title="Metric")
            plt.tight_layout()
            output_path = f"{output_path_prefix}_group{group_idx+1}_grouped.png"
            plt.savefig(output_path, dpi=300)
            plt.close()

# ------------------------------------------------------
# PRE-TRAINING COMPARISON PLOTS
# ------------------------------------------------------
def save_latex(csv_path, output_path, id_to_name, relative_to):
    """Generate bar plots (or grouped seaborn plots) for pre-training comparisons."""
    df = pd.read_csv(csv_path)

    if "ID" in df.columns:
        df["Pre-training"] = df["ID"].apply(lambda x: id_to_name.get(str(x), str(x)))

    x_label = 'Augmentations'

    # Sort by defined order if applicable
    if 'Pre-training' in df.columns:
        df['Order'] = df['Pre-training'].apply(lambda x: order_by.index(x) if x in order_by else len(order_by))
        df = df.sort_values('Order').reset_index(drop=True)
        x_label = 'Pre-training'

    metrics = [col.replace(" mean", "") for col in df.columns if col.endswith(" mean")]
    melted_data = []

    baseline_name = f'No {x_label.lower()}'

    for metric in metrics_to_plot:
        mean_col = f"{metric} mean"
        std_col = f"{metric} std"
        if mean_col not in df.columns:
            continue

        coeff = 1
        for m, val in decimal_places.items():
            if metric.startswith(m):
                coeff = val[3]
                break

        baseline = df[mean_col].iloc[0]
        # means = coeff*(df[mean_col] - baseline)
        if std_col in df.columns:
            means = coeff*(df[mean_col] - relative_to[metric])
            stds = coeff*df[std_col]
        else:
            means = coeff*df[mean_col]
            stds = None

        start = 0
        if x_label == 'Pre-training' and df['Pre-training'].iloc[0] in ['None', 'No pre-training', 'No augmentations']:
            start = 1

        # Collect data for grouped plots
        for i in range(start, len(df)):
            melted_data.append({
                x_label: df[x_label].iloc[i],
                "Metric": metric,
                "Mean $\\Delta$": means.iloc[i],
                "STD": stds.iloc[i] if stds is not None else None
            })


        if not grouped_plots:
            plt.figure(figsize=(10, 5))
            if stds is not None:
                plt.bar(df[x_label].iloc[start:], means.iloc[start:], yerr=stds.iloc[start:], capsize=4,
                        color='lightgreen', edgecolor='black')
                plt.ylabel(f"{metric} ($\\Delta$ vs No Pre-training)", fontsize=FONTSIZE+2)
            else:
                plt.bar(df[x_label].iloc[start:], means.iloc[start:], capsize=4,
                        color='lightgreen', edgecolor='black')
                plt.ylabel(f"{metric}", fontsize=FONTSIZE+2)
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            plt.xticks(rotation=45, ha="right", fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.xlabel(x_label, fontsize=FONTSIZE+2)
            plt.tight_layout()
            plot_path = output_path.replace("_table.tex", f"_{metric}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

    # If grouped_plots = True → make a combined seaborn plot
    if grouped_plots and melted_data:
        melted_df = pd.DataFrame(melted_data)
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=melted_df, x=x_label, y="Mean $\\Delta$",
                         hue="Metric", errorbar=None, capsize=0.1, palette="Set2")
        bars = ax.patches[:len(melted_df)]
        for i, bar in enumerate(bars):
            std = melted_df.iloc[i]["STD"]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.errorbar(x, y, yerr=std, fmt="none", ecolor="black", capsize=4)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xticks(rotation=45)
        plt.legend(title="Metric")
        plt.tight_layout()
        plot_path = output_path.replace("_table.tex", "_grouped.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

# ------------------------------------------------------
# MAIN DRIVER
# ------------------------------------------------------
def main():
    print("generating plots")
    avoid = ['supervised', 'unsupervised', 'large_query']
    full_ablations = ['dvd-ablations', 'mvd-ablations', 'sdvd-ablations', 'smvd-ablations']

    with open("analysis/main_ids.json", "r") as f:
        id_to_name = json.load(f)

    csv_folder = Path('analysis') / 'results'
    output_folder = Path('analysis') / 'plots'
    output_folder.mkdir(parents=True, exist_ok=True)

    baseline_names = ['SimCLR', 'Segmentation', 'SSDCL', 'VADeR']
    baseline_performance = {name: {} for name in baseline_names}
    relative_to = {}

    bl_df = pd.read_csv(csv_folder / 'mains.csv')
    bl_df['ID'] = bl_df['ID'].apply(lambda x: id_to_name[str(x)])
    columns = bl_df.columns
    for i,row in bl_df.iterrows():
        if row['ID'] == 'None':
            for col in columns:
                if col.endswith('mean'):
                    relative_to[col[:-5]] = row[col]
            continue
        if row['ID'] not in baseline_names:
            continue
        for col in columns:
            if col.endswith('mean'):
                baseline_performance[row['ID']][col[:-5]] = row[col]

    bl_df = pd.read_csv(csv_folder / 'mains-boundary.csv')
    bl_df['ID'] = bl_df['ID'].apply(lambda x: id_to_name[str(x)])
    columns = bl_df.columns
    for i,row in bl_df.iterrows():
        if row['ID'] not in baseline_names:
            continue
        for col in columns:
            if col.endswith('mean'):
                baseline_performance[row['ID']][col[:-5]] = row[col]

    for csv_file in tqdm(csv_folder.glob("*.csv")):
        name = csv_file.stem
        if name in avoid:
            continue

        output_file = output_folder / f"{name}_table.tex"

        if name in full_ablations:
            save_ablation(str(csv_file), str(output_folder / name), output_groups, relative_to, baseline_performance)
        else:
            save_latex(str(csv_file), str(output_file), id_to_name, relative_to)

if __name__ == "__main__":
    main()
