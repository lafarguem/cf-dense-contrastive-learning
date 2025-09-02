import pandas as pd
import json
from pathlib import Path

decimal_places = {"DICE": (1,3,3), "Hausdorff": (2,1,3), "ASD": (1,1,2)}
metric_direction = {
    "DICE": True,        # higher is better
    "Hausdorff": False,  # lower is better
    "ASD": False         # lower is better
}
def bool_to_truth(x):
    if x is True:
        return "Yes"
    if x is False:
        return "No"
    if x == "No augmentations":
        return "None"
    return x

import pandas as pd

def save_ablation(csv_path, output_path_prefix, output_groups):
    df = pd.read_csv(csv_path)

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
    # Sort by binary lexicographic order
    df = df.sort_values('BinaryKey', ascending=False).reset_index(drop=True)

    for group_idx, group in enumerate(output_groups):
        new_df = pd.DataFrame()
        new_df['Augmentations'] = df['Augmentations']

        metrics = []
        best_values = {}
        for metric_base in group:
            mean_col = f"{metric_base} mean"
            std_col = f"{metric_base} std"
            new_col = f"{{{metric_base}}}"
            new_df[new_col] = [
                f"{m:.{decimal_places[metric_base.split()[0]][1]}f}({int(round(s*10**decimal_places[metric_base.split()[0]][1]))})"
                for m, s in zip(df[mean_col], df[std_col])
            ]
            metrics.append(new_col)

            numeric_values = df[mean_col].values
            if metric_direction[metric_base.split()[0]]:
                best_idx = numeric_values.argmax()
            else:
                best_idx = numeric_values.argmin()
            best_values[new_col] = best_idx

        metric_columns = [
            f"S[table-format={decimal_places[metric[1:-1].split()[0]][0]}.{decimal_places[metric[1:-1].split()[0]][1]}({decimal_places[metric[1:-1].split()[0]][2]})]"
            for metric in metrics
        ]
        column_format = ">{\\centering\\arraybackslash}X" + "!{\\vrule width 1pt}" + "".join(metric_columns)

        latex_lines = []
        latex_lines.append("\\centering")
        latex_lines.append("\\small")
        latex_lines.append("\\rowcolors{2}{white}{gray!15}")
        latex_lines.append(f"\\begin{{tabularx}}{{\\linewidth}}{{{column_format}}}")
        latex_lines.append("Augmentations & " + " & ".join(metrics) + " \\\\")
        latex_lines.append("\\specialrule{1pt}{0pt}{0pt}")

        for idx, row in new_df.iterrows():
            param_str = row['Augmentations']
            values = []
            for metric in metrics:
                val = str(row[metric])
                if idx == best_values[metric]:
                    val = f"\\cellcolor{{green!30}} {val}"
                values.append(val)
            latex_lines.append(f"{param_str} & {' & '.join(values)} \\\\")

        latex_lines.append("\\specialrule{1pt}{0pt}{0pt}")
        latex_lines.append("\\end{tabularx}")

        output_path = f"{output_path_prefix}_group{group_idx+1}.tex"
        with open(output_path, "w") as f:
            f.write("\n".join(latex_lines))

        print(f"LaTeX table saved to {output_path}")

def save_latex(csv_path, output_path, id_to_name):
    df = pd.read_csv(csv_path)

    def format_mean_std(mean, std, decimal_places):
        return f"{mean:.{decimal_places}f}({int(round(std*10**decimal_places))})"

    cols = df.columns
    new_df = pd.DataFrame()

    first_metric_idx = -1

    metrics = []
    parameters = []
    for i,col in enumerate(cols):
        if col == "ID":
            parameters.append('Pre-training')
            new_df["Pre-training"] = df[col].apply(lambda x: id_to_name[str(x)])
        elif col.endswith(" mean"):
            if first_metric_idx < 0:
                first_metric_idx = i
            base = col.replace(" mean", "")
            prefix = base.split()[0]
            key = f"{{{base}}}"
            metrics.append(key)
            std_col = base + " std"
            new_df[key] = [
                format_mean_std(m, s, decimal_places=decimal_places[prefix][1])
                for m, s in zip(df[col], df[std_col])
            ]
        elif not col.endswith(" std"):
            parameters.append(col)
            new_df[col] = df[col].apply(bool_to_truth)

    best_values = {}
    for metric in metrics:
        numeric_values = df[f"{metric[1:-1]} mean"].values
        if metric_direction[metric[1:-1].split()[0]]:  # higher is better
            best_idx = numeric_values.argmax()
        else:
            best_idx = numeric_values.argmin()
        best_values[metric] = best_idx

    metric_columns = [f"S[table-format={decimal_places[metric[1:-1].split()[0]][0]}.{decimal_places[metric[1:-1].split()[0]][1]}({decimal_places[metric[1:-1].split()[0]][2]})]" for metric in metrics]
    column_format = len(parameters)*">{\\centering\\arraybackslash}X" + "!{\\vrule width 1pt}" + "".join(metric_columns)

    latex_lines = []

    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\rowcolors{2}{white}{gray!15}")
    latex_lines.append(f"\\begin{{tabularx}}{{\\linewidth}}{{{column_format}}}")
    latex_lines.append(" & ".join(parameters) + " & " + " & ".join(metrics) + " \\\\")
    latex_lines.append("\\specialrule{1pt}{0pt}{0pt}")

    for idx, row in new_df.iterrows():
        param_str = " & ".join(str(row[p]) for p in parameters)
        values = []
        for metric in metrics:
            val = str(row[metric])
            if idx == best_values[metric]:
                val = f"\\cellcolor{{green!30}} {val}"
            values.append(val)
        latex_lines.append(f"{param_str} & {" & ".join(values)} \\\\")

    latex_lines.append("\\specialrule{1pt}{0pt}{0pt}")
    latex_lines.append("\\end{tabularx}")

    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"LaTeX table saved to {output_path}")

def main():
    output_groups = [['DICE', 'DICE (nf)', 'DICE (pe)', 'Hausdorff 95', 'ASD'], ['Hausdorff 95 (nf)', 'Hausdorff 95 (pe)', 'ASD (nf)', 'ASD (pe)']]
    avoid = ['supervised','unsupervised','large_query']
    full_ablations = ['dvd-ablations', 'mvd-ablations', 'sdvd-ablations', 'smvd-ablations']

    with open("analysis/main_ids.json", "r") as f:
        id_to_name = json.load(f)

    csv_folder = Path('analysis') / 'results'
    output_folder = Path('analysis') / 'latex'

    for csv_file in csv_folder.glob("*.csv"):
        name = csv_file.stem
        if name in avoid:
            continue

        output_file = output_folder / f"{name}_table.tex"
        
        if name in full_ablations:
            save_ablation(str(csv_file), str(output_file)[:-4], output_groups)
        else:
            save_latex(str(csv_file), str(output_file), id_to_name)