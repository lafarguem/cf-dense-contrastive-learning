import pandas as pd
import json
from pathlib import Path

decimal_places = {"DSC": (2,2,4,100), "Hausdorff": (2,1,3,1), "ASD": (1,2,3,1), "K-means": (2,2,4,1), '$d$/$\\sigma$': (2,2,4,1), 'Probe': (2,2,None,100)}
metric_direction = {
    "DSC": True,         # higher is better
    "Hausdorff": False,  # lower is better
    "ASD": False         # lower is better
}
order_by = ['None', 'SimCLR', 'VADeR', 'Segmentation', 'SSDCL', 'DVD-CL\\textsuperscript{\\textdagger}', 'MVD-CL\\textsuperscript{\\textdagger}', 
            'S-DVD-CL\\textsuperscript{\\textdagger}', 'S-MVD-CL\\textsuperscript{\\textdagger}', 'DVD-CL', 'MVD-CL', 'S-DVD-CL', 'S-MVD-CL']

def bool_to_truth(x):
    if x is True:
        return "Yes"
    if x is False:
        return "No"
    if x == "No augmentations":
        return "None"
    return x


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

    df = sort_binary_df(df)

    for group_idx, group in enumerate(output_groups):
        new_df = pd.DataFrame()
        new_df['Augmentations'] = df['Augmentations']

        metrics = []
        best_values = {}
        metric_columns = []
        for metric_base in group:
            dec_place = decimal_places[metric_base.split()[0]]
            mean_col = f"{metric_base} mean"
            std_col = f"{metric_base} std"
            new_col = f"{{{metric_base}}}"
            if std_col in df.columns:
                new_df[new_col] = [
                    f"{m*dec_place[3]:.{dec_place[1]}f}({int(round(s*dec_place[3]*10**dec_place[1]))})"
                    for m, s in zip(df[mean_col], df[std_col])
                ]
                metric_columns.append(f"S[table-format={decimal_places[new_col[1:-1].split()[0]][0]}.{decimal_places[new_col[1:-1].split()[0]][1]}({decimal_places[new_col[1:-1].split()[0]][2]})]")
            else:
                new_df[new_col] = [
                    f"{m*dec_place[3]:.{dec_place[1]}f}"
                    for m in df[mean_col]
                ]
                metric_columns.append(f"S[table-format={decimal_places[new_col[1:-1].split()[0]][0]}.{decimal_places[new_col[1:-1].split()[0]][1]}]")

            metrics.append(new_col)

            numeric_values = df[mean_col].values
            if metric_direction.get(metric_base.split()[0],True):
                best_idx = numeric_values.argmax()
            else:
                best_idx = numeric_values.argmin()
            best_values[new_col] = best_idx

        column_format = ">{\\centering\\arraybackslash}X" + "!{\\vrule width 1pt}" + "".join(metric_columns)

        latex_lines = []
        latex_lines.append("\\centering")
        latex_lines.append("\\small")
        latex_lines.append("\\rowcolors{2}{white}{gray!15}")
        latex_lines.append(f"\\begin{{tabularx}}{{\\linewidth}}{{{column_format}}}")
        latex_lines.append("& " + " & ".join(metrics) + " \\\\")
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

def save_latex(csv_path, output_path, id_to_name):
    df = pd.read_csv(csv_path)

    def format_mean_std(mean, std, decimal_places):
        return f"{mean*decimal_places[3]:.{decimal_places[1]}f}({int(round(std*decimal_places[3]*10**decimal_places[1]))})"

    cols = df.columns
    new_df = pd.DataFrame()

    if 'ID' in cols:
        df['Order'] = df['ID'].apply(lambda x: order_by.index(id_to_name[str(x)]))
        df = df.sort_values('Order').reset_index(drop=True)
        df.drop(columns=['Order'])

    first_metric_idx = -1

    metric_columns = []

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
            if std_col in cols:
                new_df[key] = [
                    format_mean_std(m, s, decimal_places=decimal_places[prefix])
                    for m, s in zip(df[col], df[std_col])
                ]
                metric_columns.append(f"S[table-format={decimal_places[prefix][0]}.{decimal_places[prefix][1]}({decimal_places[prefix][2]})]")
            else:
                new_df[key] = [
                    f"{m*decimal_places[prefix][3]:.{decimal_places[prefix][1]}f}"
                    for m in df[col]
                ]
                metric_columns.append(f"S[table-format={decimal_places[prefix][0]}.{decimal_places[prefix][1]}]")
        elif not col.endswith(" std"):
            parameters.append(col)
            new_df[col] = df[col].apply(bool_to_truth)

    best_values = {}
    for metric in metrics:
        numeric_values = df[f"{metric[1:-1]} mean"].values
        if metric_direction.get(metric[1:-1].split()[0],True):  # higher is better
            best_idx = numeric_values.argmax()
        else:
            best_idx = numeric_values.argmin()
        best_values[metric] = best_idx

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

def main():
    print("generating latex")
    output_groups = [['DSC', 'DSC (nf)', 'DSC (pe)', 'Hausdorff 95', 'ASD'], ['Hausdorff 95 (nf)', 'Hausdorff 95 (pe)', 'ASD (nf)', 'ASD (pe)', 'Probe DSC']]
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