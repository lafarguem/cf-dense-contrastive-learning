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

def main():
    avoid = ['supervised','unsupervised','large_query']

    with open("analysis/main_ids.json", "r") as f:
        id_to_name = json.load(f)

    def save_latex(csv_path, output_path):
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

    csv_folder = Path('analysis') / 'results'
    output_folder = Path('analysis') / 'latex'

    for csv_file in csv_folder.glob("*.csv"):
        name = csv_file.stem
        if name in avoid:
            continue

        output_file = output_folder / f"{name}_table.tex"
        
        save_latex(str(csv_file), str(output_file))