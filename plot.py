import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

import os
import glob
import argparse
import math

import importlib.util

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_VIZ_DIR = "visualizations"

def pretty(text):
    """Convert a string into a consistent format for
    presentation in a matplotlib pyplot:
    this version looks like: One Two Three Four
    """

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.strip()
    prev_c = None
    out_str = []
    for c in text:
        if prev_c is not None and \
                prev_c.islower() and c.isupper():
            out_str.append(" ")
            prev_c = " "
        if prev_c is None or prev_c == " ":
            c = c.upper()
        out_str.append(c)
        prev_c = c
    return "".join(out_str)

def annotate_plot(ax, data, method_name):
    """Annotate the plot with values for a specific method."""
    method_data = data[data["method"] == method_name]
    for line in ax.get_lines():
        if line.get_label() == method_name:
            xys = {}
            for x, y in zip(method_data["examples_per_class"], method_data["value"]):
                if x not in xys.keys():
                    xys[x] = []
                xys[x].append(y)
            for x, ys in xys.items():
                mean_y = sum(ys) / len(ys)
                ax.annotate(f"{mean_y:.3f}", xy=(x, mean_y), textcoords="offset points", xytext=(17, -12), ha='center')

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--config", type=str, default=None)

    config_name = parser.parse_args().config
    config_file_path = os.path.join("configs_plot", f"{config_name}.py")

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file {config_file_path} not found")

    module_name = os.path.splitext(os.path.basename(config_file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, config_file_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    logdirs = cfg.logdirs
    datasets = cfg.datasets
    method_dirs = cfg.method_dirs
    method_names = cfg.method_names
    name = cfg.name
    rows = cfg.rows

    combined_dataframe = []

    for logdir, dataset in zip(logdirs, datasets):

        for bname in os.listdir(logdir):

            bpath = os.path.join(logdir, bname)

            if not os.path.isdir(bpath):
                continue

            files = list(glob.glob(os.path.join(bpath, "*.csv")))

            if len(files) == 0:
                continue

            data = pd.concat([pd.read_csv(x, index_col=0) 
                              for x in files], ignore_index=True)

            data = data[(data["metric"] == "Accuracy") & 
                        (data[ "split"] == "Validation")]
            
            if hasattr(cfg, "trial"):
                data = data[data['seed'].isin(range(cfg.trial))]

            def select_by_epoch(df):
                selected_row = df.loc[df["value"].idxmax()]
                return data[(data["epoch"] == selected_row["epoch"]) & 
                            (data[ "examples_per_class"] == 
                            selected_row["examples_per_class"])]

            best = data.groupby(["examples_per_class", "epoch"])
            best = best["value"].mean().to_frame('value').reset_index()
            best = best.groupby("examples_per_class").apply(
                select_by_epoch
            )

            best["method"] = bname
            best["dataset"] = dataset
            combined_dataframe.append(best)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    combined_dataframe = pd.concat(
        combined_dataframe, ignore_index=True)

    combined_dataframe = pd.concat([combined_dataframe[
        combined_dataframe['method'] == n] for n in method_dirs])
    
    color_palette = sns.color_palette(n_colors=len(method_dirs))

    legend_rows = int(math.ceil(len(method_names) / len(datasets)))
    columns = int(math.ceil(len(datasets) / rows))

    fig, axs = plt.subplots(
        rows, columns,
        figsize=(6 * columns, 4 * rows + (
            2.0 if legend_rows == 1 else
            2.5 if legend_rows == 2 else 3
        )))

    for i, dataset in enumerate(datasets):

        results = combined_dataframe
        if dataset not in ["all", "All", "Overall"]:
            results = results[results["dataset"] == dataset]
        
        examples_per_class_nunique = results.groupby(['seed', 'method'])['examples_per_class'].nunique().to_frame().reset_index()
        for _, (seed, method, examples_per_class) in examples_per_class_nunique.iterrows():
            if examples_per_class != 5:
                results.drop(results[(results['seed'] == seed) & (results['method'] == method)].index, inplace=True)

        results_nunique = results.groupby('method')['seed'].nunique()

        for exp_name in results_nunique.index:
            idx = method_dirs.index(exp_name)
            method_names[idx] = f"({results_nunique.loc[exp_name]}) {method_names[idx]}"

        axis = sns.lineplot(x="examples_per_class", y="value", hue="method", 
                            data=results, errorbar=('ci', 68),
                            linewidth=3, palette=color_palette,
                            marker='o', markersize=4,
                            ax=(
            axs[i // columns, i % columns] 
            if rows > 1 and len(datasets) > 1 
            else axs[i] if len(datasets) > 1 else axs
        ))

        axis.set_xticks(results["examples_per_class"].unique())

        if i == 0: handles, labels = axis.get_legend_handles_labels()
        axis.legend([],[], frameon=False)

        axis.set(xlabel=None)
        axis.set(ylabel=None)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')

        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        if i // columns == rows - 1:
            axis.set_xlabel("Examples Per Class", fontsize=16,
                            labelpad=12)

        axis.set_ylabel("Accuracy (Val)", fontsize=16,
                        labelpad=12)

        axis.set_title(dataset, fontsize=24, pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=1)

        if hasattr(cfg, "value_index"):
            value_index = cfg.value_index
        else:
            value_index = -1

        annotate_plot(axis, results, method_dirs[value_index])

    legend = fig.legend(handles, [x for x in method_names],
                        loc="lower center", prop={'size': 10}, 
                        ncol=min(len(method_names), len(datasets)))

    target_method_name = method_names[value_index]
    target_color = '#FF6666'

    for i, legend_object in enumerate(legend.legend_handles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(color_palette[i])

        if method_names[i] == target_method_name:
            legend.get_texts()[i].set_color(target_color)

    plt.tight_layout(pad=1.4)
    fig.subplots_adjust(hspace=0.3)

    fig.subplots_adjust(bottom=(
        0.25 if legend_rows == 1 else
        0.35 if legend_rows == 2 else 0.4
    ) / rows + 0.05)

    plt.savefig(f"{DEFAULT_VIZ_DIR}/{name}.png")