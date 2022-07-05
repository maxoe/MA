#!/usr/bin/python3

from cProfile import label, run
from math import log2
from multiprocessing.pool import RUN
from os.path import join
import subprocess
import os
import glob
from collections import OrderedDict
import importlib
import itertools
import pickle
import hashlib
import argparse
import time
import argparse

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import math
import re

from pandas.core.base import DataError

ggPlotColors = importlib.import_module("ggplot_colors").__dict__["ggColorSlice"]
style.use("ggplot")
plt.rcParams["lines.linewidth"] = 1


# all paths are relative to this directory
OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../plots/")
)

LATEX_GEN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gen/")
)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if not os.path.exists(LATEX_GEN_PATH):
    os.makedirs(LATEX_GEN_PATH)


REPO_PATH = os.path.normpath(
    r"C:\Users\Max\Documents\KIT\Masterarbeit\code\rust_truck_router"
)

DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
)

BIN_PATH = os.path.normpath(os.path.join(REPO_PATH, "target/release"))

GRAPH_PATH = os.path.normpath(
    r"C:\Users\Max\Documents\KIT\Masterarbeit\code",
)


def read_measurement(bin, *args, **kwargs):
    return pd.read_csv(os.path.join(DATA_PATH, bin + ".txt"), *args, **kwargs)


def write_plt(filename, graph):
    # path = os.path.join(os.path.join(OUTPUT_PATH, graph))
    path = os.path.join(os.path.join(OUTPUT_PATH))

    # if not os.path.exists(path):
    #     os.makedirs(path)

    plt.savefig(
        os.path.join(
            path,
            filename,
        ),
        dpi=1200,
    )
    plt.close()


def write_textfile(filename, content):
    path = os.path.join(OUTPUT_PATH, filename)

    with open(path, "w") as f:
        f.write(content)


def exp2_ticks(x):
    if x == 0:
        return "$0$"

    exponent = int(np.log2(x))
    coeff = x / 2 ** exponent

    if coeff == 1:
        return r"$2^{{ {:2d} }}$".format(exponent)

    return r"${:2.0f} \times 2^{{ {:2d} }}$".format(coeff, exponent)


def exp2_ticks_from_exponent(exponent):
    return r"$2^{{ {:2d} }}$".format(exponent)


def make_dijkstra_rank_tick_labels_from_exponent(ax_axis, exponents):
    ax_axis.set_ticklabels([exp2_ticks_from_exponent(int(i)) for i in exponents])


def make_dijkstra_rank_tick_labels_from_number(ax_axis, exponents):
    ax_axis.set_ticklabels([exp2_ticks(int(i)) for i in exponents])


def get_boxplot_outliers(df, by_column_name):
    q1 = df[by_column_name].quantile(0.25)
    q3 = df[by_column_name].quantile(0.75)
    iqr = q3 - q1
    filter = np.invert(
        (df[by_column_name] >= q1 - 1.5 * iqr) & (df[by_column_name] <= q3 + 1.5 * iqr)
    )
    return df.loc[filter]


def gen_avg_times():
    name_ger = "thesis_avg_all-csp"
    name_ger_2 = "thesis_avg_all-csp_2"
    name_eur = "thesis_avg_mid-csp"
    name_eur_2 = "thesis_avg_mid-csp_2"

    avg_ger = read_measurement(name_ger + "-" + "parking_ger_hgv")
    avg_ger_2 = read_measurement(name_ger_2 + "-" + "parking_ger_hgv")

    avg_eur = read_measurement(name_eur + "-" + "parking_europe_hgv")
    avg_eur_2 = read_measurement(name_eur_2 + "-" + "parking_europe_hgv")

    avg_ger = avg_ger.groupby(["algo"]).mean()
    avg_eur = avg_eur.groupby(["algo"]).mean()
    avg_ger_2 = avg_ger_2.groupby(["algo"]).mean()
    avg_eur_2 = avg_eur_2.groupby(["algo"]).mean()

    avg_all = avg_ger.join(avg_eur, rsuffix="_eur")[["time_ms", "time_ms_eur"]]
    avg_all_2 = avg_ger_2.join(avg_eur_2, rsuffix="_eur")[["time_ms", "time_ms_eur"]]

    avg_all["time_ms_2"] = avg_all_2["time_ms"]
    avg_all["time_ms_eur_2"] = avg_all_2["time_ms_eur"]
    # avg_ger["num_nodes_searched_2"] = avg_ger_2["num_nodes_searched"]

    order = [
        "dijkstra",
        "astar_chpot",
        "dijkstra_bidir",
        "astar_bidir_chpot",
        "core_ch",
        "core_ch_chpot",
    ]

    avg_all = avg_all.reindex(order)

    # HERE ORDER OF COLUMNS
    values = (
        # avg_all[["time_ms", "time_ms_2", "num_nodes_searched", "num_nodes_searched_2"]]
        avg_all[["time_ms", "time_ms_2", "time_ms_eur", "time_ms_eur_2"]]
        .to_numpy()
        .flatten()
    )

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_all-TEMPLATE.tex")) as f:
        template = f.read()

    if len(values) != template.count("@"):
        print(
            "ERROR: NUMBER OF VALUES ("
            + str(len(values))
            + ") DOES NOT EQUAL NUMBER OF BLANKS ("
            + str(template.count("@"))
            + ") FOR TEMPLATE "
            + name_ger
        )
        return

    i = 0

    value_to_string = lambda x: "-" if np.isnan(x) else "{:.2f}".format(round(x, 2))

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + value_to_string(values[len(values) - 1 - i])
            + template[m.end(0) :]
        )
        i += 1

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_all.tex"), "w") as f:
        f.write(template)


def gen_median_all_times():
    name_ger = "thesis_avg_all-csp"
    name_ger_2 = "thesis_avg_all-csp_2"
    name_eur = "thesis_avg_mid-csp"
    name_eur_2 = "thesis_avg_mid-csp_2"

    avg_ger = read_measurement(name_ger + "-" + "parking_ger_hgv")
    avg_ger_2 = read_measurement(name_ger_2 + "-" + "parking_ger_hgv")

    avg_eur = read_measurement(name_eur + "-" + "parking_europe_hgv")
    avg_eur_2 = read_measurement(name_eur_2 + "-" + "parking_europe_hgv")

    avg_ger = avg_ger.groupby(["algo"]).median()
    avg_eur = avg_eur.groupby(["algo"]).median()
    avg_ger_2 = avg_ger_2.groupby(["algo"]).median()
    avg_eur_2 = avg_eur_2.groupby(["algo"]).median()

    avg_all = avg_ger.join(avg_eur, rsuffix="_eur")[["time_ms", "time_ms_eur"]]
    avg_all_2 = avg_ger_2.join(avg_eur_2, rsuffix="_eur")[["time_ms", "time_ms_eur"]]

    avg_all["time_ms_2"] = avg_all_2["time_ms"]
    avg_all["time_ms_eur_2"] = avg_all_2["time_ms_eur"]
    # avg_ger["num_nodes_searched_2"] = avg_ger_2["num_nodes_searched"]

    order = [
        "dijkstra",
        "astar_chpot",
        "dijkstra_bidir",
        "astar_bidir_chpot",
        "core_ch",
        "core_ch_chpot",
    ]

    avg_all = avg_all.reindex(order)

    # HERE ORDER OF COLUMNS
    values = (
        # avg_all[["time_ms", "time_ms_2", "num_nodes_searched", "num_nodes_searched_2"]]
        avg_all[["time_ms", "time_ms_2", "time_ms_eur", "time_ms_eur_2"]]
        .to_numpy()
        .flatten()
    )

    with open(
        os.path.join(LATEX_GEN_PATH, "eval_median_running_times_all-TEMPLATE.tex")
    ) as f:
        template = f.read()

    if len(values) != template.count("@"):
        print(
            "ERROR: NUMBER OF VALUES ("
            + str(len(values))
            + ") DOES NOT EQUAL NUMBER OF BLANKS ("
            + str(template.count("@"))
            + ") FOR TEMPLATE "
            + name_ger
        )
        return

    i = 0

    value_to_string = lambda x: "-" if np.isnan(x) else "{:.2f}".format(round(x, 2))

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + value_to_string(values[len(values) - 1 - i])
            + template[m.end(0) :]
        )
        i += 1

    with open(
        os.path.join(LATEX_GEN_PATH, "eval_median_running_times_all.tex"), "w"
    ) as f:
        f.write(template)


def gen_all_times_no_path():
    name_eur = "thesis_avg_mid-csp"
    name_eur_2 = "thesis_avg_mid-csp_2"

    avg_eur = read_measurement(name_eur + "-" + "parking_europe_hgv")
    avg_eur_2 = read_measurement(name_eur_2 + "-" + "parking_europe_hgv")

    # avg_eur = avg_eur.groupby(["algo"]).median()
    # avg_eur_2 = avg_eur_2.groupby(["algo"]).median()

    # avg_eur["time_ms_eur_2"] = avg_eur_2["time_ms_eur"]
    # avg_ger["num_nodes_searched_2"] = avg_ger_2["num_nodes_searched"]

    longest = pd.DataFrame()
    longest_2 = pd.DataFrame()

    order = [
        "dijkstra",
        "astar_chpot",
        "dijkstra_bidir",
        "astar_bidir_chpot",
        "core_ch",
        "core_ch_chpot",
    ]

    for algo in order:
        # num_rows = avg_eur.loc[avg_eur["algo"] == algo].size[0] / 10
        # longest = longest.append(avg_eur.loc[avg_eur["algo"] == algo].sort_values("time_ms")[-num_rows:])
        subset = avg_eur.loc[avg_eur["algo"] == algo]
        longest = longest.append(subset.loc[np.isnan(subset["path_distance"])])

        subset_2 = avg_eur_2.loc[avg_eur_2["algo"] == algo]
        longest_2 = longest_2.append(subset_2.loc[np.isnan(subset_2["path_distance"])])

    longest["time_ms_2"] = longest_2["time_ms"]
    longest_mean = longest.groupby("algo").mean()
    longest_median = longest.groupby("algo").median()
    longest_mean = longest_mean.reindex(order)
    longest_median = longest_median.reindex(order)

    longest_all = pd.DataFrame()
    longest_all["mean"] = longest_mean["time_ms"]
    longest_all["median"] = longest_median["time_ms"]
    longest_all["mean_2"] = longest_mean["time_ms_2"]
    longest_all["median_2"] = longest_median["time_ms_2"]

    # HERE ORDER OF COLUMNS
    values = (
        # avg_eur[["time_ms", "time_ms_2", "num_nodes_searched", "num_nodes_searched_2"]]
        longest_all[["mean", "mean_2", "median", "median_2"]]
        .to_numpy()
        .flatten()
    )

    with open(
        os.path.join(LATEX_GEN_PATH, "eval_running_times_no_path-TEMPLATE.tex")
    ) as f:
        template = f.read()

    if len(values) != template.count("@"):
        print(
            "ERROR: NUMBER OF VALUES ("
            + str(len(values))
            + ") DOES NOT EQUAL NUMBER OF BLANKS ("
            + str(template.count("@"))
            + ") FOR TEMPLATE "
            + "running_times_no_path"
        )
        return

    i = 0

    value_to_string = lambda x: "-" if np.isnan(x) else "{:.2f}".format(round(x, 2))

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + value_to_string(values[len(values) - 1 - i])
            + template[m.end(0) :]
        )
        i += 1

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_no_path.tex"), "w") as f:
        f.write(template)


def gen_avg_opt():
    name = "thesis_avg_opt-csp"
    name_2 = "thesis_avg_opt-csp_2"

    avg = read_measurement(name + "-" + "parking_europe_hgv")
    avg_2 = read_measurement(name_2 + "-" + "parking_europe_hgv")

    avg = avg.groupby(["algo"]).mean()
    avg_2 = avg_2.groupby(["algo"]).mean()

    avg_all = avg.join(avg_2, rsuffix="_2")[["time_ms", "time_ms_2"]]

    # HERE ORDER OF ROWS
    order = [
        "no_bw_no_prune",
        "no_prune",
        "no_bw",
        "core_ch_chpot",
    ]

    avg_all = avg_all.reindex(order)

    # HERE ORDER OF COLUMNS
    values = avg_all[["time_ms", "time_ms_2"]].to_numpy().flatten()

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_opt-TEMPLATE.tex")) as f:
        template = f.read()

    if len(values) != template.count("@"):
        print(
            "ERROR: NUMBER OF VALUES ("
            + str(len(values))
            + ") DOES NOT EQUAL NUMBER OF BLANKS ("
            + str(template.count("@"))
            + ") FOR TEMPLATE "
            + name
        )
        return

    i = 0

    value_to_string = lambda x: "-" if np.isnan(x) else "{:.2f}".format(round(x, 2))

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + value_to_string(values[len(values) - 1 - i])
            + template[m.end(0) :]
        )
        i += 1

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_opt.tex"), "w") as f:
        f.write(template)


def plot_breaks_running_times(graph):
    problem = "csp"
    name = "thesis_rank_times_all-" + problem + "-" + graph
    name_2 = "thesis_avg_all-" + problem + "-" + graph

    queries_all = read_measurement(name).append(read_measurement(name_2))

    # plot only 2^10 and larger
    queries_all = queries_all.loc[queries_all["dijkstra_rank_exponent"] >= 10]

    colors = ggPlotColors(4)
    to_plot = [
        ("time_ms", "log"),
    ]

    # algos = ["astar_chpot", "astar_bidir_chpot", "core_ch", "core_ch_chpot"]
    algos = ["astar_chpot", "core_ch", "core_ch_chpot"]

    for algo in algos:
        queries = queries_all.loc[queries_all["algo"] == algo]
        # queries = queries.loc[queries["path_distance"] == -1]
        for (column_name, plot_scale) in to_plot:
            if column_name in queries.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                bp = queries.boxplot(ax=ax, by="path_number_pauses", column=column_name)

                bp.get_figure().gca().set_title("")
                fig.suptitle("")

                ax.set_xlabel("number of breaks")
                ax.set_ylabel(column_name)
                ax.set_yscale(plot_scale)

                if plot_scale == "linear":
                    ax.set_ylim(bottom=-0.1)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

                plt.title("breaks_running_times" + "-" + algo + ": " + graph)
                fig.tight_layout()
                write_plt(
                    "breaks_running_times" + "-" + algo + "-" + column_name + ".png",
                    graph,
                )


def plot_all_rank_times(problem, graph):
    name = "thesis_rank_times_all-" + problem + "-" + graph

    queries_all = read_measurement(name)

    # plot only 2^10 and larger
    queries_all = queries_all.loc[queries_all["dijkstra_rank_exponent"] >= 10]

    colors = ggPlotColors(4)
    to_plot = [
        ("time_ms", "log"),
    ]

    # algos = ["astar_chpot", "astar_bidir_chpot", "core_ch", "core_ch_chpot"]
    algos = ["astar_chpot", "core_ch", "core_ch_chpot"]

    for algo in algos:
        queries = queries_all.loc[queries_all["algo"] == algo]
        # queries = queries.loc[queries["path_distance"] == -1]
        for (column_name, plot_scale) in to_plot:
            if column_name in queries.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                bp = queries.boxplot(
                    ax=ax, by="dijkstra_rank_exponent", column=column_name
                )

                bp.get_figure().gca().set_title("")
                fig.suptitle("")

                ax.set_xlabel("Dijkstra Rank")
                ax.set_ylabel(column_name)
                ax.set_yscale(plot_scale)
                ax.set_ylabel("Running Time [ms]")

                if plot_scale == "linear":
                    ax.set_ylim(bottom=-0.1)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

                make_dijkstra_rank_tick_labels_from_exponent(
                    ax.xaxis, queries["dijkstra_rank_exponent"].unique()
                )

                # plt.title(name + "-" + algo + ": " + graph)
                plt.title("")
                fig.tight_layout()
                write_plt(name + "-" + algo + "-" + column_name + ".png", graph)


if __name__ == "__main__":
    gen_avg_times()
    gen_median_all_times()
    gen_avg_opt()

    gen_all_times_no_path()

    plot_breaks_running_times("parking_europe_hgv")
    plot_all_rank_times("csp", "parking_europe_hgv")
    plot_all_rank_times("csp_2", "parking_europe_hgv")
