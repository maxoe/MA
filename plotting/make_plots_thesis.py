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
    path = os.path.join(os.path.join(OUTPUT_PATH, graph))

    if not os.path.exists(path):
        os.makedirs(path)

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


def run_avg_all_times():
    name = "thesis_avg_all-csp"
    name_2 = "thesis_avg_all-csp_2"
    name_mid = "thesis_avg_mid-csp"
    name_mid_2 = "thesis_avg_mid-csp_2"

    avg_all = read_measurement(name + "-" + "parking_ger_hgv")
    avg_all_2 = read_measurement(name_2 + "-" + "parking_ger_hgv")

    avg_mid_eur = read_measurement(name_mid + "-" + "parking_eur_hgv")
    avg_mid_2_eur = read_measurement(name_mid_2 + "-" + "parking_eur_hgv")

    avg_all = avg_all.groupby(["algo"]).mean()
    avg_all_2 = avg_all.groupby(["algo"]).mean()

    avg_all["time_ms_2"] = avg_all_2["time_ms"]
    avg_all["num_nodes_searched_2"] = avg_all_2["num_nodes_searched"]

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
        avg_all[["time_ms", "time_ms_2"]]
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
            + name
        )
        return

    i = 0

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + "{:.2f}".format(round(values[len(values) - 1 - i], 2))
            + template[m.end(0) :]
        )
        i += 1

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_all.tex"), "w") as f:
        f.write(template)


def run_avg_fast_times(graph):
    name = "thesis_avg_fast-csp"
    name_2 = "thesis_avg_fast-csp_2"

    avg_fast = read_measurement(name + "-" + graph)
    avg_fast_2 = read_measurement(name_2 + "-" + graph)

    avg_fast = avg_fast.groupby(["algo"]).mean()
    avg_fast_2 = avg_fast.groupby(["algo"]).mean()

    avg_fast["time_ms_2"] = avg_fast_2["time_ms"]
    # avg_fast["num_nodes_searched_2"] = avg_fast_2["num_nodes_searched"]

    order = [
        # "astar_bidir_chpot",
        "core_ch",
        "core_ch_chpot",
    ]

    avg_fast = avg_fast.reindex(order)

    # HERE ORDER OF COLUMNS
    values = (
        # avg_fast[["time_ms", "time_ms_2", "num_nodes_searched", "num_nodes_searched_2"]]
        avg_fast[["time_ms", "time_ms_2"]]
        .to_numpy()
        .flatten()
    )

    with open(
        os.path.join(LATEX_GEN_PATH, "eval_running_times_fast-TEMPLATE.tex")
    ) as f:
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

    for m in reversed(list(re.finditer("@", template))):
        template = (
            template[: m.start(0)]
            + "{:.2f}".format(round(values[len(values) - 1 - i], 2))
            + template[m.end(0) :]
        )
        i += 1

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_fast.tex"), "w") as f:
        f.write(template)


def plot_query_insights(graph):
    # name = "thesis_avg_all-csp"
    name_2 = "thesis_avg_all-csp_2"
    chpot_corech_2 = read_measurement(name_2 + "-" + graph)["core_ch_chpot"]

    with open(os.path.join(LATEX_GEN_PATH, "eval_running_times_fast.tex"), "w") as f:
        f.write("c")


def plot_all_rank_times(problem, graph):
    name = "rank_times_chpot_corech-" + problem + "-" + graph

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

                ax.set_xlabel("dijkstra rank")
                ax.set_ylabel(column_name)
                ax.set_yscale(plot_scale)

                if plot_scale == "linear":
                    ax.set_ylim(bottom=-0.1)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

                make_dijkstra_rank_tick_labels_from_exponent(
                    ax.xaxis, queries["dijkstra_rank_exponent"].unique()
                )

                plt.title(name + "-" + algo + ": " + graph)
                fig.tight_layout()

                write_plt(name + "-" + algo + "-" + column_name + ".png", graph)


if __name__ == "__main__":
    run_avg_all_times()
    run_avg_fast_times("parking_europe_hgv")
